"""Desktop configuration and monitoring for two-agent competitive learning."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import secrets
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from unity_bridge.training_settings import GAME, AGENT, WEAPON, TOGGLES, defaults, validate, migrate
from unity_bridge.weapon_presets import NAMES,PRESETS
from copy import deepcopy
from unity_bridge.player_presets import PRESETS as PLAYER_PRESETS,MODES as PLAYER_MODES,KEYS as PLAYER_KEYS,LABEL_TO_ID,provenance
from unity_bridge.model_path_display import build_model_paths
from unity_bridge.trainer_state import save_state,load_state,checkpoint
from unity_bridge.storage import TRAINING_ROOT,MODEL_ROOT,MATCH_ROOT,training_directory,model_directory,resolve_path
import psutil
from unity_bridge.competitive_rewards import defaults as default_rewards,validate as validate_rewards,normalize as normalize_rewards
from unity_bridge.reward_editor import build_reward_editor

ROOT=Path(__file__).resolve().parents[1]
RESULTS=TRAINING_ROOT
STATE_PATH=ROOT/'local-results'/'trainer-state.json'

class Trainer:
    def __init__(self,root,*,simulator=False,state_path=None):
        self.root=root
        self.simulator=simulator
        self.state_path=Path(state_path) if state_path else ROOT/'local-results'/'match-simulator-state.json' if simulator else STATE_PATH
        root.title('커스텀 대전 시뮬레이터' if simulator else 'A/B 경쟁 학습기')
        root.geometry('1580x960' if simulator else f'{min(1880,root.winfo_screenwidth()-40)}x{min(1040,root.winfo_screenheight()-70)}')
        root.minsize(1300,780)
        self.process=None
        self.processes={};self.job_paths={};self.job_choice=tk.StringVar()
        self.save_job=None
        self.run=None
        self.match_options={}
        self.inputs={}
        self.reward_previews={}
        self.rewards={side:{key:{k:tk.StringVar(value=('고정' if k=='mode' else str(v))) for k,v in spec.items()} for key,spec in fields.items()} for side,fields in default_rewards().items()}
        self.weapon_types={side:tk.StringVar(value='기본형') for side in ('A','B')}
        self.custom_weapons={}
        self.changing_weapon=False
        self.changing_player=False
        self.player_types={side:tk.StringVar(value='커스텀') for side in ('A','B')}
        self.raw_players={side:tk.StringVar() for side in ('A','B')}
        self.player_info={side:tk.StringVar(value='원본 모델 8개 파라미터 직접 설정') for side in ('A','B')}
        self.learning_enabled={side:tk.BooleanVar(value=True) for side in ('A','B')}
        self.raw_selectors={}
        self.custom_players={}
        self.steps=tk.StringVar(value='20' if simulator else '1000')
        self.timeout=tk.StringVar(value='30')
        self.seed=tk.StringVar(value='421')
        self.compute_mode=tk.StringVar(value='CPU')
        self.workers=tk.StringVar(value='0')
        self.resume=tk.StringVar()
        self.status=tk.StringVar(value='설정을 확인한 뒤 학습을 시작하세요.')
        if simulator:
            shell=ttk.Frame(root);shell.pack(fill='both',expand=True)
            self.match_sidebar=ttk.Frame(shell,padding=(8,14,14,14),width=315);self.match_sidebar.pack(side='right',fill='y')
            frame=ttk.Frame(shell,padding=14);frame.pack(side='left',fill='both',expand=True)
        else:
            frame=ttk.Frame(root,padding=14);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='커스텀 대전 시뮬레이터' if simulator else '두 에이전트 경쟁 학습',font=('Malgun Gothic',18,'bold')).pack(anchor='w')
        ttk.Label(frame,text='지정한 A/B 모델로 반복 대결 · 모델 가중치 고정 · 경기 결과 및 3D 리플레이' if simulator else '발사 요청이 있을 때만 PPO 판단 · 승인 후 계획 유지 · 보류 또는 쿨다운에 걸리면 추적 후 재판단').pack(anchor='w')
        ttk.Label(frame,text='보상함수: 내 처치 +보상 / 상대 처치 −패배 비용 / 무승부 0 · 고정 또는 범위·추출 단위 설정').pack(anchor='w',pady=(3,10))
        controls=ttk.Frame(frame); controls.pack(fill='x')
        for label,var in [('반복 대결 경기 수' if simulator else '총 학습 경기 수',self.steps),('경기 타임아웃 (초)',self.timeout),('Seed (빈칸: 랜덤)',self.seed)]:
            ttk.Label(controls,text=label).pack(side='left',padx=4)
            ttk.Entry(controls,textvariable=var,width=12).pack(side='left')
        if not simulator:
            ttk.Label(controls,text='병렬 경기 수 (0: 자동)').pack(side='left',padx=4)
            ttk.Spinbox(controls,from_=0,to=64,textvariable=self.workers,width=5).pack(side='left')
        self.effective=tk.StringVar(); ttk.Label(frame,textvariable=self.effective).pack(anchor='w')
        self.steps.trace_add('write',lambda *_:self.update_target()); self.update_target()
        if simulator:self.build_match_controls(frame)
        else:
            resume=ttk.Frame(frame);resume.pack(fill='x',pady=6)
            ttk.Label(resume,text='이어 학습 폴더 (비워두면 새 학습)').pack(side='left')
            ttk.Entry(resume,textvariable=self.resume,width=55).pack(side='left',padx=5)
            ttk.Button(resume,text='선택',command=self.select_resume).pack(side='left')
            jobs=ttk.Frame(frame);jobs.pack(fill='x')
            ttk.Label(jobs,text='표시·중단할 학습 작업').pack(side='left')
            self.job_combo=ttk.Combobox(jobs,textvariable=self.job_choice,width=32,state='readonly');self.job_combo.pack(side='left',padx=5)
            self.job_combo.bind('<<ComboboxSelected>>',lambda e:self.monitor_job())
            ttk.Button(jobs,text='새 학습 준비',command=self.new_job).pack(side='left')
            ttk.Label(jobs,text='  연산').pack(side='left')
            ttk.Combobox(jobs,textvariable=self.compute_mode,values=['CPU','GPU 묶음 업데이트'],state='readonly',width=19).pack(side='left')
            ttk.Label(jobs,text='  GPU 선택 시 합산 65% 목표 · 단건 판단은 CPU').pack(side='left')
        initial=defaults()
        if simulator:
            panels=ttk.Frame(frame);panels.pack(fill='x',pady=8)
        else:
            parameter_area=ttk.Frame(frame);parameter_area.pack(fill='x',pady=8)
            parameter_canvas=tk.Canvas(parameter_area,height=460,highlightthickness=0)
            parameter_scroll=ttk.Scrollbar(parameter_area,orient='vertical',command=parameter_canvas.yview)
            parameter_canvas.configure(yscrollcommand=parameter_scroll.set);parameter_scroll.pack(side='right',fill='y');parameter_canvas.pack(fill='x',expand=True)
            panels=ttk.Frame(parameter_canvas);parameter_window=parameter_canvas.create_window((0,0),window=panels,anchor='nw')
            panels.bind('<Configure>',lambda e:parameter_canvas.configure(scrollregion=parameter_canvas.bbox('all')))
            parameter_canvas.bind('<Configure>',lambda e:parameter_canvas.itemconfigure(parameter_window,width=e.width))
            root.bind('<Configure>',lambda e:parameter_canvas.configure(height=max(230,min(520,e.height-500))) if e.widget is root else None,add='+')
        self.condition_panels=[]
        titles=['에이전트 A\n노이즈 / 보상 조건','에이전트 A\n추가 레이어 보상 조건','공통 게임 조건','에이전트 B\n추가 레이어 보상 조건','에이전트 B\n노이즈 / 보상 조건']
        for column,title in enumerate(titles):
            panels.columnconfigure(column,weight=3)
            panel=ttk.LabelFrame(panels,text=title,padding=8)
            panel.grid(row=0,column=column,sticky='nsew',padx=4)
            self.condition_panels.append(panel)
        def parameter_table(parent,specs,prefix=''):
            for c,title in enumerate(['방식','고정값','최소','최대','단위']):
                ttk.Label(parent,text=title).grid(row=0,column=c,padx=2,pady=(0,4))
                parent.columnconfigure(c,weight=1)
            for row,(key,label,value,lo,hi) in enumerate(specs):
                full=prefix+key;setting=initial[full]
                mode=tk.StringVar(value='고정' if setting['mode']=='fixed' else '범위')
                variables=[tk.StringVar(value=str(setting[k])) for k in ('value','min','max','step')]
                self.inputs[full]=(mode,*variables)
                short_label=label.replace('원본 조준 성공 조건 (PPO 보상 아님)','원본 모델 성공 보상 조건').replace('원본 조준 실패 조건 (PPO 비용 아님)','원본 모델 실패 비용 조건')
                limit=f'{lo}~{hi if hi is not None else "∞"}'
                ttk.Label(parent,text=f'{short_label}  ({limit})',wraplength=285).grid(row=row*2+1,column=0,columnspan=5,sticky='w',pady=(5,1))
                ttk.Combobox(parent,textvariable=mode,values=['고정','범위'],state='readonly',width=4).grid(row=row*2+2,column=0,sticky='ew',padx=1)
                entries=[]
                for c,variable in enumerate(variables,1):
                    entry=ttk.Entry(parent,textvariable=variable,width=5)
                    entry.grid(row=row*2+2,column=c,sticky='ew',padx=1);entries.append(entry)
                def toggle(*_,m=mode,e=entries):
                    e[0].configure(state='normal' if m.get()=='고정' else 'disabled')
                    for item in e[1:]:item.configure(state='disabled' if m.get()=='고정' else 'normal')
                mode.trace_add('write',toggle);toggle()
        for column,side in ((0,'A'),(4,'B')):
            panel=self.condition_panels[column]
            selector=ttk.Combobox(panel,textvariable=self.player_types[side],values=PLAYER_MODES,state='readonly',height=18)
            selector.pack(fill='x',pady=(0,4));selector.bind('<<ComboboxSelected>>',lambda event,s=side:self.choose_player(s))
            raw_selector=ttk.Combobox(panel,textvariable=self.raw_players[side],state='disabled',width=24)
            self.raw_selectors[side]=raw_selector
            ttk.Label(panel,textvariable=self.player_info[side],wraplength=275).pack(anchor='w',pady=(0,6))
            table=ttk.Frame(panel);table.pack(fill='x')
            parameter_table(table,AGENT,side+'.')
            for key in PLAYER_KEYS:
                for variable in self.inputs[side+'.'+key]:variable.trace_add('write',lambda *_,s=side:self.player_edited(s))
            ttk.Label(panel,text='원본 모델의 보상 조건 입력입니다.\n추가 PPO의 처치 보상과는 별도입니다.',wraplength=285).pack(anchor='w',pady=(10,0))
        for column,side in ((1,'A'),(3,'B')):
            panel=self.condition_panels[column]
            if not simulator:
                ttk.Checkbutton(panel,text=f'{side} 추가 PPO 학습 허용',variable=self.learning_enabled[side]).pack(anchor='w')
                ttk.Label(panel,text='해제: 대결 참여 · 모델 갱신 차단\n다음 학습 시작부터 적용',wraplength=250).pack(anchor='w',pady=(0,6))
            reward=ttk.LabelFrame(panel,text='경쟁 보상 설정',padding=8);reward.pack(fill='x')
            self.reward_previews[side]=build_reward_editor(reward,self.rewards[side]['alpha'])
            ttk.Separator(panel).pack(fill='x',pady=14)
            sub=ttk.LabelFrame(panel,text='서브 조건',padding=10);sub.pack(fill='both',expand=True)
            selector=ttk.Combobox(sub,textvariable=self.weapon_types[side],values=NAMES,state='readonly',width=16)
            selector.pack(fill='x',pady=(0,6))
            selector.bind('<<ComboboxSelected>>',lambda event,s=side:self.choose_weapon(s))
            weapon_table=ttk.Frame(sub);weapon_table.pack(fill='x')
            parameter_table(weapon_table,WEAPON,side+'.')
            for key,label,value,lo,hi in TOGGLES:
                setting=initial[side+'.'+key]
                values=tuple(tk.StringVar(value=str(v)) for v in ['고정',setting['value'],setting['min'],setting['max'],setting['step']])
                self.inputs[side+'.'+key]=values
                ttk.Checkbutton(sub,text=label,variable=values[1],onvalue='1',offvalue='0').pack(anchor='w',pady=3)
            for key,*_ in WEAPON+TOGGLES:
                for variable in self.inputs[side+'.'+key]:
                    variable.trace_add('write',lambda *_,s=side:self.weapon_edited(s))
        table=ttk.Frame(self.condition_panels[2]);table.pack(fill='x')
        parameter_table(table,GAME)
        ttk.Label(frame,text='경기마다 [최소 + k × 추출 단위]를 균등 추출합니다 (최대 이하). 경기 도중 고정 · A/B 독립 설정.').pack(anchor='w')
        buttons=ttk.Frame(frame);buttons.pack(fill='x',pady=10)
        actions=[('시나리오 저장',self.save_settings),('시나리오 불러오기',self.load_settings),('대결 시작',self.start),('중단',self.stop),('결과 폴더',self.open_folder)] if simulator else [('설정 저장',self.save_settings),('설정 불러오기',self.load_settings),('새 시나리오 학습',lambda:self.start(fresh=True)),('이어서 학습',self.start),('선택 작업 중단·저장',self.stop),('모델 폴더',self.open_model_folder),('기록 폴더',self.open_folder),('3D 리플레이 생성',self.replay),('대전 시뮬레이터 열기',self.custom_matches)]
        for label,fn in actions:
            ttk.Button(buttons,text=label,command=fn).pack(side='left',padx=3)
        if not simulator:
            self.refresh_model_paths=build_model_paths(frame,MODEL_ROOT,[('최신 A 모델','agent_a.zip'),('최신 B 모델','agent_b.zip')])
        ttk.Label(frame,textvariable=self.status,wraplength=1020).pack(anchor='w')
        self.bar=ttk.Progressbar(frame,maximum=100);self.bar.pack(fill='x',pady=5)
        if simulator:self.build_match_results(frame)
        else:
            tabs=ttk.Notebook(frame);tabs.pack(fill='both',expand=True)
            jobs_tab=ttk.Frame(tabs);log_tab=ttk.Frame(tabs);tabs.add(jobs_tab,text='모든 학습 작업');tabs.add(log_tab,text='선택 작업 로그')
            self.job_tree=ttk.Treeview(jobs_tab,columns=('name','status','progress','device'),show='headings',height=5)
            for key,title,width in [('name','학습 작업',400),('status','상태',110),('progress','완료 / 목표 경기',160),('device','연산',100)]:
                self.job_tree.heading(key,text=title);self.job_tree.column(key,width=width)
            self.job_tree.pack(fill='both',expand=True);self.job_tree.bind('<<TreeviewSelect>>',self.select_job_row)
            self.log=tk.Text(log_tab,height=8,font=('Consolas',10),state='disabled');self.log.pack(fill='both',expand=True)
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.restore_draft()
        for var in [self.steps,self.timeout,self.seed,self.resume,self.compute_mode,self.workers,*self.learning_enabled.values()]+[v for r in self.rewards.values() for fields in r.values() for v in fields.values()]+[v for values in self.inputs.values() for v in values]:
            var.trace_add('write',self.schedule_save)
        self.poll()

    def player_raw(self,side):
        return {key:[v.get() for v in self.inputs[side+'.'+key]] for key in PLAYER_KEYS}

    def player_edited(self,side):
        if self.changing_player or self.changing_weapon:return
        self.player_types[side].set('커스텀');self.raw_selectors[side].configure(state='disabled')
        self.player_info[side].set('원본 모델 8개 파라미터 직접 설정')
        self.custom_players[side]=self.player_raw(side)

    def selected_player_id(self,side):
        return LABEL_TO_ID.get(self.player_types[side].get())

    def refresh_player_selector(self,side,preset_id=None):
        self.raw_selectors[side].configure(state='disabled')
        self.raw_players[side].set('')

    def choose_player(self,side):
        self.custom_players.setdefault(side,self.player_raw(side))
        self.refresh_player_selector(side)
        preset_id=self.selected_player_id(side)
        if preset_id:
            preset=PLAYER_PRESETS[preset_id]
            raw={key:['고정',str(value),str(value),str(value),self.inputs[side+'.'+key][4].get()] for key,value in preset['parameters'].items()}
        else:
            raw=self.custom_players.setdefault(side,self.player_raw(side))
        self.changing_player=True
        try:
            for key,values in raw.items():
                for variable,value in zip(self.inputs[side+'.'+key],values):variable.set(value)
        finally:self.changing_player=False
        self.update_player_info(side);self.persist()

    def update_player_info(self,side):
        preset_id=self.selected_player_id(side)
        if not preset_id:self.player_info[side].set('원본 모델 8개 파라미터 직접 설정');return
        p=PLAYER_PRESETS[preset_id]
        if p['kind']=='group_mean':self.player_info[side].set(f"참가자 {p['n_players']}명 · 각 4조건 평균")
        else:self.player_info[side].set(f"평균 없이 CSV 원값 · player_index {p['player_index']}\n감도·색상은 추정 당시 조건")

    def player_selection(self):
        return {side:provenance(self.selected_player_id(side)) for side in ('A','B')}

    def restore_player_selection(self,selections):
        for side in ('A','B'):
            p=(selections.get(side) or {});preset_id=p.get('id')
            if preset_id in PLAYER_PRESETS:
                expected=PLAYER_PRESETS[preset_id]
                try:valid=all(self.inputs[side+'.'+key][0].get()=='고정' and float(self.inputs[side+'.'+key][1].get())==value for key,value in expected['parameters'].items())
                except ValueError:valid=False
                if not valid:preset_id=None
            else:preset_id=None
            if preset_id:
                self.player_types[side].set(PLAYER_PRESETS[preset_id]['label'])
            else:self.player_types[side].set('커스텀')
            self.refresh_player_selector(side,preset_id);self.update_player_info(side)

    def reward_config(self):
        config={side:{key:{k:('fixed' if v.get()=='고정' else 'uniform') if k=='mode' else float(v.get()) for k,v in fields.items()} for key,fields in rewards.items()} for side,rewards in self.rewards.items()}
        validate_rewards(config);return config

    def apply_rewards(self,config):
        for side,rewards in normalize_rewards(config).items():
            for key,fields in rewards.items():
                for k,value in fields.items():
                    if k=='mode':value='고정' if value in ('fixed','고정') else '범위'
                    self.rewards[side][key][k].set(str(value))

    def weapon_raw(self,side):
        return {key:[v.get() for v in self.inputs[side+'.'+key]] for key,*_ in WEAPON+TOGGLES}

    def weapon_edited(self,side):
        if self.changing_weapon:return
        self.weapon_types[side].set('커스텀')
        self.custom_weapons[side]=self.weapon_raw(side)

    def choose_weapon(self,side):
        name=self.weapon_types[side].get()
        if name=='커스텀':
            raw=self.custom_weapons.get(side)
            if raw is None:self.custom_weapons[side]=self.weapon_raw(side);self.persist();return
        else:
            preset=PRESETS[name]
            if preset is None:
                self.status.set(f'{name} 수치가 아직 지정되지 않았습니다. 직접 편집하면 커스텀으로 전환됩니다.')
                self.persist();return
            raw={key:['고정' if s['mode']=='fixed' else '범위',*[str(s[k]) for k in ('value','min','max','step')]] for key,s in preset.items()}
        self.changing_weapon=True
        try:
            for key,values in raw.items():
                for variable,value in zip(self.inputs[side+'.'+key],values):variable.set(value)
        finally:self.changing_weapon=False
        self.persist()

    def draft(self):
        # Preserve unfinished numeric edits too; validation is only for starting.
        return dict(episodes=self.steps.get(),timeout=self.timeout.get(),seed=self.seed.get(),
            resume=self.resume.get(),lastRun=str(self.run) if self.run else None,computeMode=self.compute_mode.get(),workers=self.workers.get(),
            inputs={key:[v.get() for v in values] for key,values in self.inputs.items()},
            weaponTypes={side:v.get() for side,v in self.weapon_types.items()},customWeapons=self.custom_weapons,
            playerPresets=self.player_selection(),customPlayers=self.custom_players,learningEnabled=self.learning_config(),
            rewardConfig={side:{key:{k:('fixed' if v.get()=='고정' else 'uniform') if k=='mode' else v.get() for k,v in fields.items()} for key,fields in rewards.items()} for side,rewards in self.rewards.items()},matchOptions=self.match_options)

    def persist(self):
        self.save_job=None
        try:save_state(self.state_path,self.draft())
        except OSError as error:self.status.set(f'설정 자동 저장 실패: {error}')

    def schedule_save(self,*_):
        if self.save_job is not None:self.root.after_cancel(self.save_job)
        self.save_job=self.root.after(300,self.persist)

    def restore_draft(self):
        state=load_state(self.state_path)
        if not state:return
        self.compute_mode.set(state.get('computeMode','CPU'))
        self.workers.set(str(state.get('workers',0)))
        self.apply_learning_config(state.get('learningEnabled',{}))
        self.match_options=state.get('matchOptions',{})
        self.apply_rewards(state.get('rewardConfig',default_rewards()))
        self.changing_weapon=True
        draft_inputs=dict(state.get('inputs',{}))
        for key in ('minDamage','maxDamage'):
            if key in draft_inputs:
                for side in ('A','B'):draft_inputs.setdefault(side+'.'+key,draft_inputs[key])
        for key,values in draft_inputs.items():
            if key in self.inputs:
                for variable,value in zip(self.inputs[key],values):variable.set(value)
        self.custom_weapons=state.get('customWeapons',{})
        for side,value in state.get('weaponTypes',{}).items():
            if side in self.weapon_types and value in NAMES:self.weapon_types[side].set(value)
        for side in ('A','B'):
            if side not in state.get('weaponTypes',{}):self.weapon_types[side].set('커스텀')
        self.changing_weapon=False
        self.custom_players=state.get('customPlayers',{})
        self.restore_player_selection(state.get('playerPresets',{}))
        for key,variable in [('episodes',self.steps),('timeout',self.timeout),('seed',self.seed),('resume',self.resume)]:
            if key in state:variable.set(state[key])
        if state.get('lastRun') and resolve_path(state['lastRun']).is_dir():self.run=resolve_path(state['lastRun'])

    def close(self):
        if self.save_job is not None:self.root.after_cancel(self.save_job)
        self.persist();self.root.destroy()

    def running(self):
        if self.process is not None and self.process.poll() is None:return True
        if not self.run:return False
        data=load_state(self.run/'progress.json') or {}
        if data.get('status')!='running' or not data.get('pid'):return False
        try:
            args=psutil.Process(data['pid']).cmdline()
            return 'unity_bridge.train_competitive' in args and str(self.run) in args
        except (psutil.Error,OSError):return False

    def restore_run(self,folder):
        folder=training_directory(folder)
        metadata=load_state(Path(folder)/'training.json')
        if not metadata:raise ValueError('학습 기록이 없는 폴더입니다.')
        self.apply_learning_config(metadata.get('learningEnabled',{}))
        source,saved=checkpoint(folder)
        params=migrate(metadata['settings']);validate(params)
        self.apply_rewards(metadata.get('rewardConfig',default_rewards()))
        self.changing_weapon=True
        for key,s in params.items():
            values=self.inputs[key];values[0].set('고정' if s['mode']=='fixed' else '범위')
            for var,k in zip(values[1:],('value','min','max','step')):var.set(str(s[k]))
        self.changing_weapon=False
        for side in ('A','B'):
            self.weapon_types[side].set('커스텀');self.custom_weapons[side]=self.weapon_raw(side)
        self.restore_player_selection(metadata.get('playerPresets',{}))
        self.steps.set(str(metadata['targetEpisodes']))
        self.timeout.set(str(metadata['timeoutSeconds']));self.seed.set(str(metadata['seed']))
        self.resume.set(str(folder));self.run=Path(folder);self.persist()
        self.status.set(f'기존 목표 {metadata["targetEpisodes"]}경기 유지 · 저장된 {saved["episodes"]}경기부터 재개 가능')

    def update_target(self):
        try:
            n=int(self.steps.get());self.effective.set(f'실행 목표: A/B가 함께 치르는 총 {n:,}경기. 이어 학습은 누적 경기 수 기준입니다.')
        except ValueError:self.effective.set('양의 정수 스텝 수를 입력하세요.')

    def settings(self):
        result={key:dict(mode='fixed' if vs[0].get()=='고정' else 'uniform',**{k:float(v.get()) for k,v in zip(('value','min','max','step'),vs[1:])}) for key,vs in self.inputs.items()}
        validate(result);return result

    def seed_value(self,*,resolve=False):
        text=self.seed.get().strip()
        if not text:return secrets.randbelow(2**31) if resolve else None
        value=int(text)
        if value<0:raise ValueError('Seed는 0 이상의 정수 또는 빈칸이어야 합니다.')
        return value

    def learning_config(self):
        return {side:var.get() for side,var in self.learning_enabled.items()}

    def apply_learning_config(self,config):
        for side,var in self.learning_enabled.items():var.set(config.get(side,True))

    def save_settings(self):
        try:
            data=dict(episodes=int(self.steps.get()),timeoutSeconds=float(self.timeout.get()),seed=self.seed_value(),parameters=self.settings(),rewardConfig=self.reward_config(),playerPresets=self.player_selection(),learningEnabled=self.learning_config(),computeMode=self.compute_mode.get(),workers=int(self.workers.get()))
            path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
            if path:Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')
        except Exception as e:messagebox.showerror('설정 오류',str(e))

    def load_settings(self):
        path=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:return
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'));params=migrate(data.get('parameters',data));validate(params)
            for key,s in params.items():
                values=self.inputs[key];values[0].set('고정' if s['mode']=='fixed' else '범위')
                for var,k in zip(values[1:],('value','min','max','step')):var.set(str(s[k]))
            self.apply_rewards(data.get('rewardConfig',default_rewards()))
            self.apply_learning_config(data.get('learningEnabled',{}))
            self.compute_mode.set(data.get('computeMode','CPU'))
            self.workers.set(str(data.get('workers',0)))
            self.restore_player_selection(data.get('playerPresets',{}))
            if 'episodes' in data:self.steps.set(str(data['episodes']))
            if 'timeoutSeconds' in data:self.timeout.set(str(data['timeoutSeconds']))
            if 'seed' in data:self.seed.set('' if data['seed'] is None else str(data['seed']))
        except Exception as e:messagebox.showerror('설정 오류',str(e))

    def select_resume(self):
        path=filedialog.askdirectory(initialdir=MODEL_ROOT if MODEL_ROOT.exists() else RESULTS if RESULTS.exists() else ROOT)
        if path:
            try:self.restore_run(path)
            except Exception as error:messagebox.showerror('불러오기 실패',str(error))

    def launch(self,module,args,log):
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as stream:
            return subprocess.Popen([str(Path(sys.executable).with_name('python.exe')),'-u','-m',module,*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

    def refresh_jobs(self):
        paths={p.name:p for p in RESULTS.glob('*') if p.is_dir() and (p/'training.json').is_file()}
        if self.run:paths[self.run.name]=self.run
        self.job_paths=paths;self.job_combo.configure(values=tuple(sorted(paths,reverse=True)))
        if self.run:self.job_choice.set(self.run.name)
        for name,path in paths.items():
            progress=load_state(path/'progress.json') or {};metadata=load_state(path/'training.json') or {}
            state=progress.get('status',metadata.get('status','starting'))
            if state=='running':
                try:
                    args=psutil.Process(progress['pid']).cmdline()
                    if 'unity_bridge.train_competitive' not in args:state='실행 종료'
                except (psutil.Error,KeyError):state='시작 중' if str(path) in self.processes and self.processes[str(path)].poll() is None else '실행 종료'
            count=f'{progress.get("episodes",0):,} / {progress.get("target",metadata.get("targetEpisodes",0)):,}'
            values=(name,state,count,progress.get('device',metadata.get('device','cpu')))
            if self.job_tree.exists(name):self.job_tree.item(name,values=values)
            else:self.job_tree.insert('','end',iid=name,values=values)

    def select_job_row(self,event=None):
        selection=self.job_tree.selection()
        if selection:self.job_choice.set(selection[0]);self.monitor_job()

    def monitor_job(self):
        folder=self.job_paths.get(self.job_choice.get())
        if folder:
            self.run=folder;self.process=self.processes.get(str(folder));self.persist()

    def new_job(self):
        self.resume.set('');self.run=None;self.process=None;self.job_choice.set('');self.bar['value']=0
        self.status.set('새 시나리오의 조건을 설정하고 학습 시작을 누르세요. 실행 중인 다른 작업은 계속 진행됩니다.');self.persist()

    def start(self,*,fresh=False):
        try:
            for side,selected in self.weapon_types.items():
                if selected.get() in PRESETS and PRESETS[selected.get()] is None:
                    raise ValueError(f'{side}: {selected.get()}의 수치를 먼저 지정하거나 커스텀으로 편집하세요.')
            n=int(self.steps.get());seed=self.seed_value(resolve=True);settings=self.settings();rewards=self.reward_config()
            if n<1 or seed<0:raise ValueError('스텝은 양수, seed는 0 이상이어야 합니다.')
            workers=int(self.workers.get())
            if workers<0:raise ValueError('병렬 경기 수는 0 이상이어야 합니다.')
            resume='' if fresh else self.resume.get().strip()
            if resume:resume=str(training_directory(resume))
            if resume:
                _,saved=checkpoint(resume)
                if n<=saved['episodes']:raise ValueError(f'이미 {saved["episodes"]}경기를 완료했습니다. 더 학습하려면 총 경기 목표를 늘리세요.')
            folder=RESULTS/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
            path=folder/'settings.json';path.write_text(json.dumps(dict(parameters=settings,playerPresets=self.player_selection(),learningEnabled=self.learning_config(),seed=seed),indent=2),encoding='utf-8')
            reward_path=folder/'rewards.json';reward_path.write_text(json.dumps(rewards,indent=2),encoding='utf-8')
            timeout=float(self.timeout.get())
            if not 0<timeout<1e6:raise ValueError('타임아웃은 양수여야 합니다.')
            args=['--episodes',str(n),'--timeout',str(timeout),'--seed',str(seed),'--settings',str(path),'--rewards',str(reward_path),'--output',str(folder)]
            args+=['--device','cpu' if self.compute_mode.get()=='CPU' else 'cuda:0','--workers',str(workers)]
            if resume:args+=['--resume',resume]
            self.run=folder;self.process=self.launch('unity_bridge.train_competitive',args,folder/'training.log')
            self.processes[str(folder)]=self.process;self.refresh_jobs()
            self.persist()
            self.status.set('경쟁 학습 시작 중… · '+' / '.join(f'{side} {"학습" if enabled else "고정"}' for side,enabled in self.learning_config().items()))
        except Exception as e:messagebox.showerror('시작 실패',str(e))

    def stop(self):
        if self.run and self.running():
            (self.run/'stop.request').touch();self.status.set('중단 후 양측 체크포인트 저장 요청됨')

    def open_folder(self):
        if self.run:os.startfile(self.run)

    def open_model_folder(self):
        folder=model_directory(self.run) if self.run else MODEL_ROOT
        if folder.exists():os.startfile(folder)

    def custom_matches(self):
        try:
            self.persist()
            self.launch('unity_bridge.match_simulator',[],ROOT/'local-results'/'logs'/f'match-simulator-launch-{uuid.uuid4().hex}.log')
        except Exception as error:messagebox.showerror('반복 대결 설정 오류',str(error))

    def replay(self):
        folder=self.run
        if folder is None:
            path=filedialog.askdirectory(initialdir=RESULTS if RESULTS.exists() else ROOT)
            if not path:return
            folder=Path(path)
        try:checkpoint(folder)
        except ValueError:
            messagebox.showinfo('체크포인트 대기','양측 첫 학습 저장 후 리플레이를 생성할 수 있습니다.');return
        destination=MATCH_ROOT/(f'training-{folder.name}-'+datetime.now().strftime('%Y%m%d-%H%M%S'));destination.mkdir(parents=True)
        self.launch('unity_bridge.replay_competitive',['--run',str(folder),'--output',str(destination)],destination/'replay.log')
        self.status.set(f'현재 양측 모델의 경기를 별도로 기록 중: {destination}')
        os.startfile(destination)

    def poll(self):
        self.refresh_model_paths(self.run)
        self.refresh_jobs()
        if self.run:
            try:
                p=json.loads((self.run/'progress.json').read_text(encoding='utf-8-sig'));count,target=p['episodes'],p['target']
                state=p['status']
                if state=='running' and not self.running():state='실행 종료 · 저장된 모델에서 이어 학습 가능'
                modes=' / '.join(f'{side} {"학습" if enabled else "고정"}' for side,enabled in p.get('learningEnabled',{'A':True,'B':True}).items())
                seed_info=f' | Seed {p["seed"]}' if 'seed' in p else ''
                gpu=p.get('gpuBudget') or {};device_info=f' | {p.get("device","cpu")} · 병렬 {p.get("workers",1)}경기'
                if gpu:device_info+=f' · GPU 전체 {gpu["measuredGpuPercent"]:g}% / 목표 {gpu["targetPercent"]:g}%'
                self.status.set(f'{state} | 총 {count:,} / {target:,} 경기 | {p["elapsedSeconds"]:.0f}초 | {modes}{seed_info}{device_info}')
                self.bar['value']=100*count/target
            except (OSError,ValueError,KeyError):
                if self.process and self.process.poll() is not None:self.status.set('학습 프로세스 종료 · 로그를 확인하세요.')
            try:
                text=(self.run/'training.log').read_text(encoding='utf-8',errors='replace')[-10000:]
                self.log.configure(state='normal');self.log.delete('1.0','end');self.log.insert('end',text);self.log.see('end');self.log.configure(state='disabled')
            except OSError:pass
        self.root.after(1000,self.poll)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--matches',action='store_true')
    parser.add_argument('--instance',type=int,choices=(1,2,3),help='Independent persistent settings for each trainer window')
    args=parser.parse_args()
    if args.matches:
        from unity_bridge.match_simulator import main as simulator_main
        simulator_main();return
    state_path=ROOT/'local-results'/f'trainer-state-{args.instance}.json' if args.instance else None
    if state_path and not state_path.exists():
        initial=load_state(STATE_PATH) or {};initial.update(resume='',lastRun=None)
        save_state(state_path,initial)
    root=tk.Tk();trainer=Trainer(root,state_path=state_path)
    root.mainloop()

if __name__=='__main__':main()
