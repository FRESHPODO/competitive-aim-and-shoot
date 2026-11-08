"""Independent custom-duel app with the same visible parameter editor as training."""
import json,math
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import psutil
from unity_bridge.decision_trainer import Trainer,ROOT,STATE_PATH
from unity_bridge.training_settings import migrate,validate
from unity_bridge.reward_generation.rewards import normalize,validate as validate_rewards
from unity_bridge.reward_editor import build_reward_editor
from unity_bridge.trainer_state import load_state,checkpoint
from unity_bridge.storage import TRAINED_MODELS,MATCH_ROOT,resolve_path

class MatchSimulator(Trainer):
    def __init__(self,root,*,state_path=None):
        self.rows=[];self.output=None;self.toggled=set();self.pending_view=None;self.recovery=None
        super().__init__(root,simulator=True,state_path=state_path)
        root.geometry(f'{min(1880,root.winfo_screenwidth()-40)}x{min(960,root.winfo_screenheight()-70)}')
        if not self.output:self.status.set('화면의 A/B·보상·무기·게임 조건으로 반복 대결합니다. 모델은 학습하지 않습니다.')
    def update_target(self):
        self.effective.set('두 모델의 가중치를 고정하고 입력한 횟수만큼 대결합니다.')
    def build_match_controls(self,frame):
        self.models={side:tk.StringVar() for side in ('a','b')}
        row=ttk.Frame(frame);row.pack(fill='x',pady=4)
        for side in ('a','b'):
            ttk.Label(row,text=f'{side.upper()} 모델').pack(side='left',padx=4)
            ttk.Entry(row,textvariable=self.models[side],width=48).pack(side='left',fill='x',expand=True)
            ttk.Button(row,text='선택',command=lambda s=side:self.select_model(s)).pack(side='left',padx=4)
        factors=ttk.Frame(frame);factors.pack(fill='x',pady=4)
        self.match_factors={side:{} for side in 'AB'}
        for side in 'AB':
            panel=ttk.LabelFrame(factors,text=f'{side} · 피해·Critical 모델 보상');panel.pack(side='left',fill='x',expand=True,padx=4)
            for key,label in [('damageFactor','누적 피해 계수 (0~1)'),('criticalFactor','Critical 처치 보상 (0~1)')]:
                editor=ttk.Frame(panel);editor.pack(side='left',fill='x',expand=True,padx=4)
                fields={k:tk.StringVar(value=v) for k,v in dict(mode='고정',value='0',min='0',max='1',step='0.1').items()}
                self.match_factors[side][key]=fields
                build_reward_editor(editor,fields,factor_label=label)
                for var in fields.values():var.trace_add('write',self.schedule_save)
        options=ttk.Frame(frame);options.pack(fill='x')
        self.resample=tk.BooleanVar(value=False)
        ttk.Label(options,text='모든 경기의 행동 기록 자동 저장 · ').pack(side='left')
        ttk.Checkbutton(options,text='매 경기 범위 조건 재추출 (끄면 한 번 추출한 조건으로 반복)',variable=self.resample).pack(side='left')
        for var in [*self.models.values(),self.resample]:var.trace_add('write',self.schedule_save)
    def build_match_results(self,frame):
        frame=self.match_sidebar
        ttk.Label(frame,text='대결 결과',font=('Malgun Gothic',14,'bold')).pack(anchor='w',pady=(0,8))
        self.winrate=tk.StringVar(value='A 승률 — / B 승률 —\n전체 완료 경기 기준 · 무승부 포함')
        ttk.Label(frame,textvariable=self.winrate,font=('Malgun Gothic',11,'bold'),wraplength=285).pack(fill='x',pady=(0,8))
        self.view_button=ttk.Button(frame,text='토글 경기 3D 보기 (0)',command=self.replay,state='disabled');self.view_button.pack(fill='x',pady=(0,4))
        toggles=ttk.Frame(frame);toggles.pack(fill='x')
        ttk.Button(toggles,text='전체 토글',command=lambda:self.toggle_all(True)).pack(side='left',expand=True,fill='x')
        ttk.Button(toggles,text='전체 해제',command=lambda:self.toggle_all(False)).pack(side='left',expand=True,fill='x')
        self.view_notice=tk.StringVar(value='행 더블클릭: 해당 경기 반복 재생\n보기 토글: 선택 경기들을 순서대로 반복')
        ttk.Label(frame,textvariable=self.view_notice,wraplength=285).pack(fill='x',pady=6)
        tabs=ttk.Notebook(frame);tabs.pack(fill='both',expand=True)
        results=ttk.Frame(tabs);logs=ttk.Frame(tabs);tabs.add(results,text='경기 결과');tabs.add(logs,text='실행 로그')
        cols=('toggle','round','winner','seconds','shotsA','shotsB','holdA','holdB','replay')
        self.tree=ttk.Treeview(results,columns=cols,displaycolumns=('toggle','round','winner','seconds','replay'),show='headings',height=18)
        for key,label in zip(cols,('보기','경기','결과','시간(초)','A 발사','B 발사','A 서브 보류','B 서브 보류','기록')):
            self.tree.heading(key,text=label);self.tree.column(key,width=62,minwidth=40,anchor='center',stretch=True)
        scroll=ttk.Scrollbar(results,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.tree.pack(fill='both',expand=True)
        self.tree.column('toggle',width=36,minwidth=32,stretch=False)
        self.tree.bind('<Button-1>',self.click_toggle)
        self.tree.bind('<Double-1>',self.double_click)
        self.row_details=tk.StringVar(value='경기를 선택하면 상세 결과가 표시됩니다.')
        ttk.Label(frame,textvariable=self.row_details,wraplength=285,justify='left').pack(fill='x',pady=10)
        self.tree.bind('<<TreeviewSelect>>',lambda e:self.show_round_details())
        self.log=tk.Text(logs,width=34,height=6,font=('Consolas',9),state='disabled');self.log.pack(fill='both',expand=True)
    def show_round_details(self):
        selected=self.tree.selection()
        if not selected:return
        r=self.rows[int(selected[0])]
        self.row_details.set(f"경기 {r['round']} · {r['winner']+' 승리' if r['winner'] else '무승부'}\n시간 {r['timeMs']/1000:.2f}초\nA 발사 {r['shots'][0]} / B 발사 {r['shots'][1]}\nA 명중 {r['hits'][0]} / B 명중 {r['hits'][1]}\nA 피해 {r['damage'][0]:.1f} / B 피해 {r['damage'][1]:.1f}\nA 보류 {r['blocked'][0]} / B 보류 {r['blocked'][1]}")
    def select_model(self,side):
        path=filedialog.askopenfilename(parent=self.root,initialdir=TRAINED_MODELS,filetypes=[('PPO checkpoint','*.zip')])
        if path:self.models[side].set(path)
    def draft(self):
        d=super().draft();d['duel']={**{side:var.get() for side,var in self.models.items()},'resample':self.resample.get()}
        for side,options in self.match_factors.items():
            for key,fields in options.items():
                spec={k:v.get() for k,v in fields.items()}
                spec['mode']='fixed' if spec['mode']=='고정' else 'uniform'
                d['rewardConfig'][side][key]=spec
        return d
    def apply_rewards(self,config):
        super().apply_rewards(config)
        rewards=normalize(config)
        for side,options in self.match_factors.items():
            for key,fields in options.items():
                spec=rewards[side][key]
                for name,var in fields.items():
                    var.set(('고정' if spec[name]=='fixed' else '범위') if name=='mode' else str(spec[name]))
    def reward_config(self):
        config=normalize(self.draft()['rewardConfig'])
        validate_rewards(config)
        return config
    def restore_draft(self):
        own=load_state(self.state_path)
        if own:
            super().restore_draft()
            for side in ('a','b'):
                value=own.get('duel',{}).get(side,'');self.models[side].set(str(resolve_path(value)) if value else '')
            self.resample.set(own.get('duel',{}).get('resample',False))
            self.output=self.run
        else:
            previous=load_state(STATE_PATH) or {}
            self.changing_player=True;self.changing_weapon=True
            for key,values in previous.get('inputs',{}).items():
                if key in self.inputs:
                    for var,value in zip(self.inputs[key],values):var.set(value)
            self.apply_rewards(previous.get('rewardConfig',{}));self.timeout.set(previous.get('timeout','30'));self.seed.set(previous.get('seed','421'))
            self.changing_player=False;self.changing_weapon=False
            self.custom_players=previous.get('customPlayers',{});self.custom_weapons=previous.get('customWeapons',{})
            self.restore_player_selection(previous.get('playerPresets',{}))
            for side,value in previous.get('weaponTypes',{}).items():
                if side in self.weapon_types:self.weapon_types[side].set(value)
        if any(not v.get() for v in self.models.values()):
            previous=load_state(STATE_PATH) or {};source=previous.get('lastRun') or previous.get('resume')
            if source:
                try:
                    folder,_=checkpoint(source)
                    for side,var in self.models.items():
                        if not var.get():var.set(str((folder/f'agent_{side}.zip').resolve()))
                except (ValueError,OSError):pass
    def scenario(self):
        return dict(parameters=self.settings(),rewardConfig=self.reward_config(),playerPresets=self.player_selection(),
            rounds=int(self.steps.get()),timeoutSeconds=float(self.timeout.get()),seed=self.seed_value(),
            models={side:v.get() for side,v in self.models.items()},recordAllReplays=True,resample=self.resample.get())
    def apply_document(self,data):
        params=migrate(data.get('parameters',data));validate(params)
        rewards=normalize(data.get('rewardConfig',{}));validate_rewards(rewards)
        self.changing_player=True;self.changing_weapon=True
        try:
            for key,spec in params.items():
                variables=self.inputs[key];variables[0].set('고정' if spec['mode']=='fixed' else '범위')
                for var,key2 in zip(variables[1:],('value','min','max','step')):var.set(str(spec[key2]))
            self.apply_rewards(rewards)
        finally:self.changing_player=False;self.changing_weapon=False
        self.restore_player_selection(data.get('playerPresets',{}))
        for side in ('A','B'):
            self.weapon_types[side].set('커스텀');self.custom_weapons[side]=self.weapon_raw(side)
        for key,var in [('rounds',self.steps),('timeoutSeconds',self.timeout),('seed',self.seed)]:
            if key in data:var.set('' if key=='seed' and data[key] is None else str(data[key]))
        if 'resample' in data:self.resample.set(data['resample'])
        for side,path in data.get('models',{}).items():
            if side in self.models:self.models[side].set(str(resolve_path(path)))
        self.persist()
    def save_settings(self):
        try:
            data=self.scenario();path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.json',filetypes=[('Scenario JSON','*.json')])
            if path:Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
        except Exception as e:messagebox.showerror('시나리오 오류',str(e),parent=self.root)
    def load_settings(self):
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('Scenario JSON','*.json')])
        if not path:return
        try:self.apply_document(json.loads(Path(path).read_text(encoding='utf-8-sig')));self.status.set('시나리오를 화면의 파라미터 입력란에 적용했습니다.')
        except Exception as e:messagebox.showerror('시나리오 오류',str(e),parent=self.root)
    def running(self):
        if self.process and self.process.poll() is None:return True
        data=load_state(self.output/'progress.json') if self.output else None
        if data and data.get('status')=='running' and data.get('pid'):
            try:
                args=psutil.Process(data['pid']).cmdline();return 'unity_bridge.match_series' in args and str(self.output) in args
            except (psutil.Error,OSError):pass
        return False
    def start(self):
        if self.running():messagebox.showinfo('대결 중','현재 대결을 먼저 중단하세요.',parent=self.root);return
        try:
            data=self.scenario()
            if data['seed'] is None:data['seed']=self.seed_value(resolve=True)
            if data['rounds']<1 or data['seed']<0 or not math.isfinite(data['timeoutSeconds']) or data['timeoutSeconds']<=0:raise ValueError('경기 수·시간 제한·seed를 확인하세요.')
            for side,path in data['models'].items():
                if not path or not resolve_path(path).is_file():raise ValueError(f'{side.upper()} 모델 파일을 선택하세요.')
                data['models'][side]=str(resolve_path(path))
                self.models[side].set(data['models'][side])
            self.output=MATCH_ROOT/datetime.now().strftime('%Y%m%d-%H%M%S-%f');self.output.mkdir(parents=True)
            path=self.output/'scenario.json';path.write_text(json.dumps(data,indent=2),encoding='utf-8')
            args=['--scenario',str(path),'--model-a',data['models']['a'],'--model-b',data['models']['b'],'--output',str(self.output),'--rounds',str(data['rounds']),'--seed',str(data['seed']),'--timeout',str(data['timeoutSeconds'])]
            if data['resample']:args+=['--resample']
            self.rows=[];self.toggled.clear();self.pending_view=None;self.tree.delete(*self.tree.get_children());self.update_toggles();self.run=self.output
            self.winrate.set('A 승률 — / B 승률 —\n전체 완료 경기 기준 · 무승부 포함')
            self.process=self.launch('unity_bridge.match_series',args,self.output/'match.log');self.persist();self.status.set('대결 시작 중…')
        except Exception as e:messagebox.showerror('대결 시작 실패',str(e),parent=self.root)
    def stop(self):
        if self.output and self.running():(self.output/'stop.request').touch();self.status.set('현재 경기 종료 후 중단합니다.')
    def replay(self):
        if self.toggled:self.request_view(sorted(self.toggled),multiple=True)
    def click_toggle(self,event):
        iid=self.tree.identify_row(event.y)
        if iid and self.tree.identify_column(event.x)=='#1':
            index=int(iid);self.tree.selection_set(iid)
            if index in self.toggled:self.toggled.remove(index)
            else:self.toggled.add(index)
            self.update_toggles();return 'break'
    def double_click(self,event):
        iid=self.tree.identify_row(event.y)
        if iid and self.tree.identify_column(event.x)!='#1':
            self.tree.selection_set(iid);self.request_view([int(iid)],multiple=False)
        return 'break'
    def toggle_all(self,enabled):
        self.toggled=set(range(len(self.rows))) if enabled else set();self.update_toggles()
    def update_toggles(self):
        for iid in self.tree.get_children():self.tree.set(iid,'toggle','☑' if int(iid) in self.toggled else '☐')
        self.view_button.configure(text=f'토글 경기 3D 보기 ({len(self.toggled)})',state='normal' if self.toggled else 'disabled')
    def replay_path(self,row,folder):
        original=Path(row['replay']) if row.get('replay') else None
        if original and original.is_file():return original
        generated=folder/'replays'/f'round-{row["round"]:05d}.json'
        return generated if generated.is_file() else None
    def request_view(self,indices,*,multiple):
        self.pending_view=(self.output,[self.rows[i] for i in indices],multiple);self.prepare_view()
    def prepare_view(self):
        if not self.pending_view or (self.recovery and self.recovery.poll() is None):return
        folder,rows,multiple=self.pending_view
        missing=[r for r in rows if not self.replay_path(r,folder)]
        if missing:
            self.view_notice.set(f'미저장 경기 {len(missing)}개를 저장된 모델·조건으로 재구성 중…')
            self.recovery=self.launch('unity_bridge.replay_recover',['--run',str(folder),'--rounds',*[str(r['round']) for r in missing]],folder/'replay-recover.log');return
        self.pending_view=None;paths=[str(self.replay_path(r,folder)) for r in rows]
        if multiple:
            manifest=folder/f'view-selection-{datetime.now().strftime("%H%M%S-%f")}.json';manifest.write_text(json.dumps(paths),encoding='utf-8')
            self.launch('unity_bridge.replay_native',['--records',str(manifest)],folder/'playlist-viewer.log')
        else:self.launch('unity_bridge.replay_native',paths,folder/f'viewer-{rows[0]["round"]}.log')
        self.view_notice.set(f'{len(paths)}경기 {"순차 " if multiple else ""}반복 재생 창을 열었습니다.')
    def poll(self):
        if self.output:
            try:
                for line in (self.output/'rounds.jsonl').read_text(encoding='utf-8-sig').splitlines()[len(self.rows):]:
                    try:r=json.loads(line)
                    except ValueError:break
                    index=len(self.rows);self.rows.append(r)
                    self.tree.insert('','end',iid=str(index),values=('☐',r['round'],r['winner']+' 승리' if r['winner'] else '무승부',round(r['timeMs']/1000,2),*r['shots'],*r['blocked'],'저장' if self.replay_path(r,self.output) else '재구성'))
            except OSError:pass
            summary=load_state(self.output/'progress.json') or {}
            if summary:
                completed=summary['completed']
                if completed:
                    self.winrate.set(f"A 승률 {summary['winsA']/completed:.1%} ({summary['winsA']}/{completed})\nB 승률 {summary['winsB']/completed:.1%} ({summary['winsB']}/{completed})\n무승부 {summary['draws']/completed:.1%} · 전체 완료 경기 기준")
                state=summary['status']
                if state=='running' and not self.running():state='실행 종료 · 로그 확인'
                seed_info=f' | Seed {summary["seed"]}' if 'seed' in summary else ''
                self.status.set(f"{state} | {summary['completed']}/{summary['requested']} 경기 | A 승 {summary['winsA']} / B 승 {summary['winsB']} / 무승부 {summary['draws']}{seed_info}")
                self.bar['value']=100*summary['completed']/summary['requested']
            if self.process and self.process.poll() not in (None,0):self.status.set('대결 실행 실패 · 실행 로그 확인')
            try:
                text=(self.output/'match.log').read_text(encoding='utf-8',errors='replace')[-10000:];self.log.configure(state='normal');self.log.delete('1.0','end');self.log.insert('end',text);self.log.configure(state='disabled')
            except OSError:pass
        if self.recovery and self.recovery.poll() is not None:
            code=self.recovery.returncode;self.recovery=None
            if code:
                self.pending_view=None;self.view_notice.set('기록 재구성 실패 · replay-recover.log를 확인하세요.')
            else:self.prepare_view()
        self.root.after(700,self.poll)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',help='Open a saved match result with its own viewer settings')
    args=parser.parse_args()
    folder=resolve_path(args.run) if args.run else None
    root=tk.Tk();app=MatchSimulator(root,state_path=folder/'viewer-state.json' if folder else None)
    if folder:
        manifest=load_state(folder/'manifest.json') or {}
        app.apply_document(dict(parameters=manifest['parameters'],rewardConfig=manifest.get('rewardConfig',{}),
            playerPresets=manifest.get('playerPresets',{}),rounds=manifest.get('rounds',50),
            timeoutSeconds=manifest.get('timeoutSeconds',30),seed=manifest.get('seed',421),
            resample=manifest.get('resample',False),models={s:str(folder/f'agent_{s}.zip') for s in ('a','b')}))
        app.output=app.run=folder;app.persist()
        root.title(f'대전 시뮬레이터 · {folder.parent.name} / {folder.name}')
    root.mainloop()
if __name__=='__main__':main()
