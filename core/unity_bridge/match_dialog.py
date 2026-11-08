"""Custom repeated-duel controls, kept separate from training execution."""
import json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from unity_bridge.training_settings import migrate,validate
from unity_bridge.competitive_rewards import normalize,validate as validate_rewards
from unity_bridge.trainer_state import load_state,checkpoint

class MatchDialog:
    def __init__(self,owner):
        self.owner=owner;self.window=tk.Toplevel(owner.root);self.window.title('커스텀 시나리오 · 반복 대결');self.window.geometry('1120x740')
        self.process=None;self.output=None;self.rows=[]
        self.scenario=dict(parameters=owner.settings(),rewardConfig=owner.reward_config(),playerPresets=owner.player_selection())
        saved=getattr(owner,'match_options',{})
        self.vars={key:tk.StringVar(value=str(saved.get(key,value))) for key,value in dict(model_a='',model_b='',rounds=20,seed=900123,timeout=owner.timeout.get(),replays=5).items()}
        self.resample=tk.BooleanVar(value=saved.get('resample',False));self.status=tk.StringVar(value='학습 가중치를 바꾸지 않고 지정한 두 모델을 대결시킵니다.')
        source=owner.run or (Path(owner.resume.get()) if owner.resume.get() else None)
        if source:
            try:
                folder,_=checkpoint(source)
                for side in ('a','b'):
                    if not self.vars['model_'+side].get():self.vars['model_'+side].set(str((folder/f'agent_{side}.zip').resolve()))
            except ValueError:pass
        outer=ttk.Frame(self.window,padding=14);outer.pack(fill='both',expand=True)
        for side in ('a','b'):
            row=ttk.Frame(outer);row.pack(fill='x',pady=4)
            ttk.Label(row,text=f'에이전트 {side.upper()} 모델',width=18).pack(side='left')
            ttk.Entry(row,textvariable=self.vars['model_'+side]).pack(side='left',fill='x',expand=True,padx=5)
            ttk.Button(row,text='선택',command=lambda s=side:self.browse(s)).pack(side='left')
        row=ttk.Frame(outer);row.pack(fill='x',pady=8)
        for key,label in [('rounds','반복 경기 수'),('seed','Seed'),('timeout','경기 제한(초)'),('replays','저장할 리플레이 수')]:
            ttk.Label(row,text=label).pack(side='left',padx=3);ttk.Entry(row,textvariable=self.vars[key],width=8).pack(side='left',padx=(0,12))
        ttk.Checkbutton(outer,text='매 경기 범위 조건 재추출 (끄면 한 번 추출한 동일 조건으로 반복 대결)',variable=self.resample).pack(anchor='w')
        controls=ttk.Frame(outer);controls.pack(fill='x',pady=8)
        for label,fn in [('현재 화면 조건 가져오기',self.capture),('시나리오 불러오기',self.load),('시나리오 저장',self.save),('조건 보기',self.inspect),('대결 시작',self.start),('중단',self.stop),('선택 경기 3D 보기',self.replay)]:ttk.Button(controls,text=label,command=fn).pack(side='left',padx=2)
        ttk.Label(outer,textvariable=self.status,wraplength=1060).pack(anchor='w',pady=6)
        columns=('round','winner','seconds','shotsA','shotsB','blockA','blockB','replay')
        self.tree=ttk.Treeview(outer,columns=columns,show='headings',height=14)
        for col,label in zip(columns,('경기','결과','경과(초)','A 발사','B 발사','A 서브 보류/차단','B 서브 보류/차단','리플레이')):
            self.tree.heading(col,text=label);self.tree.column(col,width=110,anchor='center')
        self.tree.pack(fill='both',expand=True);self.tree.bind('<Double-1>',lambda e:self.replay())
        self.log=tk.Text(outer,height=6,state='disabled');self.log.pack(fill='x',pady=6)
        for variable in self.vars.values():variable.trace_add('write',self.persist)
        self.resample.trace_add('write',self.persist)
        self.poll()
    def persist(self,*_):
        self.owner.match_options={k:v.get() for k,v in self.vars.items()};self.owner.match_options['resample']=self.resample.get();self.owner.schedule_save()
    def browse(self,side):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[('PPO checkpoints','*.zip')])
        if path:self.vars['model_'+side].set(path)
    def capture(self):
        try:self.scenario=dict(parameters=self.owner.settings(),rewardConfig=self.owner.reward_config(),playerPresets=self.owner.player_selection());self.status.set('현재 학습기 화면의 A/B·게임·보상 조건을 복사했습니다.')
        except Exception as e:messagebox.showerror('조건 오류',str(e),parent=self.window)
    def load(self):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[('Scenario JSON','*.json')])
        if not path:return
        try:
            d=json.loads(Path(path).read_text(encoding='utf-8-sig'));settings=migrate(d.get('parameters',d));validate(settings)
            rewards=normalize(d.get('rewardConfig',{}));validate_rewards(rewards);self.scenario=dict(parameters=settings,rewardConfig=rewards,playerPresets=d.get('playerPresets',{}))
            self.status.set('불러온 시나리오: '+path)
        except Exception as e:messagebox.showerror('조건 오류',str(e),parent=self.window)
    def save(self):
        path=filedialog.asksaveasfilename(parent=self.window,defaultextension='.json',filetypes=[('Scenario JSON','*.json')])
        if path:Path(path).write_text(json.dumps(self.scenario,indent=2,ensure_ascii=False),encoding='utf-8')
    def inspect(self):
        top=tk.Toplevel(self.window);top.title('대결에 사용할 시나리오');text=tk.Text(top,width=90,height=35);text.pack(fill='both',expand=True);text.insert('end',json.dumps(self.scenario,indent=2,ensure_ascii=False));text.configure(state='disabled')
    def start(self):
        if self.process and self.process.poll() is None:return
        try:
            rounds=int(self.vars['rounds'].get());seed=int(self.vars['seed'].get());timeout=float(self.vars['timeout'].get());replays=int(self.vars['replays'].get())
            if rounds<1 or seed<0 or timeout<=0 or replays<0:raise ValueError('경기 수·시간 제한은 양수, seed·리플레이 수는 0 이상이어야 합니다.')
            for side in ('a','b'):
                if not Path(self.vars['model_'+side].get()).is_file():raise ValueError(f'{side.upper()} 모델을 선택하세요.')
            root=Path(__file__).resolve().parents[1];self.output=root/'match-results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');self.output.mkdir(parents=True)
            scenario=self.output/'scenario.json';scenario.write_text(json.dumps(self.scenario,indent=2),encoding='utf-8')
            args=['--scenario',str(scenario),'--model-a',self.vars['model_a'].get(),'--model-b',self.vars['model_b'].get(),'--rounds',str(rounds),'--seed',str(seed),'--timeout',str(timeout),'--replays',str(replays),'--output',str(self.output)]
            if self.resample.get():args+=['--resample']
            self.rows=[];self.tree.delete(*self.tree.get_children());self.process=self.owner.launch('unity_bridge.match_series',args,self.output/'match.log');self.persist();self.status.set('반복 대결 시작 중…')
        except Exception as e:messagebox.showerror('대결 시작 실패',str(e),parent=self.window)
    def stop(self):
        if self.output and self.process and self.process.poll() is None:(self.output/'stop.request').touch();self.status.set('현재 경기 종료 후 대결을 중단합니다.')
    def replay(self):
        selected=self.tree.selection()
        if not selected:return
        row=self.rows[int(selected[0])]
        if not row['replay']:messagebox.showinfo('리플레이 없음','저장할 리플레이 수를 늘려 실행하세요.',parent=self.window);return
        self.owner.launch('unity_bridge.replay_native',[row['replay']],self.output/f'viewer-{row["round"]}.log')
    def poll(self):
        if not self.window.winfo_exists():return
        if self.output:
            try:
                lines=(self.output/'rounds.jsonl').read_text().splitlines()
                for line in lines[len(self.rows):]:
                    try:r=json.loads(line)
                    except ValueError:break
                    index=len(self.rows);self.rows.append(r);self.tree.insert('', 'end',iid=str(index),values=(r['round'],r['winner']+' 승리' if r['winner'] else '무승부',round(r['timeMs']/1000,2),*r['shots'],*r['blocked'],'저장됨' if r['replay'] else '없음'))
            except OSError:pass
            summary=load_state(self.output/'progress.json') or {}
            if summary:self.status.set(f"{summary['status']} | {summary['completed']}/{summary['requested']} 경기 | A 승 {summary['winsA']} / B 승 {summary['winsB']} / 무승부 {summary['draws']}")
            if self.process and self.process.poll() not in (None,0):self.status.set('대결 실행 실패 · 아래 로그 확인')
            try:
                text=(self.output/'match.log').read_text(encoding='utf-8',errors='replace')[-5000:];self.log.configure(state='normal');self.log.delete('1.0','end');self.log.insert('end',text);self.log.configure(state='disabled')
            except OSError:pass
        self.window.after(700,self.poll)
