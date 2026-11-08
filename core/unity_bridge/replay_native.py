"""Native offline 3D observation of recorded trajectories (no browser/JS)."""
import argparse,json,math,bisect,time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from unity_bridge.storage import resolve_path
ALLOW_COLOR='#166534'
HOLD_COLOR='#991b1b'
UNKNOWN_COLOR='#475569'

def state_at(data,t):
    events=[e for e in data['events'] if e['timeMs']<=t]
    hp=events[-1]['hpAfter'] if events else data['hp']
    killed={e['agent'] for e in events if e.get('killed')}
    result=('A 처치 · B 패배' if killed=={0} else 'B 처치 · A 패배' if killed=={1} else '동시 처치 · 무승부') if killed else ('시간 종료 · 무승부' if t>=data['duration'] else '진행 중')
    agents=[]
    settings=data['settings']
    for i in range(2):
        shots=[shot for shot in data['shots'] if shot['agent']==i and shot['t']<=t]
        hits=[e for e in events if e['agent']==i and e['hit']]
        decisions=[d for d in data.get('decisions',[[],[]])[i] if d['timeMs']<=t]
        original=[d for d in data.get('originalDecisions',[[],[]])[i] if d['timeMs']<=t]
        current=original[-1] if original else None
        judging=bool(current and current.get('requested') and not current.get('pendingPlan'))
        last=next((d for d in reversed(decisions) if judging and abs(d['timeMs']-current['timeMs'])<1e-6),None)
        cooldown_blocked=bool(judging and current.get('cooldownBlocked'))
        main=('발사 요청' if current.get('requested') else '기존 발사 계획 유지' if current.get('pendingPlan') else '격발하지 않음') if current else '판단 기록 없음'
        sub='쿨다운 중' if cooldown_blocked else ('격발 승인' if last['approved'] else '보류') if last else ''
        if current is None:
            block='기록 없음' if last else '아직 판단 없음'
        elif t-current['timeMs']>=100:block='다음 판단 대기'
        elif current.get('ppoBlocked') and current.get('cooldownBlocked'):block='차단 · PPO + 쿨다운'
        elif current.get('ppoBlocked'):block='차단 · PPO 보류'
        elif current.get('cooldownBlocked'):block='차단 · 쿨다운'
        elif not current['requested'] and not current.get('pendingPlan'):block='발사 요청 없음'
        elif current.get('actual') or current.get('pendingPlan'):block='허용 · 발사 계획 진행'
        else:block='요청 보류'
        effective=block
        if current and not current.get('requested') and not current.get('pendingPlan'):effective='발사 요청 없음 · 차단 미적용'
        interval=settings.get('shotIntervalsMs',[150,150])[i]
        cooldown=max(0.,shots[-1]['t']+interval-t) if shots else 0.
        if current and 'effectiveShotTimeMs' in current:
            awaiting_shot=t<current['effectiveShotTimeMs']
        else:
            awaiting_shot=not shots or bool(current and (shots[-1]['t']<current['timeMs'] or
                (current.get('actual') and shots[-1]['t']==current['timeMs'])))
        plan_active=bool(current and (current.get('actual') or current.get('pendingPlan'))
                         and not current.get('ppoBlocked') and awaiting_shot and t<data['duration'])
        agents.append(dict(shots=len(shots),hits=len(hits),damage=sum(e['damage'] for e in events if e['agent']==i),
            cooldown=cooldown,last=last,original=original[-1] if original else None,
            approved=sum(bool(d['approved']) for d in decisions),rejected=sum(not d['approved'] for d in decisions),
            last_hit=next((e for e in reversed(events) if e['agent']==i),None)))
        agents[-1].update(gate_reason=block,block=('격발 승인' if last['approved'] else '보류 (차단)') if last else '판단 기록 없음',blocked=sum(not d['approved'] for d in decisions),
            cooldown_blocked=sum(bool(d.get('cooldownBlocked')) for d in original),main=main,sub=sub,effective=effective,
            last_request=next((d for d in reversed(original) if d.get('requested')),None),plan_active=plan_active,
            sub_visible=bool(last or cooldown_blocked),cooldown_rejected=cooldown_blocked)
    return dict(hp=hp,result=result,agents=agents,events=events)

class Replay:
    def __init__(self,root,path,*,records=None,groups=None,pause_on_main=False):
        self.groups=groups
        self.records=[str(resolve_path(p)) for p in (records or [path])];self.record_index=0;path=self.records[0]
        self.data=json.loads(Path(path).read_text(encoding='utf-8-sig'));self.root=root
        self.path=path;self.closed=False;self.after_id=None
        root.title('경기 행동 기록 · 3D 리플레이');root.geometry(f'1500x{min(1020,root.winfo_screenheight()-70)}');root.minsize(1150,800)
        self.t=0.;self.playing=True;self.end_hold=None;self.yaw=.4;self.pitch=.3;self.zoom=1.;self.drag=None;self.previous=time.perf_counter()
        self.frames=self.data['frames'];self.times=[f['t'] for f in self.frames];self.duration=self.data['duration'];self.distance=self.data['distance']
        self.speed=tk.StringVar(value='.5');self.loop=tk.BooleanVar(value=True);self.status=tk.StringVar()
        playlist_box=ttk.LabelFrame(root,text='리플레이 목록',padding=4)
        playlist_box.pack(side='right',fill='y',padx=(0,8),pady=8)
        playlist_box.columnconfigure(0,weight=1)
        playlist_box.rowconfigure(2,weight=1)
        if groups:
            self.preset_list=ttk.Combobox(playlist_box,state='readonly',values=[g['label'] for g in groups])
            self.preset_list.grid(row=0,column=0,sticky='ew',pady=(0,4))
            self.preset_list.current(0)
            self.preset_list.bind('<<ComboboxSelected>>',self.select_preset)
            self.setting_list=ttk.Combobox(playlist_box,state='readonly',
                values=[s['label'] for s in groups[0]['settings']])
            self.setting_list.grid(row=1,column=0,sticky='ew',pady=(0,4))
            self.setting_list.current(0)
            self.setting_list.bind('<<ComboboxSelected>>',self.select_setting)
        self.record_list=tk.Listbox(playlist_box,width=43,exportselection=False,activestyle='none')
        vertical=ttk.Scrollbar(playlist_box,orient='vertical',command=self.record_list.yview)
        horizontal=ttk.Scrollbar(playlist_box,orient='horizontal',command=self.record_list.xview)
        self.record_list.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        self.record_list.grid(row=2,column=0,sticky='nsew')
        vertical.grid(row=2,column=1,sticky='ns');horizontal.grid(row=3,column=0,sticky='ew')
        names=[r['label'] for r in groups[0]['settings'][0]['records']] if groups else [Path(r).stem for r in self.records]
        self.populate_records(names)
        self.record_list.selection_set(self.record_index)
        self.record_list.bind('<<ListboxSelect>>',self.select_record)
        main=ttk.Frame(root);main.pack(side='left',fill='both',expand=True)
        header=ttk.Frame(main,padding=(18,8));header.pack(fill='x')
        self.record_label=ttk.Label(header,text=self.heading(),font=('Malgun Gothic',14,'bold'));self.record_label.pack(anchor='w')
        self.replay_conditions=tk.StringVar(value=self.condition_heading())
        ttk.Label(header,textvariable=self.replay_conditions,font=('Malgun Gothic',12,'bold'),justify='left').pack(anchor='w',pady=(5,0))
        bar=ttk.Frame(main,padding=(12,4));bar.pack(fill='x')
        self.button=ttk.Button(bar,text='일시정지',command=self.toggle)
        self.button.pack(side='left')
        ttk.Button(bar,text='처음',command=self.reset).pack(side='left',padx=5)
        if groups or len(self.records)>1:
            ttk.Button(bar,text='이전 경기',command=lambda:self.load_index((self.record_index-1)%len(self.records))).pack(side='left')
            ttk.Button(bar,text='다음 경기',command=lambda:self.load_index((self.record_index+1)%len(self.records))).pack(side='left')
        ttk.Combobox(bar,textvariable=self.speed,values=['.1','.25','.5','1','2','4'],width=5,state='readonly').pack(side='left')
        ttk.Label(bar,text='배속').pack(side='left');ttk.Checkbutton(bar,text='현재 경기 반복',variable=self.loop).pack(side='left',padx=10)
        ttk.Label(main,text='파랑 A · 주황 B | 두 에이전트는 같은 물리 위치에서 시작 | 드래그: 시점 회전 · 휠: 확대/축소').pack(anchor='w',padx=20)
        camera_bar=ttk.Frame(main);camera_bar.pack(fill='x',padx=20)
        ttk.Label(camera_bar,text='카메라').pack(side='left');self.camera_mode=tk.StringVar(value='전체 보기')
        camera_select=ttk.Combobox(camera_bar,textvariable=self.camera_mode,values=['전체 보기','타겟 추적','타겟 시점'],state='readonly',width=14);camera_select.pack(side='left',padx=6);camera_select.bind('<<ComboboxSelected>>',self.change_camera)
        self.pause_on_main=tk.BooleanVar(value=pause_on_main)
        ttk.Checkbutton(camera_bar,text='메인 발사 요청 시 일시정지',variable=self.pause_on_main).pack(side='left',padx=12)
        self.game_status=tk.StringVar()
        ttk.Label(main,textvariable=self.game_status,font=('Malgun Gothic',13,'bold'),padding=(15,8)).pack(fill='x')
        self.hp_bar=tk.Canvas(main,height=46,bg='#111827',highlightthickness=0);self.hp_bar.pack(fill='x',padx=15,pady=(0,8))
        body=ttk.Frame(main);body.pack(fill='both',expand=True,padx=12)
        self.agent_panels=[]
        for side in ('A','B'):
            panel=ttk.LabelFrame(body,text='에이전트 '+side,padding=10,width=270)
            panel.pack(side='left' if side=='A' else 'right',fill='y',padx=4)
            self.agent_panels.append(panel)
        self.agent_status=[tk.StringVar(),tk.StringVar()]
        self.condition_status=[tk.StringVar(),tk.StringVar()]
        self.cooldown_panels=[]
        self.stats_panels=[]
        for i,panel in enumerate(self.agent_panels):
            cooldown=tk.Canvas(panel,width=250,height=40,bg='#0f172a',highlightthickness=0)
            cooldown.pack(fill='x',pady=(0,8));self.cooldown_panels.append(cooldown)
            stats=tk.Canvas(panel,width=250,height=118,bg='#0f172a',highlightthickness=0)
            stats.pack(fill='x',pady=(0,8));self.stats_panels.append(stats)
            ttk.Label(panel,textvariable=self.agent_status[i],justify='left',wraplength=270,font=('Malgun Gothic',10)).pack(anchor='nw')
            ttk.Separator(panel).pack(fill='x',pady=5)
            ttk.Label(panel,text='이 경기의 조건',font=('Malgun Gothic',10,'bold')).pack(anchor='w')
            ttk.Label(panel,textvariable=self.condition_status[i],justify='left',wraplength=270).pack(anchor='nw',pady=8)
            self.condition_status[i].set(self.condition_text(i))
        self.canvas=tk.Canvas(body,bg='#101827',highlightthickness=0);self.canvas.pack(fill='both',expand=True,padx=8)
        log_box=ttk.LabelFrame(main,text='현재 재생 시점까지의 발사 / 피해 사건',padding=4)
        self.event_log=tk.Text(log_box,height=4,width=1,font=('Consolas',10),state='disabled')
        self.event_log.pack(fill='both',expand=True)
        self.previous_log=None
        self.canvas.bind('<ButtonPress-1>',lambda e:setattr(self,'drag',(e.x,e.y)))
        self.canvas.bind('<B1-Motion>',self.rotate);self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'drag',None))
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.seek=tk.Scale(main,from_=0,to=self.duration,orient='horizontal',showvalue=False,command=self.scrub);self.seek.pack(fill='x',padx=15)
        self.footer=ttk.Label(main,textvariable=self.status,padding=10)
        self.footer.pack(side='bottom',anchor='w',before=body)
        self.seek.pack_configure(side='bottom',before=body)
        log_box.pack(side='bottom',fill='x',padx=15,pady=6,before=body)
        root.bind('<Left>',lambda e:self.step_time(-100))
        root.bind('<Right>',lambda e:self.step_time(100))
        root.bind('<space>',lambda e:self.toggle());root.bind('<Destroy>',self.on_destroy,add='+');self.tick()
    def on_destroy(self,event):
        if event.widget is self.root:
            self.closed=True
            if self.after_id:self.root.after_cancel(self.after_id)
    def heading(self):
        comparison=self.data.get('comparison')
        if comparison:
            return (f"비교 {comparison['pairIndex']}/{comparison['pairCount']} · {comparison['modelLabel']} · "
                    f"{comparison['stage']} · 경기 {comparison['round']} / 시드 {comparison['seed']}")
        prefix=f'[{self.record_index+1}/{len(self.records)}] ' if len(self.records)>1 else ''
        return prefix+self.data.get('label',Path(self.path).stem)
    def condition_heading(self):
        rewards=self.data.get('settings',{}).get('rewardValues',{})
        lines=[]
        for side in ('A','B'):
            reward=rewards.get(side)
            if reward:
                fixed='  [고정 상대]' if self.data.get('comparison',{}).get('fixedOpponent')==side else ''
                alpha=f"α {reward['alpha']:g} · " if 'alpha' in reward else ''
                lines.append(f"{side}: {alpha}내 처치 보상 +{reward['win']:g} / 패배 비용 −{reward['loss']:g}{fixed}")
                if 'damageFactor' in reward or 'criticalFactor' in reward:
                    lines[-1]+=f" · 피해 {reward.get('damageFactor',0):g} · Critical {reward.get('criticalFactor',0):g}"
        if self.data.get('comparison'):
            lines.append('비교: A 처치 보상 1 → 5  |  B 모델·보상, 게임 조건, 시드 동일')
        return '\n'.join(lines)
    def select_record(self,_event):
        selected=self.record_list.curselection()
        if selected and selected[0]!=self.record_index:
            self.load_index(selected[0])
    def populate_records(self,names):
        self.record_list.delete(0,'end')
        for index,name in enumerate(names,1):
            self.record_list.insert('end',f'{index:02d}. {name}')
    def select_preset(self,_event):
        settings=self.groups[self.preset_list.current()]['settings']
        self.setting_list.configure(values=[s['label'] for s in settings])
        self.setting_list.current(0)
        self.select_setting(None)
    def select_setting(self,_event):
        settings=self.groups[self.preset_list.current()]['settings']
        records=settings[self.setting_list.current()]['records']
        self.records=[str(resolve_path(r['path'])) for r in records]
        self.populate_records([r['label'] for r in records])
        self.load_index(0)
    def load_index(self,index):
        path=self.records[index];data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        self.record_index=index;self.path=path;self.data=data
        self.frames=data['frames'];self.times=[f['t'] for f in self.frames];self.duration=data['duration'];self.distance=data['distance']
        self.t=0.;self.end_hold=None;self.previous=time.perf_counter();self.previous_log=None;self._camera_key=None
        self.seek.configure(to=self.duration);self.seek.set(0);self.record_label.configure(text=self.heading())
        self.record_list.selection_clear(0,'end')
        self.record_list.selection_set(index)
        self.record_list.activate(index)
        self.record_list.see(index)
        self.replay_conditions.set(self.condition_heading())
        for i in range(2):self.condition_status[i].set(self.condition_text(i))
    def condition_text(self,i):
        settings=self.data['settings'];profile=settings.get('profiles',[[],[]])[i]
        weapon=settings.get('weapons',[dict(minimum=settings.get('minDamage',100),maximum=settings.get('maxDamage',300),instantKill=True)]*2)[i]
        reward=settings.get('rewardValues',{}).get('AB'[i])
        lines=[f"피해 외곽 {weapon['minimum']:g} / 중앙 {weapon['maximum']:g}",
            f"치명타 즉사 {'켬' if weapon['instantKill'] else '끔'} · 탄 퍼짐 {settings.get('shotSpreadDeg',[0,0])[i]:g}°",
            f"재장전 {settings.get('shotIntervalsMs',[150,150])[i]:g} ms"]
        if len(profile)>=8:
            lines += [f'운동·위치 노이즈 {profile[0]:g} / {profile[1]:g}',
                      f'속도·시계 노이즈 {profile[2]:g} / {profile[3]:g}',
                      f'원본 성공 보상·실패 비용 {profile[4]:g} / {profile[5]:g}',
                      f'원본 보상·비용 감쇠 {profile[6]:g} / {profile[7]:g}']
        if reward:
            alpha=f"α {reward['alpha']:g} · " if 'alpha' in reward else ''
            lines.append(f"{alpha}처치 +{reward['win']:g} / 패배 −{reward['loss']:g}")
        else:
            lines.append('경쟁 보상 계수: 기록 없음')
        return '\n'.join(lines)
    def step_time(self,delta):
        self.playing=False;self.end_hold=None;self.t=max(0,min(self.duration,self.t+delta))
    def update_status(self,f):
        state=state_at(self.data,self.t)
        ratio=max(0,min(1,state['hp']/self.data['hp']))
        color='#22c55e' if ratio>.5 else '#f59e0b' if ratio>.2 else '#ef4444'
        width=max(1,self.hp_bar.winfo_width());self.hp_bar.delete('all')
        self.hp_bar.create_rectangle(0,0,width,46,fill='#1e293b',outline='')
        self.hp_bar.create_rectangle(0,0,width*ratio,46,fill=color,outline='')
        self.hp_bar.create_text(width/2,23,text=f"TARGET HP   {state['hp']:,.1f} / {self.data['hp']:,.0f}   ({ratio*100:.1f}%)",fill='white',font=('Malgun Gothic',20,'bold'))
        settings=self.data['settings']
        self.game_status.set(f"{state['result']}   |   표적 체력 {state['hp']:.1f} / {self.data['hp']:g}   |   표적 반지름 {settings.get('targetRadius','?')}° · 속도 {settings.get('targetSpeed','?')}°/s · 치명타 범위 {self.data['critical']*100:g}%")
        for i,a in enumerate(state['agents']):
            remaining=a['cooldown'];interval=settings.get('shotIntervalsMs',[150,150])[i]
            cooldown=self.cooldown_panels[i];width=max(1,cooldown.winfo_width());cooldown.delete('all')
            label=f'쿨다운 중  {remaining:.0f} ms' if remaining>0 else '격발 가능 · 쿨다운 0 ms'
            color='#f59e0b' if remaining>0 else '#22c55e'
            cooldown.create_text(width/2,13,text=label,fill=color,font=('Malgun Gothic',12,'bold'))
            cooldown.create_rectangle(5,29,width-5,36,fill='#334155',outline='')
            ready=max(0,min(1,1-remaining/max(interval,1)))
            cooldown.create_rectangle(5,29,5+(width-10)*ready,36,fill=color,outline='')
            target=f['target'];aim=f['aims'][i]
            norm=math.sqrt(sum(x*x for x in target))*math.sqrt(sum(x*x for x in aim))
            angle=math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(target,aim))/max(norm,1e-12)))))
            last_hit=a['last_hit']
            hit_label=(f"{'적중' if last_hit['hit'] else '빗나감'} · 피해 {last_hit['damage']:.1f}\n@ {last_hit['timeMs']/1000:.2f}초") if last_hit else '아직 발사 없음'
            stats=self.stats_panels[i];width=max(1,stats.winfo_width());stats.delete('all')
            stats.create_text(width/2,12,text='누적 피해',fill='#cbd5e1',font=('Malgun Gothic',10,'bold'))
            stats.create_text(width/2,36,text=f"{a['damage']:,.1f}",fill=('#69baff','#ffb560')[i],font=('Malgun Gothic',23,'bold'))
            for k,(name,value,color) in enumerate((('발사',a['shots'],'#f8fafc'),('명중',a['hits'],'#86efac'),('빗나감',a['shots']-a['hits'],'#fca5a5'))):
                x=width*(k+.5)/3
                stats.create_text(x,61,text=name,fill='#cbd5e1',font=('Malgun Gothic',10,'bold'))
                stats.create_text(x,83,text=str(value),fill=color,font=('Malgun Gothic',18,'bold'))
            criticals=[e for e in state['events'] if e['agent']==i and e.get('critical') and self.t-e['timeMs']<700]
            if criticals:stats.create_text(width/2,105,text='CRITICAL!',fill='#fbbf24',font=('Malgun Gothic',12,'bold'),tags='critical')
            lines=[f'조준 오차 {angle:.2f}°',f'최근 발사: {hit_label}']
            self.agent_status[i].set('\n'.join(lines))
        text='\n'.join(f"{e['timeMs']/1000:6.3f}s  {'AB'[e['agent']]}  {'처치' if e.get('killed') else '적중' if e['hit'] else '빗나감'}  피해 {e['damage']:7.1f}  남은 체력 {e['hpAfter']:7.1f}" for e in state['events'][-4:]) or '아직 발사 / 피해 사건 없음'
        if text!=self.previous_log:
            self.event_log.configure(state='normal');self.event_log.delete('1.0','end');self.event_log.insert('end',text);self.event_log.configure(state='disabled');self.previous_log=text
    def toggle(self):
        if self.t>=self.duration:self.t=0
        self.playing=not self.playing
    def reset(self):self.t=0;self.end_hold=None
    def scrub(self,value):
        # Tk dispatches Scale.set callbacks after this frame. Ignore its own
        # rounded value; otherwise it resets the end hold on every tick.
        if not getattr(self,'setting_seek',False) and abs(float(value)-self.t)>1:
            self.t=float(value);self.end_hold=None
    def rotate(self,e):
        if self.drag:self.yaw+=(e.x-self.drag[0])*.008;self.pitch+=(e.y-self.drag[1])*.008
        self.drag=(e.x,e.y)
    def wheel(self,e):self.zoom=max(.3,min(6,self.zoom*math.exp(e.delta*.001)))
    def change_camera(self,event=None):
        self._camera_key=None;self.zoom=1.
        self.yaw,self.pitch=(0.,0.) if self.camera_mode.get()=='타겟 시점' else (.4,.3)
    def view_coordinates(self,p):
        delta=[p[i]-self.camera_target[i] for i in range(3)]
        return [sum(delta[i]*axis[i] for i in range(3)) for axis in self.camera_basis]
    def raw_projection(self,p):
        if self.camera_mode.get()=='타겟 시점':
            x,y,z=self.view_coordinates(p);depth=max(self.distance*.02,z)
            return x/depth,y/depth,depth/self.distance
        x,y,z=[v/self.distance for v in p];a=x*math.cos(self.yaw)-y*math.sin(self.yaw);b=x*math.sin(self.yaw)+y*math.cos(self.yaw)
        v=z*math.cos(self.pitch)-b*math.sin(self.pitch);depth=z*math.sin(self.pitch)+b*math.cos(self.pitch)
        factor=max(.2,2.8-depth)
        return a/factor,v/factor,factor
    def fit_camera(self):
        mode=self.camera_mode.get()
        key=(self.canvas.winfo_width(),self.canvas.winfo_height(),self.yaw,self.pitch,self.zoom,mode,tuple(self.camera_target) if mode!='전체 보기' else None)
        if getattr(self,'_camera_key',None)==key:return
        self._camera_key=key
        if mode=='타겟 시점':
            def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
            def unit(a):
                length=math.sqrt(sum(x*x for x in a));return [x/max(length,1e-12) for x in a]
            forward=unit([-v for v in self.camera_target]);up=[0,0,1] if abs(forward[2])<.95 else [0,1,0]
            right=unit(cross(forward,up));up=unit(cross(right,forward))
            forward=[forward[i]*math.cos(self.yaw)+right[i]*math.sin(self.yaw) for i in range(3)]
            right=unit(cross(forward,up));forward=[forward[i]*math.cos(self.pitch)+up[i]*math.sin(self.pitch) for i in range(3)]
            up=unit(cross(right,forward));self.camera_basis=(right,up,forward)
            self.camera_center=(0.,0.);self.camera_scale=min(self.canvas.winfo_width(),self.canvas.winfo_height())*.8*self.zoom;return
        if mode=='타겟 추적':
            x,y,factor=self.raw_projection(self.camera_target);self.camera_center=(x,y)
            self.camera_scale=min(self.canvas.winfo_width(),self.canvas.winfo_height())*.18*factor*self.distance/max(self.data['radius'],1e-6)*self.zoom;return
        points=[[0,0,0]]
        span=self.distance*.45
        points.extend([[span,0,0],[-span,0,0],[0,span,0],[0,-span,0]])
        radius=self.data['radius']
        for frame in self.frames[::max(1,len(self.frames)//80)]:
            center=frame['target'];points.append(center)
            for axis in range(3):
                for sign in (-1,1):
                    p=list(center);p[axis]+=sign*radius;points.append(p)
        projected=[self.raw_projection(p) for p in points]
        xs=[p[0] for p in projected];ys=[p[1] for p in projected]
        self.camera_center=((min(xs)+max(xs))/2,(min(ys)+max(ys))/2)
        self.camera_scale=min(max(1,self.canvas.winfo_width()-200)/max(max(xs)-min(xs),.1),max(1,self.canvas.winfo_height()-230)/max(max(ys)-min(ys),.1))*.85*self.zoom
    def project(self,p):
        x,y,factor=self.raw_projection(p)
        if not hasattr(self,'camera_center'):self.fit_camera()
        return self.canvas.winfo_width()/2+(x-self.camera_center[0])*self.camera_scale,self.canvas.winfo_height()*.56-(y-self.camera_center[1])*self.camera_scale,self.camera_scale/(factor*self.distance)
    def draw_target_hud(self,f):
        state=state_at(self.data,self.t);ratio=max(0,min(1,state['hp']/self.data['hp']))
        for i,agent in enumerate(state['agents']):
            left=12 if i==0 else max(12,self.canvas.winfo_width()-222)
            requested=bool(agent['original'] and agent['original'].get('requested'))
            main_color=ALLOW_COLOR if requested or agent['plan_active'] else HOLD_COLOR if agent['original'] else UNKNOWN_COLOR
            sub_color=(ALLOW_COLOR if agent['last']['approved'] else HOLD_COLOR) if agent['last'] else HOLD_COLOR
            self.canvas.create_rectangle(left,8,left+210,70,fill=main_color,outline='',tags='main_decision_box')
            self.canvas.create_text(left+105,23,text=f"{'AB'[i]} · 메인",fill='white',font=('Malgun Gothic',11,'bold'))
            self.canvas.create_text(left+105,48,text='발사 계획 수행중' if agent['plan_active'] else '요청' if requested else '요청 없음' if agent['original'] else '판단 기록 없음',fill='white',font=('Malgun Gothic',14 if agent['plan_active'] else 18,'bold'))
            if agent['sub_visible']:
                self.canvas.create_rectangle(left,77,left+210,139,fill=sub_color,outline='',tags='sub_decision_box')
                self.canvas.create_text(left+105,92,text=f"{'AB'[i]} · "+('상태' if agent['cooldown_rejected'] else '서브'),fill='white',font=('Malgun Gothic',11,'bold'))
                self.canvas.create_text(left+105,117,text=agent['sub'],fill='white',font=('Malgun Gothic',18,'bold'))
            remaining=agent['cooldown'];interval=self.data['settings'].get('shotIntervalsMs',[150,150])[i]
            ready=max(0,min(1,1-remaining/max(interval,1)))
            cooldown_top=146 if agent['sub_visible'] else 77
            self.canvas.create_rectangle(left,cooldown_top,left+210,cooldown_top+39,fill='#0f172a',outline='')
            label=f'쿨다운 {remaining:.0f} ms' if remaining>0 else '격발 가능 · 쿨다운 0'
            self.canvas.create_text(left+105,cooldown_top+13,text=label,fill='#fbbf24' if remaining>0 else '#86efac',font=('Malgun Gothic',11,'bold'))
            self.canvas.create_rectangle(left+10,cooldown_top+29,left+200,cooldown_top+35,fill='#334155',outline='')
            self.canvas.create_rectangle(left+10,cooldown_top+29,left+10+190*ready,cooldown_top+35,fill='#fbbf24' if remaining>0 else '#22c55e',outline='')
        color='#22c55e' if ratio>.5 else '#f59e0b' if ratio>.2 else '#ef4444'
        x,y,scale=self.project(f['target']);radius=self.data['radius']*scale
        target_pov=self.camera_mode.get()=='타겟 시점'
        if target_pov:x=self.canvas.winfo_width()/2;y=self.canvas.winfo_height()*.75;radius=0
        width=min(260,max(170,self.canvas.winfo_width()*.32))
        cx=min(max(x,width/2+12),max(width/2+12,self.canvas.winfo_width()-width/2-12))
        top=min(max(200,y-radius-65),max(200,self.canvas.winfo_height()-110))
        if not target_pov:self.canvas.create_line(x,y-radius,cx,top+43,fill='#94a3b8',dash=(3,3))
        self.canvas.create_rectangle(cx-width/2-7,top-27,cx+width/2+7,top+48,fill='#111827',outline='#64748b')
        self.canvas.create_text(cx,top-10,text='TARGET · '+('파괴됨' if state['hp']<=0 else '남은 체력'),fill='#e2e8f0',font=('Malgun Gothic',11,'bold'))
        self.canvas.create_rectangle(cx-width/2,top+6,cx+width/2,top+43,fill='#334155',outline='')
        self.canvas.create_rectangle(cx-width/2,top+6,cx-width/2+width*ratio,top+43,fill=color,outline='')
        self.canvas.create_text(cx,top+24,text=f"{state['hp']:,.0f} / {self.data['hp']:,.0f}",fill='white',font=('Malgun Gothic',18,'bold'))
        if state['hp']<=0:
            self.canvas.create_text(self.canvas.winfo_width()/2,35,text=state['result'],fill='#fbbf24',font=('Malgun Gothic',23,'bold'))
        criticals=[e for e in state['events'] if e.get('critical') and 0<=self.t-e['timeMs']<700]
        if criticals:
            who=' / '.join('AB'[i] for i in sorted({e['agent'] for e in criticals}))
            self.canvas.create_text(self.canvas.winfo_width()/2,140,text='CRITICAL!  '+who,fill='#fbbf24',font=('Malgun Gothic',30,'bold'),tags='critical')
        for event in state['events']:
            age=self.t-event['timeMs']
            if 0<=age<400 and event['hit']:
                self.canvas.create_text(x+30,y+20-age*.10,text=f"{'AB'[event['agent']]} −{event['damage']:.0f}",fill=('#69baff','#ffb560')[event['agent']],font=('Malgun Gothic',15,'bold'))
    def line(self,a,b,color,width=1):
        if self.camera_mode.get()=='타겟 시점':
            za=self.view_coordinates(a)[2];zb=self.view_coordinates(b)[2];near=self.distance*.02
            if za<near and zb<near:return
            if za<near:
                fraction=(near-za)/(zb-za);a=[a[i]+fraction*(b[i]-a[i]) for i in range(3)]
            elif zb<near:
                fraction=(near-zb)/(za-zb);b=[b[i]+fraction*(a[i]-b[i]) for i in range(3)]
        self.canvas.create_line(*self.project(a)[:2],*self.project(b)[:2],fill=color,width=width)
    def draw_floor_axes(self):
        span=self.distance*.45
        for axis,color in ((0,'#87606a'),(1,'#5d8274')):
            negative=[0.,0.,0.];positive=[0.,0.,0.]
            negative[axis]=-span;positive[axis]=span
            self.line(negative,positive,color)
            for sign,point in (('-',negative),('+',positive)):
                x,y,_=self.project(point)
                self.canvas.create_text(x,y+14,text=sign+'XY'[axis],fill=color,font=('Consolas',10),tags='floor_axis')
    def dot(self,p,r,color):
        if self.camera_mode.get()=='타겟 시점' and self.view_coordinates(p)[2]<self.distance*.02:return
        x,y,s=self.project(p);radius=max(2,r*s);self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,outline='')
    def ring(self,center,r,color):
        n=[v/self.distance for v in center];ref=[0,0,1] if abs(n[2])<.9 else [0,1,0]
        def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        u=cross(n,ref);length=math.sqrt(sum(x*x for x in u));u=[x/length for x in u];v=cross(n,u);last=None
        for i in range(65):
            a=i/64*math.tau;p=[center[k]+r*(u[k]*math.cos(a)+v[k]*math.sin(a)) for k in range(3)]
            if last:self.line(last,p,color,2)
            last=p
    def tick(self):
        if self.closed:return
        now=time.perf_counter()
        before=self.t
        if self.playing:self.t+=(now-self.previous)*1000*float(self.speed.get())
        if self.playing and self.pause_on_main.get():
            times=[d['timeMs'] for stream in self.data.get('originalDecisions',[[],[]]) for d in stream if d.get('requested') and before<d['timeMs']<=min(self.t,self.duration)]
            if times:self.t=min(times);self.playing=False
        self.previous=now
        if self.t>=self.duration:
            self.t=self.duration
            if not self.loop.get():
                self.playing=False;self.end_hold=None
            elif self.playing:
                if self.end_hold is None:self.end_hold=now+1.5
                elif now>=self.end_hold:self.t=0.;self.end_hold=None
        self.button.configure(text='일시정지' if self.playing else '재생')
        self.canvas.delete('all');i=max(0,min(len(self.frames)-1,bisect.bisect_right(self.times,self.t)-1));f=self.frames[i];zero=[0,0,0]
        self.camera_target=f['target'];self.fit_camera()
        self.draw_floor_axes()
        for j in range(max(1,i-120),i+1):self.line(self.frames[j-1]['target'],self.frames[j]['target'],'#495d7c')
        self.dot(zero,self.distance*.016,'white');x,y,_=self.project(zero);self.canvas.create_text(x,y+20,text='A / B',fill='white')
        if self.camera_mode.get()!='타겟 시점':
            self.dot(f['target'],self.data['radius'],'#234d42');self.ring(f['target'],self.data['radius'],'#8ce7b4');self.ring(f['target'],self.data['radius']*self.data['critical'],'#ff4e6b')
        for k,p in enumerate(f['aims']):self.line(zero,p,('#69baff','#ffb560')[k]);self.dot(p,self.distance*.007,('#69baff','#ffb560')[k])
        for shot in self.data['shots']:
            if 0<=self.t-shot['t']<180:self.line(zero,shot['point'],('#69baff','#ffb560')[shot['agent']],4)
        events=[e for e in self.data['events'] if e['timeMs']<=self.t];hp=events[-1]['hpAfter'] if events else self.data['hp']
        shots=[s for s in self.data['shots'] if s['t']<=self.t]
        self.status.set(f'{self.t/1000:.2f} / {self.duration/1000:.2f}초    남은 체력 {hp:g}    A 발사 {sum(s["agent"]==0 for s in shots)} / B 발사 {sum(s["agent"]==1 for s in shots)}')
        self.draw_target_hud(f)
        self.update_status(f)
        self.setting_seek=True;self.seek.set(self.t);self.setting_seek=False
        if self.root.winfo_exists():self.after_id=self.root.after(16,self.tick)

def main():
    p=argparse.ArgumentParser();p.add_argument('record',nargs='?');p.add_argument('--records');p.add_argument('--pause-on-main',action='store_true');args=p.parse_args()
    records=json.loads(Path(args.records).read_text(encoding='utf-8-sig')) if args.records else [args.record]
    groups=None
    if isinstance(records,dict):
        groups=records.get('groups')
        if not groups or not all(g.get('settings') and all(s.get('records') for s in g['settings']) for g in groups):
            p.error('Select at least one replay in each group and setting')
        records=[r['path'] for r in groups[0]['settings'][0]['records']]
    if not records or not all(records):p.error('Select at least one replay')
    root=tk.Tk();Replay(root,records[0],records=records,groups=groups,pause_on_main=args.pause_on_main)
    if args.pause_on_main:root.title('경기 행동 기록 · 메인 발사 요청 확인')
    root.mainloop()
if __name__=='__main__':main()
