"""An accepted motor/shot plan cannot be revoked by later gate decisions."""
import json
from pathlib import Path
import numpy as np
import torch
from unity_bridge.server import DualSession, DEFAULTS
from unity_bridge.replay import save_replay
from unity_bridge.replay_native import state_at


def main():
    torch.set_num_threads(1)
    for mode in ('bump','candidate'):
        dual=DualSession(recording='replay')
        profile=list(DEFAULTS);profile[:4]=[0.,0.,0.,0.]
        dual.reset(dict(profiles=[profile,profile],targetSpeed=0,targetRadius=5,
                        weapons=[dict(minimum=1,maximum=1,instantKill=False)]*2,
                        shotIntervalsMs=[400,400],decisionEnabled=[False,False]))
        agent=dual.shooters[0].session.agent
        action=np.zeros(len(agent.action_list),dtype=np.float32)
        for key in ('th','tc','kc'):action[agent.action_list.index(key)]=1.
        actors={id(s.session.sim.model):s.session.sim.model for s in dual.shooters}
        originals={key:actor.predict for key,actor in actors.items()}
        for actor in actors.values():actor.predict=lambda *args,**kwargs:(action.copy(),None)
        calls=[[],[]]
        class Gate:
            def __init__(self,i):self.i=i
            def decide(self,now,context):
                calls[self.i].append(now)
                return dict(approved=len(calls[self.i])==1,reason='test_once_then_deny',snapshotTimeMs=context['time'])
        try:
            for i,shooter in enumerate(dual.shooters):shooter.decider=Gate(i);shooter.decision_mode=mode
            while dual.time<4000:dual.step(deadline_ms=4000)
        finally:
            for key,actor in actors.items():actor.predict=originals[key]
        for i,shooter in enumerate(dual.shooters):
            assert len(shooter.shots)==1,shooter.shots
            first=shooter.shots[0]['timeMs'];accepted=shooter.commit_spans[0][0]
            assert first-accepted>=1000
            assert not any(accepted<t<first for t in calls[i]),calls[i]
            assert any(t>first for t in calls[i])
            pending=[d for d in shooter.decision_log if accepted<d['timeMs']<first]
            assert pending and all(d['pendingPlan'] and d['ppoApproved'] is None and not d['ppoBlocked'] for d in pending)
        path=Path('storage-reports/verification/committed-plan')/f'{mode}.json'
        save_replay(dual,path,{},'승인된 발사 계획 취소 금지 검증',html=False)
        data=json.loads(path.read_text(encoding='utf-8'))
        first=dual.shooters[0].shots[0]['timeMs']
        assert state_at(data,500)['agents'][0]['plan_active']
        assert not state_at(data,first+10)['agents'][0]['plan_active']
        print(f'PASS {mode}: accepted plan executes once; no re-gating while pending; later denial blocks new shots; replay state clears after shot')


if __name__=='__main__':main()
