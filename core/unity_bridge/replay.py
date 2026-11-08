"""Export an already simulated round to a dependency-free 3D browser replay."""
import json
from pathlib import Path
import numpy as np
from src.utils.mymath import Convert


def save_replay(dual, path, settings, label, *, html=True):
    times = np.arange(0, dual.time, 1000/60).tolist() + [float(dual.time)]
    frames = []
    distance = float(np.linalg.norm(dual.motion.position(0)))
    angles_by_agent=[]
    for shooter in dual.shooters:
        ts=[f['timeMs'] for f in shooter.frames]
        angles_by_agent.append([np.interp(times,ts,[f[k] for f in shooter.frames]) for k in ('cameraAz','cameraEl')])
    for index,t in enumerate(times):
        aims = []
        for az,el in angles_by_agent:
            angles = [az[index],el[index]]
            aims.append((Convert.sphr2cart_scalar(*angles)*distance).tolist())
        frames.append(dict(t=t, target=dual.motion.position(t).tolist(), aims=aims))
    shots = []
    for shooter in dual.shooters:
        for s in shooter.shots:
            if s['timeMs'] <= dual.time:
                shots.append(dict(t=s['timeMs'], agent=s['agent'], hit=s['hit'],
                    point=(Convert.sphr2cart_scalar(s['shotAz'],s['shotEl'])*distance).tolist()))
    data = dict(label=label, settings=settings, duration=dual.time, distance=distance,
                radius=distance*np.tan(np.radians(dual.shooters[0].radius)),
                critical=dual.combat.critical, hp=dual.combat.maximum_hp,
                frames=frames, shots=shots, events=dual.combat.events,
                decisions=[[d for d in s.strategy_log if d['timeMs']<=dual.time] for s in dual.shooters],
                originalDecisions=[[d for d in s.decision_log if d['timeMs']<=dual.time] for s in dual.shooters])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(data), encoding='utf-8')
    if html:
        template = Path(__file__).with_name('replay_viewer.html').read_text(encoding='utf-8')
        path.write_text(template.replace('/*REPLAY_DATA*/', json.dumps(data).replace('<','\\u003c')), encoding='utf-8')
