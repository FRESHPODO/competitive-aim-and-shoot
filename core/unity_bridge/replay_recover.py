"""Reconstruct only missing legacy replay records from archived models/settings."""
import argparse,json
from pathlib import Path
import torch
from unity_bridge.duel_policy import DuelPolicy, bind_duel_policies
from unity_bridge.server import DualSession
from unity_bridge.replay import save_replay
from unity_bridge.trainer_state import save_state
from unity_bridge.storage import resolve_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--rounds',nargs='+',type=int,required=True);a=p.parse_args()
    folder=resolve_path(a.run);rows={r['round']:r for r in (json.loads(line) for line in (folder/'rounds.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    torch.set_num_threads(1);policies=[DuelPolicy(folder/f'agent_{side}.zip') for side in ('a','b')];dual=DualSession()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'));timeout=manifest['timeoutSeconds']
    for number in a.rounds:
        row=rows[number];destination=folder/'replays'/f'round-{number:05d}.json'
        if destination.exists():continue
        request=row['settings'];dual.round=row['seed'];dual.reset(request);dual.inactivity_limit_ms=timeout*1000
        bind_duel_policies(dual,policies,request)
        while dual.time<timeout*1000 and dual.combat.hp>0 and not dual.draw:dual.step(deadline_ms=timeout*1000)
        save_replay(dual,destination,request,f'대결 {number} · 저장 조건으로 재구성',html=False)
        data=json.loads(destination.read_text(encoding='utf-8-sig'));data['reconstruction']=dict(seed=row['seed'],expectedTimeMs=row['timeMs'],actualTimeMs=dual.time,matchingEndTime=abs(row['timeMs']-dual.time)<1e-6)
        save_state(destination,data);print(f'Reconstructed replay {number}',flush=True)
if __name__=='__main__':main()
