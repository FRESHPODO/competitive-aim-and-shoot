"""Source-traceable original-agent profiles; never changes weapon/PPO rewards."""
import json
from copy import deepcopy
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'configs'/'player_presets.json'
CATALOG=json.loads(PATH.read_text(encoding='utf-8'))
PRESETS={p['id']:p for p in CATALOG['presets']}
KEYS=('motor','position','speed','clock','success','failure','reward_decay','penalty_decay')
MODES=('커스텀','프로 평균','아마추어 평균')+tuple(p['label'] for p in CATALOG['presets'] if p['kind']=='individual_condition' and p['sensitivity_mode']=='habitual' and p['target_color_level']=='white')
LABEL_TO_ID={p['label']:p['id'] for p in CATALOG['presets']}


def group_for(mode):
    return 'professional' if mode.startswith('프로') else 'amateur'


def raw_presets(group):
    return [p for p in CATALOG['presets'] if p['kind']=='individual_condition' and p['group']==group]


def apply(settings,side,preset_id):
    if side not in ('A','B'):raise ValueError('Expected agent A or B')
    result=deepcopy(settings)
    for key,value in PRESETS[preset_id]['parameters'].items():
        name=side+'.'+key
        result[name].update(mode='fixed',value=value,min=value,max=value)
    return result


def matches(settings,side,preset_id):
    return all(settings[side+'.'+key]['mode']=='fixed' and settings[side+'.'+key]['value']==value
               for key,value in PRESETS[preset_id]['parameters'].items())


def provenance(preset_id):
    if preset_id not in PRESETS:return None
    p=PRESETS[preset_id]
    return {**{k:v for k,v in p.items() if k not in ('parameters','raw_values')},
            'doi':CATALOG['doi'],'source_csv':CATALOG['source_csv'],'source_sha256':CATALOG['source_sha256']}
