"""Named weapon configurations; undefined types never silently invent values."""
from copy import deepcopy
from unity_bridge.training_settings import defaults,WEAPON,TOGGLES

NAMES=('기본형','AR형','저격형','커스텀')
BASE={key:deepcopy(defaults()['A.'+key]) for key,*_ in WEAPON+TOGGLES}
def preset(edge,center,cooldown,spread,instant_kill=0):
    result=deepcopy(BASE)
    for key,value in dict(minDamage=edge,maxDamage=center,cooldown=cooldown,spread=spread,instantKill=instant_kill,spreadEnabled=1).items():
        result[key].update(mode='fixed',value=value,min=value,max=value)
    return result

# Editable initial values, not measured or balanced weapon specifications.
PRESETS={'기본형':BASE,'AR형':preset(75,125,100,1.5),'저격형':preset(400,800,1600,.2,instant_kill=1)}
