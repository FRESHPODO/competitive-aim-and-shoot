"""Check that checkpoint activation changes only the next round's gate."""
from pathlib import Path
import torch
from unity_bridge.decision_models import activate, active_info, model_catalog, resolve_model
from unity_bridge.fire_decider import MODEL_DIR, MODEL_PATH
from unity_bridge.server import DualSession


def main():
    torch.set_num_threads(1)
    original=active_info()['source']
    alternate=MODEL_DIR/'monitor-check'/'candidate.zip'
    dual=DualSession()
    dual.reset({})
    previous=dual.shooters[0].decider
    try:
        activate(alternate)
        assert dual.shooters[0].decider is previous
        dual.reset({})
        assert dual.shooters[0].decider is not previous
        assert dual.shooters[0].decider.model.num_timesteps==256
        assert dual.shooters[1].decider is None
        data=MODEL_PATH.read_bytes()
        invalid=MODEL_DIR/'invalid-checkpoint.tmp.zip'
        invalid.write_bytes(b'not a PPO checkpoint')
        try:
            try:
                activate(invalid)
            except Exception:
                pass
            else:
                raise AssertionError('Invalid checkpoint was accepted')
            assert MODEL_PATH.read_bytes()==data
        finally:
            invalid.unlink(missing_ok=True)
    finally:
        activate(original)
    dual.reset({})
    assert dual.shooters[0].decider.model.num_timesteps==active_info()['decisions']
    identifier=alternate.relative_to(MODEL_DIR).as_posix()
    assert any(entry['id']==identifier for entry in model_catalog()['models'])
    assert resolve_model(identifier)==alternate
    dual.reset(dict(decisionModel=identifier))
    assert dual.shooters[0].decider.model.num_timesteps==256
    assert active_info()['source']==original
    try:
        resolve_model('../other.zip')
    except ValueError:
        pass
    else:
        raise AssertionError('Arbitrary model path accepted')
    print('PASS: checkpoint swap on next target only, B untouched, incompatible model rejected atomically, original model restored.')


if __name__=='__main__': main()
