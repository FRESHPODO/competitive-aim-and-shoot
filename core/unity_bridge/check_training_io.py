"""Regression: status reader locks must not terminate PPO training."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unity_bridge.atomic_io import replace_with_retry
from unity_bridge.train_decision_ppo import SaveProgress


def main():
    base=Path(__file__).resolve().parents[1]/'local-results'/'training-io-check'
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as folder:
        assert Path(folder).absolute().is_relative_to(base.absolute())
        directory=Path(folder)
        source,destination=directory/'new',directory/'old'
        source.write_text('new')
        destination.write_text('old')
        import os
        actual=os.replace
        calls=[]
        def flaky(a,b):
            calls.append(1)
            if len(calls)<3: raise PermissionError('simulated reader lock')
            actual(a,b)
        with patch('unity_bridge.atomic_io.os.replace',side_effect=flaky):
            replace_with_retry(source,destination,delay=0)
        assert len(calls)==3 and destination.read_text()=='new'
        callback=SaveProgress(directory,1000)
        callback.model=SimpleNamespace(ep_info_buffer=[],logger=SimpleNamespace(name_to_value={}))
        callback.num_timesteps=16
        with patch('unity_bridge.train_decision_ppo.replace_with_retry',side_effect=PermissionError('persistent lock')):
            assert callback._on_step() is True
        callback.write_status('running')
        assert (directory/'progress.json').exists()
    print('PASS: transient lock retried; persistent progress lock does not stop learning; next update recovers.')


if __name__=='__main__': main()
