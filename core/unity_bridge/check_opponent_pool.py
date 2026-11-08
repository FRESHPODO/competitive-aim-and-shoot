"""Regression checks for opponent selection, observed profiles and frozen inference."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import cloudpickle
import torch
from stable_baselines3 import PPO

from unity_bridge import curriculum_workers as rollout_workers
from unity_bridge.competitive_rewards import defaults as rewards
from unity_bridge.fire_decider import FEATURE_NAMES, FEATURE_VERSION
from unity_bridge.opponent_pool import normalize, prepare, scenario, select
from unity_bridge.player_presets import PRESETS, KEYS
from unity_bridge.train_curriculum import PolicySpace
from unity_bridge.training_settings import defaults


class OpponentPoolTests(unittest.TestCase):
    def test_cycle_and_profile_override(self):
        config = dict(enabled=True, opponents=[
            dict(name='pro',model='pro.zip',presetId='professional_mean'),
            dict(name='amateur',model='amateur.zip',presetId='amateur_mean')])
        config=normalize(config,check_files=False)
        self.assertEqual([select(config,i)['index'] for i in range(3,7)], [1,0,1,0])
        settings=defaults(); original=deepcopy(settings)
        entry=select(config,1)
        updated,presets=scenario(settings,{},entry)
        self.assertEqual(settings,original)
        self.assertEqual(presets['B']['id'],'amateur_mean')
        for key in KEYS:
            self.assertEqual(updated['B.'+key]['value'],PRESETS['amateur_mean']['parameters'][key])
            self.assertEqual(updated['A.'+key],settings['A.'+key])
        with self.assertRaises(ValueError):
            normalize(dict(enabled=True,opponents=[]))

    def test_opponent_never_generates_training_records_or_changes_weights(self):
        torch.set_num_threads(1)
        scratch=Path(__file__).resolve().parents[1]/'local-results'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='pool-regression-') as folder:
            folder=Path(folder)
            model=PPO('MlpPolicy',PolicySpace(),seed=42,device='cpu',n_steps=4,batch_size=4)
            model.decision_feature_names=FEATURE_NAMES
            model.decision_feature_version=FEATURE_VERSION
            model.decision_mode='bump'
            source=folder/'original.zip';model.save(source)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            config=prepare(dict(enabled=True,opponents=[dict(name='pro',model=str(source),presetId='professional_mean')]),folder)
            blob=cloudpickle.dumps([model.policy,model.policy])
            weights=cloudpickle.dumps([model.policy.state_dict(),model.policy.state_dict()])
            rollout_workers.initialize(blob,folder/'stop.request')
            records,row=rollout_workers.collect((421,defaults(),rewards(),{},dict(A=True,B=True),.4,weights,select(config,0)))
            self.assertTrue(records[0])
            self.assertEqual(records[1],[])
            self.assertFalse(row['learningEnabled']['B'])
            self.assertGreater(row['decisions'][1],0)
            self.assertEqual(row['settings']['profiles'][1],
                             [PRESETS['professional_mean']['parameters'][key] for key in KEYS])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            opponent=next(iter(rollout_workers._opponents.values())).model
            self.assertEqual(opponent._n_updates,0)
            for name,value in model.policy.state_dict().items():
                self.assertTrue(torch.equal(value,opponent.policy.state_dict()[name]))


if __name__=='__main__':
    unittest.main()
