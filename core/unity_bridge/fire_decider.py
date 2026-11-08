"""Request-only PPO using own aiming belief and public game configuration."""
from pathlib import Path
from unity_bridge.storage import resolve_path, DECISION_MODEL_ROOT as MODEL_DIR
import numpy as np
import torch
from unity_bridge.competitive_rewards import alpha_from_values

MODEL_PATH = MODEL_DIR / 'active.zip'
FEATURE_VERSION = 7
FEATURE_NAMES = (
    'hp_fraction','log_hp','critical_radius','target_radius','target_speed',
    'edge_damage_over_hp','center_damage_limit_over_hp','instant_kill_enabled','cooldown',
    *(f'own_parameter_{i}' for i in range(4)),
    'own_ready_in','opponent_ready_in','estimated_center_error','estimated_damage_over_hp','estimated_critical',
    'proposal_horizon','proposal_tc','proposal_delay','own_reward_alpha',
)
OBS_SIZE = len(FEATURE_NAMES)


def validate_model_observation(model):
    if (tuple(getattr(model,'decision_feature_names',())) != FEATURE_NAMES
            or model.observation_space.shape != (OBS_SIZE,) or model.action_space.n != 2):
        raise ValueError(f'PPO 관측 구조가 변경되었습니다. 현재 관측 버전 {FEATURE_VERSION}({OBS_SIZE}항목)으로 학습한 모델이 필요합니다.')


def encode_context(now, context):
    hp=max(float(context['hp']),1e-6)
    own=context['own'];opponent=context['opponent'];game=context['game']
    belief=context['belief'];proposal=context['proposal'];reward=context['own_reward']
    if not context.get('shared_attack_conditions',True):
        raise ValueError('외곽/중앙 피해, 즉사 여부, 쿨다운 길이는 공통 게임 조건이어야 합니다.')
    values=[hp/max(context['maximum_hp'],1e-6),np.log1p(hp)/np.log(1001),
            context['critical']*2,game['radius']/20.,game['speed']/40.,
            np.log1p(game['minimum']/hp),np.log1p(game['maximum']/hp),
            float(game['instantKill']),game['cooldown']/2000.]
    values.extend(np.asarray(own['profile'][:4],dtype=float)/.5)
    values.extend([max(0.,own['ready_at']-now)/2000.,max(0.,opponent['ready_at']-now)/2000.,
                   min(belief['normalized_error'],20.)/20.,np.log1p(belief['damage']/hp),float(belief['critical']),
                   proposal['th']/2000.,proposal['tc'],proposal['delay_ms']/2000.,
                   alpha_from_values(reward)])
    result=np.asarray(values,dtype=np.float32)
    if result.shape != (OBS_SIZE,) or not np.isfinite(result).all():
        raise ValueError('Invalid belief-based PPO observation')
    return result


class FireDecider:
    def __init__(self, model_path=MODEL_PATH):
        from stable_baselines3 import PPO
        self.model_path=resolve_path(model_path)
        self.model=PPO.load(self.model_path,device='cpu')
        validate_model_observation(self.model)

    def decide(self, now, context):
        observation=encode_context(now,context)
        with torch.no_grad():
            obs,_=self.model.policy.obs_to_tensor(observation)
            distribution=self.model.policy.get_distribution(obs)
            probability=float(distribution.distribution.probs[0,1])
            action=int(distribution.get_actions(deterministic=True).item())
            value=float(self.model.policy.predict_values(obs).item())
        return dict(approved=bool(action),reason='ppo',approveProbability=probability,
                    value=value,snapshotTimeMs=context['time'])
