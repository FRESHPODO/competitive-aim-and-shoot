"""Load either observation schema for saved-model duels."""
import json
import zipfile

from unity_bridge.fire_decider import FireDecider
from unity_bridge.reward_generation.decider import FireDecider as RewardFireDecider, FEATURE_NAMES
from unity_bridge.storage import resolve_path


class DuelPolicy:
    def __init__(self, path):
        path = resolve_path(path)
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read('data'))
        self.extended = tuple(metadata.get('decision_feature_names', ())) == FEATURE_NAMES
        self.decider = (RewardFireDecider if self.extended else FireDecider)(path)
        self.mode = getattr(self.decider.model, 'decision_mode', 'bump' if self.extended else 'candidate')
        self.reward = {}

    def decide(self, now, context):
        if self.extended:
            context = dict(context, own_reward={**context['own_reward'], **self.reward})
        return self.decider.decide(now, context)


def bind_duel_policies(dual, policies, request):
    for side, shooter, policy in zip('AB', dual.shooters, policies):
        reward = request.get('rewardValues', {}).get(side, {})
        policy.reward = dict(damageFactor=reward.get('damageFactor', 0.),
                             criticalFactor=reward.get('criticalFactor', 0.))
        shooter.decider = policy
        shooter.decision_mode = policy.mode
