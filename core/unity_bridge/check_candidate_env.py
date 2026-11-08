"""Validate the real-simulator candidate/action handshake independently of policy."""
import numpy as np
from unity_bridge.decision_env import CandidateEnv


def encode(now, context):
    return np.array([context['hp'], context['own']['error'], now], np.float32)


def main():
    env = CandidateEnv(encode, 3)
    try:
        obs, _ = env.reset(seed=19)
        assert env.observation_space.contains(obs)
        obs, _, terminal, truncated, _ = env.step(0)
        assert not env.dual.shooters[0].shots
        for _ in range(200):
            if terminal or truncated:
                break
            obs, reward, terminal, truncated, info = env.step(1)
            assert np.isfinite(reward)
        assert terminal or truncated
        assert env.dual.shooters[0].strategy_log
        assert not env.dual.shooters[1].strategy_log
        simulator = env.dual
        env.reset(seed=20)
        assert env.dual is simulator
        print('PASS: real candidate pause, rejection without shot, continuation, terminal reward, A-only gate, reset/close.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
