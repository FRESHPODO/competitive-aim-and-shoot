from pathlib import Path
from src.agent.simulator import AimandShootSimulator

if __name__ == '__main__':
    simulator = AimandShootSimulator('ijhcs_default', ckpt='latest')
    simulator.simulate(num_simul=5, num_cpu=1, deterministic=True, save_trajectory=True)
    results = simulator.get_summarized_results()
    output = Path('local-results')
    output.mkdir(exist_ok=True)
    results.to_csv(output / 'smoke-test.csv', index=False)
    print(results[['completion_time_ms', 'accuracy', 'shoot_error_mm']].to_string(index=False))
    print('Saved:', output / 'smoke-test.csv')

