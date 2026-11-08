"""Train one checkpoint per explicit condition, then evaluate with held-out seeds."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/competitive_conditions.json')
    parser.add_argument('--output', default='local-results/parameter-sweep')
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=421)
    parser.add_argument('--resume', help='Optional existing PPO checkpoint to fine-tune separately for every condition')
    args = parser.parse_args()
    if min(args.steps, args.rounds, args.workers) < 1:
        parser.error('steps, rounds and workers must be positive')
    conditions = json.loads(Path(args.config).read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier experiment or silently mix checkpoints/results.
    if (output/'manifest.json').exists():
        raise FileExistsError('Choose a new --output directory for this experiment')
    manifest = dict(arguments=vars(args), conditions=conditions, status='running', models=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    rows = []
    try:
        for name, scenario in conditions.items():
            if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
                raise ValueError(f'Invalid condition name: {name}')
            folder = output/name
            folder.mkdir()
            scenario_path = folder/'scenario.json'
            scenario_path.write_text(json.dumps(scenario, indent=2), encoding='utf-8')
            command = [sys.executable, '-u', '-m', 'unity_bridge.train_decision_ppo',
                       '--steps', str(args.steps), '--seed', str(args.seed),
                       '--workers', str(args.workers), '--output', str(folder)]
            if scenario is not None:
                command += ['--scenario', str(scenario_path)]
            if args.resume:
                command += ['--resume', str(Path(args.resume).resolve())]
            subprocess.run(command, check=True)
            model = folder/'candidate.zip'
            command = [sys.executable, '-u', '-m', 'unity_bridge.evaluate_decision_ppo',
                       '--model', str(model), '--rounds', str(args.rounds),
                       '--seed', str(args.seed+100000000), '--output', str(folder/'evaluation.json')]
            if scenario is not None:
                command += ['--scenario', str(scenario_path)]
            subprocess.run(command, check=True)
            summary = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))['summary']['all']
            rows.append(dict(condition=name, model=str(model), **summary))
            with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest['models'].append(dict(condition=name, checkpoint=str(model)))
            save()
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
