"""Validate streamed timing, coordinate mapping and shot geometry against source."""
import json
import numpy as np
from unity_bridge.server import Session, DEFAULTS
from src.utils.mymath import Convert, angle_between_vectors


def main():
    session = Session()
    results = []
    for seed in range(1, 7):
        packet = session.reset(dict(seed=seed, parameters=DEFAULTS,
                                    targetSpeed=0 if seed % 2 else 12, targetRadius=1.5))
        last_time = 0
        for _ in range(50):
            packet = session.step()
            frames = packet['frames']
            assert frames[0]['timeMs'] >= last_time
            assert all(a['timeMs'] <= b['timeMs'] for a, b in zip(frames, frames[1:]))
            json.dumps(packet, allow_nan=False)
            last_time = packet['timeMs']
            if packet['done']:
                break
        assert packet['done']
        if not packet['truncated']:
            final = packet['frames'][-1]
            camera = Convert.sphr2cart_scalar(final['cameraAz'], final['cameraEl'])
            target = Convert.sphr2cart_scalar(final['targetAz'], final['targetEl'])
            geometry_error = float(angle_between_vectors(camera, target, return_in_degree=True))
            assert abs(geometry_error-packet['errorDeg']) < .02, (geometry_error, packet['errorDeg'])
            # Original hit test uses projected monitor distance, not angular radius.
            assert packet['hit'] == bool(session.agent.shoot_result)
        results.append({k: packet[k] for k in ('episode','done','hit','truncated','timeMs','errorDeg')})
    from pathlib import Path
    Path('local-results/bridge-check.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('PASS: 6 streamed episodes; finite JSON, monotonic timing, source shot geometry.')


if __name__ == '__main__':
    main()
