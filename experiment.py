#!/usr/bin/env python3
"""Frozen, reproducible synthetic discovery experiment. No model/API calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
METHODS = PROTOCOL['methods']
GOOD_C = PROTOCOL['good_ucb_coefficients']


def seed(*parts):
    data = json.dumps([PROTOCOL['master_seed'], *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def distribution(sizes, q, shared_size, overlap, order):
    """Rows indexed by physical arm; order maps physical arm to pool-size rank."""
    offsets = np.cumsum([shared_size, *sizes])
    p = np.zeros((len(sizes), shared_size + sum(sizes)))
    for arm, rank in enumerate(order):
        p[arm, :shared_size] = q * overlap / shared_size
        p[arm, offsets[rank]:offsets[rank + 1]] = q * (1 - overlap) / sizes[rank]
    return p


def uniform_exact(p, horizon):
    return np.sum(1 - np.power(1 - p.mean(axis=0), np.arange(1, horizon + 1)[:, None]), axis=1)


class Selector:
    def __init__(self, method, arms, outcomes, policy_seed):
        self.method, self.k = method, arms
        self.rng = random.Random(policy_seed)
        self.order = list(range(arms))
        self.rng.shuffle(self.order)
        self.n = [0] * arms
        self.successes = [0] * arms
        self.singleton = [0] * arms
        self.count = [0] * outcomes
        self.owner = [-1] * outcomes

    def choose(self, t, remaining_mass=None):
        if self.method == 'uniform':
            return self.rng.randrange(self.k)
        if self.method == 'round_robin':
            return self.order[(t - 1) % self.k]
        if self.method == 'oracle_greedy':
            if remaining_mass is None:
                raise ValueError('Oracle requires true remaining mass')
            scores = remaining_mass
        else:
            for arm in self.order:
                if self.n[arm] == 0:
                    return arm
            if self.method == 'success_ucb':
                logterm = 2 * math.log(t)
                scores = [self.successes[a] / n + math.sqrt(logterm / n) for a, n in enumerate(self.n)]
            else:
                counts = self.successes if self.method == 'success_ucb_matched' else self.singleton
                c = .5 if self.method == 'success_ucb_matched' else GOOD_C[self.method]
                logterm = math.log(4 * t)
                scores = [counts[a] / n + c * math.sqrt(logterm / n) for a, n in enumerate(self.n)]
        top = max(scores)
        tied = [a for a, value in enumerate(scores) if abs(value - top) < 1e-12]
        return self.rng.choice(tied)

    def update(self, arm, outcome):
        self.n[arm] += 1
        if outcome < 0:
            return 0
        self.successes[arm] += 1
        previous = self.count[outcome]
        if previous == 0:
            self.singleton[arm] += 1
            self.owner[outcome] = arm
        elif previous == 1:
            self.singleton[self.owner[outcome]] -= 1
            self.owner[outcome] = -1
        self.count[outcome] += 1
        return int(previous == 0)


def sample_streams(p, horizon, stream_seed):
    # An extra no-outcome column completes each categorical distribution.
    cdf = np.cumsum(np.column_stack([p, 1 - p.sum(axis=1)]), axis=1)
    cdf[:, -1] = 1.0
    uniforms = np.random.default_rng(stream_seed).random((len(p), horizon))
    draws = np.stack([np.searchsorted(cdf[a], uniforms[a], side='right') for a in range(len(p))])
    draws[draws == p.shape[1]] = -1
    return draws.astype(np.int16)


def environment_specs(family):
    spec = PROTOCOL[family]
    if family == 'canonical':
        return [{'environment': 0, 'sizes': spec['sizes'], 'q': spec['success_probability']}]
    envs = []
    for e in range(spec['environments']):
        rng = np.random.default_rng(seed('environment', e))
        sizes = [int(rng.integers(lo, hi + 1)) for lo, hi in spec['size_ranges_inclusive']]
        q = float(rng.uniform(*spec['success_probability_range']))
        envs.append({'environment': e, 'sizes': sizes, 'q': q})
    return envs


def run_condition(family, overlap, output):
    spec, horizon = PROTOCOL[family], PROTOCOL['horizon']
    envs = environment_specs(family)
    reps = spec.get('replicates', spec.get('replicates_per_environment'))
    runs = len(envs) * reps
    actions = np.zeros((runs, len(METHODS), horizon), dtype=np.uint8)
    outcomes = np.full(actions.shape, -1, dtype=np.int16)
    novelty = np.zeros(actions.shape, dtype=np.uint8)
    streams = np.zeros((runs, 3, horizon), dtype=np.int16)
    metadata = []
    exact = []
    for e in envs:
        for r in range(reps):
            index = len(metadata)
            permutation_seed = seed('arm permutation', family, e['environment'], r if family == 'canonical' else 0)
            order = np.random.default_rng(permutation_seed).permutation(3).tolist()
            p = distribution(e['sizes'], e['q'], spec['shared_size'], overlap, order)
            stream_seed = seed('outcomes', family, e['environment'], r)
            policy_seed = seed('policy', family, e['environment'], r)
            draws = sample_streams(p, horizon, stream_seed)
            streams[index] = draws
            metadata.append({**e, 'replicate': r, 'order': order, 'stream_seed': stream_seed, 'policy_seed': policy_seed})
            exact.append(uniform_exact(p, horizon))
            for m, method in enumerate(METHODS):
                state = Selector(method, 3, p.shape[1], policy_seed)
                remaining = p.sum(axis=1).tolist()
                for step in range(horizon):
                    arm = state.choose(step + 1, remaining if method == 'oracle_greedy' else None)
                    outcome = int(draws[arm, state.n[arm]])
                    new = state.update(arm, outcome)
                    if new and method == 'oracle_greedy':
                        for a in range(3):
                            remaining[a] -= float(p[a, outcome])
                    actions[index, m, step] = arm
                    outcomes[index, m, step] = outcome
                    novelty[index, m, step] = new
    # All files are written only after a condition is complete, never appended
    # across tool yields. Each compressed array also contains the source streams.
    name = f'{family}_overlap{overlap:g}'
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / 'traces.npz', actions=actions, outcomes=outcomes,
                        novelty=novelty, streams=streams, exact_uniform=np.stack(exact))
    dump(target / 'metadata.json', {'family': family, 'overlap': overlap, 'methods': METHODS, 'runs': metadata})
    return {'condition': name, 'replicates': runs, 'trajectories': runs * len(METHODS),
            'interactions': int(actions.size)}


def freeze():
    path = ROOT / 'FROZEN_MANIFEST.json'
    if path.exists():
        raise FileExistsError('Freeze already exists; do not overwrite it')
    files = ['protocol.json', 'experiment.py', 'test_experiment.py', 'analyze.py']
    dump(path, {'frozen_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                'sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}})


def verify_frozen():
    manifest = json.loads((ROOT / 'FROZEN_MANIFEST.json').read_text())
    for f, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == expected, f'Frozen source changed: {f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        print('Protocol and source frozen.', flush=True)
        return
    verify_frozen()
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    status = []
    start = time.time()
    for family in ['canonical', 'generated']:
        for overlap in PROTOCOL[family]['overlaps']:
            result = run_condition(family, overlap, output)
            status.append(result)
            dump(ROOT / 'RUN_STATUS.json', {'complete': False, 'conditions': status})
            print(json.dumps({**result, 'elapsed_seconds': round(time.time() - start, 1)}), flush=True)
    dump(ROOT / 'RUN_STATUS.json', {'complete': True, 'conditions': status,
                                  'interactions': sum(s['interactions'] for s in status),
                                  'elapsed_seconds': time.time() - start})


if __name__ == '__main__':
    main()
