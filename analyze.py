#!/usr/bin/env python3
"""Audit complete traces, calculate frozen comparisons, and render a report."""
import base64
from collections import Counter
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiment import ROOT, PROTOCOL, METHODS, Selector, distribution, sample_streams, seed, dump, verify_frozen

LABELS = {'uniform': 'Uniform', 'round_robin': 'Round-robin', 'success_ucb': 'Success-UCB',
          'success_ucb_matched': 'Success-UCB (matched C)', 'good_0.1': 'Good-UCB C=0.1',
          'good_0.5': 'Good-UCB C=0.5', 'good_1': 'Good-UCB C=1',
          'good_theory': 'Good-UCB C=4.18', 'oracle_greedy': 'Known-p greedy'}
PRIMARY = METHODS.index(PROTOCOL['primary_method'])
UNIFORM = METHODS.index('uniform')
CHECKS = PROTOCOL['checkpoints']
DISPLAY = ['uniform', 'round_robin', 'success_ucb', 'good_0.5', 'oracle_greedy']
COLORS = dict(zip(DISPLAY, ['#6a7685', '#b07f2c', '#b85b5b', '#168472', '#5646ad']))


def audit(data, meta):
    actions, outcomes, novelty, streams = (data[n] for n in ['actions', 'outcomes', 'novelty', 'streams'])
    spec = PROTOCOL[meta['family']]
    expected_runs = spec['replicates'] if meta['family'] == 'canonical' else spec['environments'] * spec['replicates_per_environment']
    assert actions.shape == (expected_runs, len(METHODS), PROTOCOL['horizon'])
    assert outcomes.shape == novelty.shape == actions.shape
    assert len(meta['runs']) == expected_runs and meta['methods'] == METHODS
    unique_runs = set()
    for r, row in enumerate(meta['runs']):
        key = row['environment'], row['replicate']
        assert key not in unique_runs
        unique_runs.add(key)
        p = distribution(row['sizes'], row['q'], spec['shared_size'], meta['overlap'], row['order'])
        np.testing.assert_array_equal(streams[r], sample_streams(p, PROTOCOL['horizon'], row['stream_seed']))
        for m, method in enumerate(METHODS):
            state = Selector(method, 3, p.shape[1], row['policy_seed'])
            counts, owner = Counter(), {}
            remaining = p.sum(axis=1).tolist()
            for t, (arm, outcome, new) in enumerate(zip(actions[r, m], outcomes[r, m], novelty[r, m]), 1):
                arm, outcome, new = int(arm), int(outcome), int(new)
                assert arm == state.choose(t, remaining if method == 'oracle_greedy' else None)
                assert outcome == streams[r, arm, state.n[arm]]
                expected_new = outcome >= 0 and outcome not in counts
                assert new == expected_new
                if outcome >= 0:
                    assert p[arm, outcome] > 0
                    counts[outcome] += 1
                    owner[outcome] = arm
                assert state.update(arm, outcome) == new
                if new and method == 'oracle_greedy':
                    for a in range(3):
                        remaining[a] -= float(p[a, outcome])
            independently_counted_singletons = [sum(n == 1 and owner[c] == a for c, n in counts.items()) for a in range(3)]
            assert state.singleton == independently_counted_singletons
            assert sum(state.successes) == sum(counts.values())
            assert sum(state.n) == PROTOCOL['horizon']
    return expected_runs * len(METHODS)


def units(values, meta):
    if meta['family'] == 'canonical':
        return values
    group = np.array([r['environment'] for r in meta['runs']])
    return np.stack([values[group == e].mean(axis=0) for e in sorted(set(group))])


def summarize(data, meta):
    coverage = np.cumsum(data['novelty'], axis=2)
    endpoint = units(coverage[:, :, np.array(CHECKS) - 1], meta)
    diff = endpoint - endpoint[:, UNIFORM:UNIFORM + 1]
    rng = np.random.default_rng(seed('bootstrap', meta['family'], meta['overlap']))
    n = len(endpoint)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=PROTOCOL['bootstrap_resamples']) / n
    boot = (weights @ endpoint.reshape(n, -1)).reshape((-1, len(METHODS), len(CHECKS)))
    bootdiff = boot - boot[:, UNIFORM:UNIFORM + 1]
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    dlo, dhi = np.quantile(bootdiff, [.025, .975], axis=0)
    means, deltas = endpoint.mean(axis=0), diff.mean(axis=0)
    records = []
    for m, method in enumerate(METHODS):
        for b, budget in enumerate(CHECKS):
            d = diff[:, m, b]
            pval = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else (1. if d.mean() == 0 else 0.)
            records.append({'method': method, 'budget': budget, 'mean': float(means[m, b]),
                            'mean_ci': [float(lower[m, b]), float(upper[m, b])],
                            'delta_uniform': float(deltas[m, b]), 'delta_ci': [float(dlo[m, b]), float(dhi[m, b])],
                            'p_unadjusted': pval})
    allocations = np.zeros((len(meta['runs']), len(METHODS), 3))
    b = PROTOCOL['primary_budget']
    for r, row in enumerate(meta['runs']):
        ranks = np.asarray(row['order'])[data['actions'][r, :, :b]]
        for rank in range(3):
            allocations[r, :, rank] = (ranks == rank).mean(axis=1)
    asr = units((data['outcomes'][:, :, :b] >= 0).mean(axis=2), meta).mean(axis=0)
    alloc = units(allocations, meta).mean(axis=0)
    auc = units(coverage.mean(axis=2), meta).mean(axis=0)
    exact = units(data['exact_uniform'], meta).mean(axis=0)
    uniform_values = endpoint[:, UNIFORM, CHECKS.index(b)]
    exact_units = units(data['exact_uniform'][:, b - 1], meta)
    err = uniform_values - exact_units
    se = float(stats.sem(err))
    return {'family': meta['family'], 'overlap': meta['overlap'], 'replicates': len(meta['runs']),
            'resampling_units': n, 'records': records,
            'allocation_by_size_rank_at150': {method: alloc[m].tolist() for m, method in enumerate(METHODS)},
            'observed_success_rate_at150': dict(zip(METHODS, asr.tolist())),
            'mean_coverage_over_300_queries': dict(zip(METHODS, auc.tolist())),
            'exact_uniform_at_checkpoints': {str(c): float(exact[c - 1]) for c in CHECKS},
            'uniform_minus_exact_at150': {'mean': float(err.mean()), 'standard_error': se,
                                         'z_score': float(err.mean() / se) if se else 0.}}, coverage


def record(summary, method='good_0.5', budget=150):
    return next(r for r in summary['records'] if r['method'] == method and r['budget'] == budget)


def plots(all_results, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (name, (summary, coverage, data, meta)) in zip(axes.flat[:5], list(all_results.items())[:5]):
        for method in DISPLAY:
            m = METHODS.index(method)
            mean = coverage[:, m].mean(axis=0)
            sem = coverage[:, m].std(axis=0, ddof=1) / np.sqrt(len(coverage))
            ax.plot(np.arange(1, 301), mean, color=COLORS[method], label=LABELS[method], linewidth=1.8)
            ax.fill_between(np.arange(1, 301), mean - 1.96 * sem, mean + 1.96 * sem,
                            color=COLORS[method], alpha=.1, linewidth=0)
        ax.axvline(150, linestyle=':', color='#8e9aa3', linewidth=1)
        ax.set(title=f"Shared-outcome probability: {summary['overlap']:.0%}", xlabel='Queries', ylabel='Distinct outcomes')
        ax.grid(alpha=.15)
    ax = axes.flat[5]
    _, _, data, meta = all_results['canonical_overlap0']
    for method in DISPLAY:
        m = METHODS.index(method)
        rich = np.stack([np.asarray(r['order'])[data['actions'][i, m]] == 2 for i, r in enumerate(meta['runs'])])
        cumulative = np.cumsum(rich, axis=1) / np.arange(1, 301)
        ax.plot(np.arange(1, 301), cumulative.mean(axis=0), color=COLORS[method], label=LABELS[method])
    ax.set(title='Disjoint case: allocation to 100-outcome arm', xlabel='Queries', ylabel='Cumulative share of queries', ylim=(0, 1))
    ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.suptitle('Equal success rates, unequal discovery opportunities', fontsize=17, x=.04, ha='left')
    fig.text(.04, .935, 'Synthetic outcomes • 400 paired replicates • bands: pointwise normal 95% mean intervals', color='#55616d')
    fig.tight_layout(rect=[0, .055, 1, .92])
    fig.savefig(output / 'discovery_curves.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, family in zip(axes, ['canonical', 'generated']):
        ss = [s for s, *_ in all_results.values() if s['family'] == family]
        xs = [s['overlap'] for s in ss]
        ys = [record(s)['delta_uniform'] for s in ss]
        low = [y - record(s)['delta_ci'][0] for s, y in zip(ss, ys)]
        high = [record(s)['delta_ci'][1] - y for s, y in zip(ss, ys)]
        ax.errorbar(xs, ys, yerr=[low, high], fmt='o-', color='#168472', capsize=5)
        ax.axhline(0, color='#8e9aa3', linestyle='--')
        ax.set(title='Fixed pool sizes' if family == 'canonical' else '40 generated environments',
               xlabel='Shared-outcome probability', ylabel='Additional distinct outcomes vs uniform', xticks=xs)
        ax.grid(alpha=.15)
    fig.suptitle('Good-UCB C=0.5 at 150 queries', fontsize=15, x=.06, ha='left')
    fig.text(.06, .885, '95% paired bootstrap intervals; generated environments resampled as clusters.', color='#55616d')
    fig.tight_layout(rect=[0, 0, 1, .87])
    fig.savefig(output / 'overlap_effect.png', dpi=170)
    plt.close(fig)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def report(all_results, output, total_traces):
    main = all_results['canonical_overlap0'][0]
    m = record(main)
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Discovery experiment v3</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#20313c;max-width:1160px;margin:45px auto;padding:0 24px}h1,h2{line-height:1.2}h1{font-size:36px}h2{margin-top:42px}p{max-width:980px}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid #dce3e6}th:first-child,td:first-child{text-align:left}th{background:#edf3f5}img{width:100%;height:auto}.callout{padding:22px;background:#e9f5f1;border-left:5px solid #168472}.small{font-size:13px;color:#586975}code{background:#eef2f4;padding:2px 5px}</style></head><body>']
    parts += ['<p class="small">REPRODUCIBLE RESEARCH · SYNTHETIC DISCOVERY · V3</p><h1>Equal success, unequal diversity</h1>',
              f'<p class="callout">At 150 queries in the disjoint 2/20/100-outcome setting, primary Good-UCB discovered <b>{m["mean"]:.2f}</b> distinct outcomes, a difference of <b>{m["delta_uniform"]:+.2f}</b> against uniform (95% paired CI {m["delta_ci"][0]:+.2f} to {m["delta_ci"][1]:+.2f}). This is a constructed mechanism experiment, not an LLM evaluation.</p>',
              '<h2>What changed and what was fixed</h2><p>Each generator succeeds with probability 0.6. On success it samples uniformly from a private pool of 2, 20, or 100 outcomes, or with probability λ from a shared 20-outcome pool. Arms are randomly relabeled. At λ=0 all private supports are disjoint; at λ=1 all arms have exactly the same distribution. This separates success probability from diversity and introduces a controlled path to a negative control.</p>',
              '<p>The protocol, implementation, tests, and analysis source were hashed before execution. The primary coefficient was fixed at C=0.5, the primary budget at 150, and all conditions ran to 300. Five canonical overlap levels used 400 paired replicates each. A separate family used 40 unfiltered generated environments, each with eight replicates and overlap 0, 0.5, or 1. Pool sizes were drawn from [2,5], [15,30], [80,140], and common per-arm success probability from [0.45,0.75]. This is robustness within one designed family; it is not an independent real-world benchmark.</p>',
              '<h2>Primary comparisons at 150 queries</h2>']
    rows = []
    for name, (s, *_rest) in all_results.items():
        r = record(s)
        rows.append([name, *[f'{record(s, method)["mean"]:.2f}' for method in DISPLAY],
                     f'{r["delta_uniform"]:+.2f} [{r["delta_ci"][0]:+.2f}, {r["delta_ci"][1]:+.2f}]', f'{r["p_holm_8"]:.3g}'])
    parts.append(table(['Condition', *[LABELS[x] for x in DISPLAY], 'Good − uniform [95% CI]', 'Holm p'], rows))
    parts.append('<p class="small">Intervals are paired percentile bootstrap intervals, not multiplicity-adjusted. Holm p-values adjust the eight predeclared primary-method-versus-uniform comparisons using paired two-sided t-tests. Canonical units are replicate seeds; generated units are environment means. Pairing shares iid per-arm outcome streams across algorithms, indexed by the number of pulls of each arm.</p>')
    for filename in ['discovery_curves.png', 'overlap_effect.png']:
        encoded = base64.b64encode((output / filename).read_bytes()).decode()
        parts.append(f'<img alt="{filename}" src="data:image/png;base64,{encoded}">')
    parts += ['<h2>Coefficient sensitivity and allocation</h2><p>All coefficients were specified before the run. Sensitivity results are descriptive; no coefficient is selected retrospectively as the primary method. The theory-derived coefficient is included for context, and its theoretical assumptions do not extend automatically to overlapping supports.</p>']
    rows = []
    for method in METHODS:
        allocation = main['allocation_by_size_rank_at150'][method]
        r = record(main, method)
        rows.append([LABELS[method], f'{r["mean"]:.2f}', f'{record(main, method, 300)["mean"]:.2f}',
                     f'{r["delta_uniform"]:+.2f} [{r["delta_ci"][0]:+.2f}, {r["delta_ci"][1]:+.2f}]',
                     ' / '.join(f'{x:.1%}' for x in allocation), f'{main["observed_success_rate_at150"][method]:.1%}'])
    parts.append(table(['Method', 'Distinct @150', 'Distinct @300', 'Δ uniform @150 [95% CI]', 'Query share: 2 / 20 / 100', 'Observed success @150'], rows))
    parts += ['<p>Good-UCB estimates each arm’s remaining discovery probability as its globally unique singleton outcomes divided by all pulls of that arm, including unsuccessful pulls. A second observation from any arm removes that outcome’s singleton. Its index adds C√(log(4t)/n). Success-UCB uses the binary success rate; a matched-bonus version uses the same C=0.5 exploration bonus. Neither learns diversity directly from binary success alone. The known-p greedy comparator receives the true remaining probability mass, information unavailable to the other methods. It is not asserted to be a universal upper bound under overlap.</p>',
              '<h2>Analytic and implementation checks</h2><p>For iid uniform arm selection, expected coverage is Σc[1−(1−mean_k p(k,c))^B]. In the canonical disjoint case this gives 43.5114 outcomes at B=150. Always pulling the rich arm gives 59.4531 in expectation; this fixed informed policy is not the same as known-p greedy.</p>']
    rows = []
    for name, (s, *_rest) in all_results.items():
        diagnostic = s['uniform_minus_exact_at150']
        rows.append([name, f'{s["exact_uniform_at_checkpoints"]["150"]:.3f}', f'{record(s, "uniform")["mean"]:.3f}',
                     f'{diagnostic["mean"]:+.3f}', f'{diagnostic["z_score"]:+.2f}'])
    parts.append(table(['Condition', 'Exact uniform expectation', 'Observed uniform', 'Difference', 'Difference / SE'], rows))
    parts.append(f'<p>All {total_traces:,} complete trajectories were replayed, including each action, per-arm sample index, outcome, novelty label, and final global singleton counts. Stored streams were regenerated from their seeds. Tests cover analytic expectations, equal success probabilities, the identical-arm limit, global singleton updates across arms, the Good–Turing expectation identity, and reproducible sampling.</p>')
    parts += ['<h2>Interpretation and limits</h2><p>This benchmark deliberately supplies a strong difference in discovery opportunities. Its purpose is to test whether a discovery-aware selector can exploit that difference and whether the benefit shrinks as the arms become equivalent. It does not erase the earlier near-null v1/v2 findings. Those benchmarks and these ones answer different questions.</p><p>Distinct outcomes here are synthetic IDs with perfect observation. They are not validated semantic vulnerabilities or independent root causes. There are no prompt generators, model responses, classifier errors, drift, or API costs. A next empirical experiment should fix a meaningful failure taxonomy and evaluate whether real generators exhibit measurable differences in repetition and unseen-outcome probability. Equal success rates and severe diversity heterogeneity are assumptions of this demonstration, not established properties of real generators.</p>',
              '<p>Primary uncertainty intervals use 5,000 paired bootstrap resamples. The generated-environment analysis resamples 40 independent environments after averaging their eight replicates. Curve bands are pointwise normal mean intervals and are descriptive. Secondary methods, budgets, and coefficients are not a new collection of confirmatory tests.</p>',
              '<h2>Reproduction and contents</h2><p>From this directory run <code>python -m unittest -v test_experiment.py</code>, <code>python experiment.py</code>, then <code>python analyze.py</code>. Install the versions recorded in <code>requirements.txt</code>. The frozen manifest is verified before execution and analysis. Each condition includes metadata and compact arrays for actions, outcomes, novelty, source streams, and exact uniform expectations. <code>analysis/summary.json</code> contains every algorithm, checkpoint, interval, and primary adjusted p-value.</p>',
              '<p class="small">Algorithm reference: Bubeck, Ernst and Garivier (2013), “Optimal Discovery with Probabilistic Expert Advice: Finite Time Analysis and Macroscopic Optimality”, <a href="https://www.jmlr.org/papers/v14/bubeck13a.html">Journal of Machine Learning Research</a>. Global-singleton implementation and finite-sample conclusions are documented in this package.</p></body></html>']
    (output / 'report.html').write_text('\n'.join(parts))


def main():
    verify_frozen()
    status = json.loads((ROOT / 'RUN_STATUS.json').read_text())
    assert status['complete'] and len(status['conditions']) == 8
    output = ROOT / 'analysis'
    output.mkdir(exist_ok=True)
    all_results = {}
    total_traces = 0
    for family in ['canonical', 'generated']:
        for overlap in PROTOCOL[family]['overlaps']:
            name = f'{family}_overlap{overlap:g}'
            path = ROOT / 'results' / name
            data = dict(np.load(path / 'traces.npz'))
            meta = json.loads((path / 'metadata.json').read_text())
            total_traces += audit(data, meta)
            summary, coverage = summarize(data, meta)
            all_results[name] = summary, coverage, data, meta
            print(f'Audited and analyzed {name}', flush=True)
    primary = [record(s) for s, *_ in all_results.values()]
    order = sorted(range(len(primary)), key=lambda i: primary[i]['p_unadjusted'])
    previous = 0
    for rank, i in enumerate(order):
        adjusted = min(1., max(previous, (len(primary) - rank) * primary[i]['p_unadjusted']))
        primary[i]['p_holm_8'] = adjusted
        previous = adjusted
    plots(all_results, output)
    report(all_results, output, total_traces)
    dump(output / 'summary.json', {'trajectories_audited': total_traces,
                                 'interactions': status['interactions'],
                                 'conditions': {name: s for name, (s, *_) in all_results.items()}})
    for name, (s, *_) in all_results.items():
        print(json.dumps({'condition': name, 'primary': record(s)}), flush=True)


if __name__ == '__main__':
    main()
