# Discovery UCB Experiment

Python simulation comparing algorithms that choose among generators to discover distinct outcomes. Outcomes are synthetic IDs; running the experiment requires no model API, credentials, or external dataset.

## Setup

Use **Python 3.12**, the version used to verify this code. From this directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_all.py
```

This runs the tests, simulates every condition in `protocol.json`, and analyzes the generated traces. It stops if any step fails. The complete experiment runs 7,992,000 queries across 26,640 trajectories; execution time depends on your computer.

You can also run the steps separately:

```bash
python -m unittest -v test_experiment.py test_analysis.py
python experiment.py
python analyze_fixed.py
```

Analysis requires a completed experiment. Repeating the commands regenerates outputs at the same paths. Fixed random seeds make the simulated trajectories reproducible with the recorded environment.

## Algorithms and configuration

The main comparison is uniform sampling, Success-UCB, Good-UCB with coefficient 0.5, and an informed greedy comparator. The saved protocol also includes round-robin, a matched-bonus Success-UCB variant, and additional Good-UCB coefficients.

| Algorithm | Selection rule |
| --- | --- |
| Uniform | Select a generator uniformly at random. |
| Success-UCB | Use binary success rate plus an exploration bonus. |
| Good-UCB | Use global singleton count per query plus an exploration bonus. A singleton is an outcome observed exactly once across all generators. |
| Informed greedy (`oracle_greedy`) | Select the generator with the largest true probability of producing an unseen outcome. |

All UCB variants query each generator once before computing an index, avoiding division by zero. The informed comparator knows the outcome distributions but does not see future samples. The primary Good-UCB coefficient is a fixed experimental setting, not a calibrated 95% confidence bound.

`protocol.json` records seeds, query budgets, outcome pools, overlap levels, replicates, algorithm coefficients, and analysis settings. The default generators have equal success probabilities and different private pool sizes. Increasing overlap makes successful queries more likely to sample a shared pool.

## Files

| File | Purpose |
| --- | --- |
| `run_all.py` | Run tests, simulation, and corrected analysis in order. |
| `experiment.py` | Generator simulation, selection algorithms, and trace generation. |
| `analyze_fixed.py` | Corrected trace audit, statistics, and plots; use this for analysis. |
| `analyze.py` | Original frozen analysis, retained for provenance and hash verification. |
| `test_experiment.py` | Tests of sampling, selection, singleton updates, and analytic checks. |
| `test_analysis.py` | Regression tests for paired differences. |
| `protocol.json` | Saved experiment configuration. |
| `requirements.txt` | Pinned Python dependencies. |
| `FROZEN_MANIFEST.json` | Original source and protocol hashes. |
| `ANALYSIS_REPAIR.json` | Explanation and hash of the corrected analysis. |
| `.gitignore` | Exclude generated outputs and local Python files from Git. |
| `.gitattributes` | Preserve LF line endings so source hashes remain valid across platforms. |

### Why two analysis scripts?

The original `analyze.py` subtracted unsigned counts, which could turn negative paired differences into large positive numbers. `analyze_fixed.py` converts counts to floating point before subtraction and draws confidence intervals directly from their endpoints. The simulation and protocol were unchanged by this correction.

Keep both scripts. The simulation and analysis verify the original frozen files against `FROZEN_MANIFEST.json`, so deleting or editing `analyze.py` causes verification to fail. `run_all.py` executes only the corrected analysis. The same integrity checks apply to `experiment.py`, `test_experiment.py`, and `protocol.json`; this snapshot is intended to reproduce the saved experiment.

## Generated outputs

| Path | Contents |
| --- | --- |
| `results/<condition>/traces.npz` | Compressed actions, outcomes, novelty indicators, and source streams. |
| `results/<condition>/metadata.json` | Seeds and environment information. |
| `RUN_STATUS.json` | Completion status and query counts. |
| `analysis/summary.json` | Numerical summaries and uncertainty estimates. |
| `analysis/discovery_curves.png` | Discovery curves. |
| `analysis/overlap_effect.png` | Effect of shared-pool probability. |
| `analysis/report.html` | Self-contained browser view of the computed experiment results. |

Outputs are created locally when the scripts run and are excluded from Git by default.
