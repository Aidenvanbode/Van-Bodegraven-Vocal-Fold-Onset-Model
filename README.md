# Phonation onset in a noisy two-mass vocal fold model: easy vs. hard onset

computational engineering project by Aiden Van Bodegraven (2026).

The question: in a simple mechanical model of the vocal folds, does starting voicing from a firmly closed ("hard") glottis, instead of a slightly open ("easy") one, change whether oscillation starts, the subglottal pressure it needs, or how fast it starts?

This is engineering and nonlinear dynamics only. The model is a lumped mechanical oscillator.
## What the model is

- Symmetric Ishizaka-Flanagan (1972) two-mass model of the vocal folds, in the simplified form used by Steinecke & Herzel (1995) and Lucero & Koenig (2005).
- Stage A: the original IF72 flow forces (Bernoulli flow, free jet after the narrowest point).
- Stage B (main model from Step 4c on): Stage A mechanics plus the viscous pressure losses of Assaneo & Trevisan (2013), as interpreted in this project (sign of P_in, flow equation, missing d1*lg factor, area floor; see step4c/model_c.py).
- Noise: a 0.5 % x Ps random pressure force on the lower mass, with fixed seeds (2 kHz Gaussian samples, linearly interpolated).
- Adductory hold (Step 4e on): a spring of 2000 N/m on each mass pulling toward -0.05 mm (past the midline), switched off by a raised-cosine release over 5, 20 or 100 ms. Easy and hard runs differ only by this hold.
- Onset detector (Step 4e on): cycle-by-cycle R = (peak - trough) / mean flow; onset when R >= 0.10 and the oscillation is not decaying. Step 4f added an amplitude-trend test as a second way to pass the "not decaying" check, and a 0.02 mm displacement criterion for zero rest gap (where R is undefined).

## Main results (short version; the full numbers are in the step reports)

- Easy and hard onset thresholds are equal within about 2 Pa (Stage B about 200 Pa, Stage A about 270 Pa). This holds for hold stiffness 500-5000 N/m, hold position -0.02 to -0.1 mm and R cut-off 0.05-0.20.
- Onset timing is a race: a hard start begins 0-5 ms after the release ends, at full size; an easy start has to grow from noise (Stage B median 62 ms at 1.05 x threshold, 12 ms at 1.2 x). A fast release wins near threshold; easy wins at high pressure or with a 100 ms release. The core pattern held in 76 of 100 wide random parameter draws; where it switches depends on the R cut-off and the hold stiffness.
- H2 (easy and hard need different pressures) rejected; H3 (easy more robust) not supported; H4 (delay ~ (Ps - Pth)^-1) partial (slope -1.6 Stage B, -0.8 Stage A); H1 (no-onset region for pressed, stiff folds) not found.
- Validation: V3 (laryngeal-size scaling, Lucero & Koenig 2005 eq. 5) passes. V7 (Q-Ps boundary, Assaneo & Trevisan 2013): shape passes, values about 2.4 x too high at Q about 1.7.
- Negative results are reported as they came out. The earlier "hard starts at lower pressure" result of Steps 4b-4d was an artifact (blow-open of the "pressed" folds and 10 ms detection windows) and was withdrawn in Step 4e.

## Install

Python 3.13 was used. Free tools only.

    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Repository layout

| Folder | Contents |
|---|---|
| step3_starter/ | vf_starter.py: first sketch of the model, linear Hopf threshold check (Lucero & Koenig 2005) |
| step4/ | Stage A model (model.py), first experiments: hysteresis, threshold variants, convergence, onset map, paths, robustness. Superseded, kept because reports cite it |
| step4b/ | 0.5 % pressure noise (common.py), fair easy/hard comparison, H4 |
| step4c/ | Stage B (model_c.py), wide-gap onset map with AC/DC rule, contact sensitivity, Stage B paths |
| step4d/ | Tests of the H2 artifact: kick runs, release time, AC/DC cut-off table, smoothing width |
| step4e/ | Adductory hold and cycle-by-cycle detector (model_e.py); thresholds, delays, paths, H4, onset maps |
| step4f/ | Detector fix and displacement criterion (model_f.py); sensitivity to hold and R cut-off; wide random draws; V3 and V7 |
| step5_figs/ | Figure scripts for the reference drafts (5c figures and the 5d Figure 1) |
| tools/runner.py | Shared helper used by the run scripts |

Each step folder keeps its own scripts and its stored result tables (CSV, JSON, analysis_output.txt). Later steps import earlier models by relative path (for example step4e imports ../step4c/model_c.py), so keep the folders side by side and run scripts from inside their folder (the run scripts do this for you). Large tables are stored gzip-compressed (*.gz); the run scripts decompress them on first use.

## How to run

There is one run script per step. Each has three modes:

    python run_step4e.py --quick      # smoke test: a few short simulations + the analysis, in a temporary copy (< 2 min)
    python run_step4e.py --analysis   # re-make that step's tables and figures from the stored result files (seconds to a minute)
    python run_step4e.py --full       # re-run every simulation of that step, then the analysis (hours; overwrites stored results)

Available: run_step3.py, run_step4.py, run_step4b.py, run_step4c.py, run_step4d.py, run_step4e.py, run_step4f.py, run_step5_figs.py.

Approximate full-run times on 8 cores, from the original run logs: Step 4 about 10 min, 4b about 16 min, 4c about 1.5 h, 4d about 40 min, 4e about 1.5 h, 4f several hours (the jobs were spread over a first full-scale pass and a reduced second pass, so there is no single clean timing). All random numbers come from numpy.random.default_rng with fixed seeds (parameter draw i uses seed 1000 + i, wide draws 9000 + i, noise seeds 7, 3000 + i, 5000 + i as written in each script), so a full re-run gives the same numbers on the same library versions.

Note on Step 4f: it was run at reduced scale (see below). run4f.py is resumable: it skips jobs already in results.jsonl. To recompute from scratch, delete step4f/results.jsonl first. The job specs that were completed in the first full-scale pass before the scale was reduced (620 thresholds, 160 old-detector thresholds, 49 Stage B zero-gap cells) are stored in step4f/first_pass_jobs.json and are included automatically, so a fresh run reproduces every 4f result.

Reproducibility check: running every analysis script on the stored result files reproduced each step's stored analysis_output.txt exactly.

## Which script makes which figure and table

| Report | Figure / table | Script (inside its step folder) | Input |
|---|---|---|---|
| Step 3 | linear threshold, starter checks (text output) | vf_starter.py | none |
| Step 4 | fig0_hysteresis, fig1_onset_map, fig2_onset_times, fig3_robustness, fig4_h4, analysis_output.txt | analysis.py (H4 runs inside it) | exp1_onset_map.csv, exp2_paths.csv, exp3_robustness.csv, part1_hyst_steps.csv |
| Step 4 | part1_*.csv, part2_*.csv, exp1/2/3 csv | part1.py, part1_hyst.py, part2_validation.py, exp1.py, exp23.py 2, exp23.py 3 100 | simulation |
| Step 4b | fig0_hysteresis_noise, fig1_onset_map, fig2_paths, fig4_h4, exp23_summary.csv, exp23_tests.csv | analysis.py | p1_*.csv, exp1_onset_map.csv, exp23_paths.csv, h4_const.csv |
| Step 4b | p1_*.csv, exp1_onset_map.csv, exp23_paths.csv, h4_const.csv | part1.py, exps.py 200 | simulation |
| Step 4c | fig1_onset_map_4c, fig2_wide_sweeps, fig3_gap_curve, fig4_h2_robustness, onset_map_4c.csv, *_summary.csv, *_tests.csv | analysis_c.py | wide_sweep_*.csv, stageB_*.csv, sens_*.csv |
| Step 4c | stageB_gap_threshold*.csv, sens_*.csv, wide_sweep*.csv, stageB_paths*.csv, stageB_hyst.csv | run_c.py, run_full.py | simulation |
| Step 4d | fig2_kick, fig3_cutoff_paths, fig4_release_eps, h23_by_cutoff.csv, kick_summary.csv, ... | analysis4d.py, map_cutoffs.py | path_runs.csv, kick_runs.csv, ../step4c/wide_sweep_*.csv |
| Step 4d | diag_hard_trace.png | diag.py | simulation |
| Step 4e | Fig. 1 traces | trace.py | simulation (3 runs) |
| Step 4e | Tables 3-8, fig2_thresholds, fig3_delays, fig4_paths, fig5_h4, fig6_maps, summary_*.csv | analysis4e.py | thresholds.csv, delays.csv, paths.csv, h4.csv, map.csv, nominal_sweep.csv |
| Step 4f | all tables, fig_v3_v7, fig_sensitivity_race, fig_robust, fig_nonmono, *_4f.csv, V3.csv, V7.csv, robust_*.csv, sensitivity_summary.csv | analysis4f.py | results.jsonl, ../step4e/*.csv |
| Draft 5c | fig1-fig6 of the 5c draft | step5_figs/make_figs.py | step4b, step4c tables |
| Draft 5d | Fig. 1 (schematic with hold and release) | step5_figs/make_figs_5d.py | none |

The 5d draft's other figures are the Step 4e and 4f figures listed above, plus the 5c gap curve.

## Corrections made while preparing this repository

- step4e/analysis4e.py: in summary_thresholds.csv the columns hard_lower and hard_higher were swapped, and in summary_paths.csv the count of draws where the easy path started at lower pressure was labelled hard_lower_Ps. The labels are fixed here (easy_lower_Ps); no numbers changed. Correct Stage B threshold counts (hard lower by > 2 Pa / within 2 Pa / hard higher by > 2 Pa): 6/13/1, 7/13/0, 3/15/2 for 5/20/100 ms releases. The Step 4e report printed them the wrong way round (1/13/6, ...). The conclusion (equal thresholds within about one 2 Pa bisection step) is unchanged.
- step4f/run4f.py: added the 829 first-pass job specs (first_pass_jobs.json) so that a fresh run reproduces every 4f result.
- step5_figs/make_figs.py: paths changed from the author's machine to this repository layout.

## Note on statistics

Steps 4b-4d computed Wilcoxon / Holm p-values (still visible in those analysis scripts and their outputs). From Step 4e on these were dropped: the +-10 % parameter draws are variation inside a deterministic model, not independent experiments, so results are reported as counts, medians, IQRs and ranges (Wilson intervals only as descriptive ranges for success fractions).

## Known limitations

- Single symmetric two-mass model, quasi-steady flow, no vocal tract.
- The Stage B equations are an interpretation of a paper whose formulas are incomplete.
- The adductory hold is an artificial spring; its stiffness and target position are choices.
- Every timing result depends on the onset definition (the R cut-off).
- V7 values are about 2.4 x too high compared with Assaneo & Trevisan (2013) at Q about 1.7.

## References

- Ishizaka, K. & Flanagan, J. L. (1972). Synthesis of voiced sounds from a two-mass model of the vocal cords. Bell Syst. Tech. J. 51, 1233-1268.
- Steinecke, I. & Herzel, H. (1995). Bifurcations in an asymmetric vocal-fold model. J. Acoust. Soc. Am. 97, 1874-1884.
- Lucero, J. C. & Koenig, L. L. (2005). Phonation thresholds as a function of laryngeal size in a two-mass model of the vocal folds (L). J. Acoust. Soc. Am. 118, 2798-2801.
- Assaneo, M. F. & Trevisan, M. A. (2013). Revisiting the two-mass model of the vocal folds. Papers in Physics 5, 050004.

## License

MIT, see LICENSE. Copyright (c) 2026 Aiden Van Bodegraven.
