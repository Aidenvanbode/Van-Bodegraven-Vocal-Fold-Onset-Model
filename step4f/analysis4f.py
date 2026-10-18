import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
def proportion_confint(k, n, method='wilson', alpha=0.05):
    z = norm.ppf(1 - alpha / 2); p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
E = '../step4e/'; OUT = open('analysis_output.txt', 'w')
def P(*a):
    print(*a); print(*a, file=OUT)
def wil(k, n):
    if n == 0: return f'{k}/{n}'
    lo, hi = proportion_confint(k, n, method='wilson'); return f'{k}/{n} [{lo:.2f},{hi:.2f}]'
def desc(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0: return 'n=0'
    q = np.percentile(x, [25, 50, 75]); return f'median {q[1]:.1f} [IQR {q[0]:.1f}, {q[2]:.1f}; range {x.min():.1f} to {x.max():.1f}] (n={len(x)})'
R = [json.loads(l) for l in open('results.jsonl')]
errs = [r for r in R if r['err']]; P('jobs', len(R), 'errors', len(errs))
def tab(job, cols):
    rows = []
    for r in R:
        if r['job'] != job: continue
        a = list(r['args']); d = dict(zip(cols, a))
        hold = a[len(cols)] if len(a) > len(cols) else [2000.0, -0.05e-3]
        d['K_ADD'] = hold[0]; d['X_AD_mm'] = round(hold[1] * 1e3, 3)
        if job == 'thr': d['rstar'] = a[5] if len(a) > 5 else 0.1
        d.update({k: v for k, v in r.items() if k not in ('job', 'args', 'key', 'err')}); rows.append(d)
    return pd.DataFrame(rows)
BH = lambda d: d[(d.K_ADD == 2000) & (d.X_AD_mm == -0.05)]
dl = tab('delay', ['stage', 'start', 'Trel', 'mult', 'draw']); pa = tab('path', ['stage', 'start', 'Tr', 'Trel', 'mult', 'draw'])
th = tab('thr', ['stage', 'start', 'Trel', 'draw']); tv1 = tab('thr_v1', ['stage', 'start', 'Trel', 'draw'])
for d in (dl, pa, th, tv1): d['Trel_ms'] = (d.Trel * 1e3).round().astype(int)
pa['Tr_ms'] = (pa.Tr * 1e3).round().astype(int)
for d in (dl, pa, th, tv1): d['cond'] = np.where(d.start == 'easy', 'easy', 'hard rel ' + d.Trel_ms.astype(str) + ' ms')
dl.to_csv('delays_4f.csv', index=False); pa.to_csv('paths_4f.csv', index=False); th.to_csv('thresholds_4f.csv', index=False)
# ============ 1. detector fix ============
P('\n# 1. DETECTOR FIX (baseline hold, R*=0.10)')
e_dl = pd.read_csv(E + 'delays.csv'); e_pa = pd.read_csv(E + 'paths.csv'); e_th = pd.read_csv(E + 'thresholds.csv')
for name, new, old, ks in (('delays', BH(dl), e_dl, ['stage', 'start', 'Trel_ms', 'mult', 'draw']), ('paths', BH(pa), e_pa, ['stage', 'start', 'Tr_ms', 'Trel_ms', 'mult', 'draw'])):
    old = old.copy(); old['Trel_ms'] = old.Trel_ms.round().astype(int)
    if 'Tr_ms' in old: old['Tr_ms'] = old.Tr_ms.round().astype(int)
    m = new.merge(old[ks + ['success', 't_on']], on=ks, how='inner')
    rep = (m['succ_v1_R0.1'] == m.success).sum(); ton = np.nanmax(np.abs(m['t_on_R0.1'] - m.t_on).where(m.success & m['succ_v1_R0.1']))
    resc = m[m['succ_R0.1'] & ~m['succ_v1_R0.1']]; fl = m[~m['succ_R0.1']]
    P(f'{name}: {len(m)} runs; 4e success reproduced by v1 in {rep}/{len(m)} (max |t_on diff| {ton:.2e} s)')
    P(f'  v1 fails {(~m["succ_v1_R0.1"]).sum()}, of which end-level>=0.10 (decay-test fails) {((~m["succ_v1_R0.1"]) & (m.endR >= 0.1)).sum()}; rescued by trend test {len(resc)}; still failing with level>=0.10: {((fl.endR >= 0.1)).sum()}')
    P('  rescued by condition:', resc.groupby(['stage', 'cond']).size().to_dict())
    if len(resc): P('  rescued: v1 decay slope ratio info, trend slope per s', desc(resc.slopeR))
    fl2 = fl[fl.endR >= 0.1]
    if len(fl2): P('  still-fail with level>=0.1 (true decay by trend):', fl2[ks + ['endR', 'slopeR']].to_string())
# thresholds
tb = BH(th[th.rstar == 0.1]); tt = tb.merge(tv1[['stage', 'cond', 'draw', 'Ps_th_v1']], on=['stage', 'cond', 'draw'])
e_th['Trel_ms'] = e_th.Trel_ms.round().astype(int); e_th['cond'] = np.where(e_th.start == 'easy', 'easy', 'hard rel ' + e_th.Trel_ms.astype(str) + ' ms')
tt = tt.merge(e_th[['stage', 'cond', 'draw', 'Ps_th']].rename(columns={'Ps_th': 'Ps_th_4e'}), on=['stage', 'cond', 'draw'])
P(f'thresholds: v1 re-run equals 4e in {(np.isclose(tt.Ps_th_v1, tt.Ps_th_4e, equal_nan=True)).sum()}/{len(tt)}; v2 differs from v1 in {(~np.isclose(tt.Ps_th, tt.Ps_th_v1, equal_nan=True)).sum()}')
if (~np.isclose(tt.Ps_th, tt.Ps_th_v1, equal_nan=True)).any(): P(tt[~np.isclose(tt.Ps_th, tt.Ps_th_v1, equal_nan=True)][['stage', 'cond', 'draw', 'Ps_th_v1', 'Ps_th']].to_string())
h4 = tab('h4', ['stage', 'mult', 'seed']); P(f"h4: v1 fails {(~h4['succ_v1_R0.1']).sum()}, v2 fails {(~h4['succ_R0.1']).sum()} of {len(h4)}")
# ============ summaries with v2 ============
def thr_summary(t, stage, label=''):
    t = t[t.stage == stage]; pv = t.pivot_table(index='draw', columns='cond', values='Ps_th'); rows = []
    for c in [c for c in pv.columns if c != 'easy']:
        if 'easy' not in pv: break
        dd = (pv[c] - pv['easy']).dropna()
        rows.append(dict(stage=stage, setting=label, cond=c, hard=desc(pv[c]), easy=desc(pv['easy']), hard_minus_easy=desc(dd),
                         hard_lower_gt2=int((dd < -2).sum()), within2=int((dd.abs() <= 2).sum()), hard_higher_gt2=int((dd > 2).sum()), n=len(dd)))
    return rows
def race(d, stage, rstar=0.1, easy=None):
    """winner per (mult, release) among draws where both succeed: hard sooner if easy delay - hard delay > 0"""
    d = d[d.stage == stage]; e = (easy if easy is not None else d)[lambda x: (x.start == 'easy') & (x.stage == stage)]; rows = []
    k, dk = f'succ_R{rstar:g}', f'delay_ms_R{rstar:g}'
    for (m, rel), h in d[d.start == 'hard'].groupby(['mult', 'Trel_ms']):
        mm = h.merge(e[e.mult == m][['draw', k, dk]], on='draw', suffixes=('_h', '_e')); mm = mm[mm[k + '_h'] & mm[k + '_e']]
        diff = mm[dk + '_e'] - mm[dk + '_h']
        rows.append(dict(mult=m, Trel_ms=rel, hard_sooner=int((diff > 0).sum()), easy_sooner=int((diff < 0).sum()), n=len(mm),
                         succ_hard=wil(int(h[k].sum()), len(h)), median_easy_minus_hard_ms=float(np.median(diff)) if len(diff) else np.nan))
    return pd.DataFrame(rows)
def winner(r): return 'hard' if r.hard_sooner > r.easy_sooner else ('easy' if r.easy_sooner > r.hard_sooner else 'tie')
P('\n# Step 4e results recomputed with fixed detector (v2) vs v1')
for v, sk in (('v1', 'succ_v1_R0.1'), ('v2', 'succ_R0.1')):
    d = BH(dl).copy(); d['succ_R0.1'] = d[sk]; d.loc[~d[sk], 'delay_ms_R0.1'] = np.nan
    d = d[(d.draw < 20) & d.mult.isin([1.05, 1.2, 1.5])]
    for st in 'B':
        r = race(d, st); r['w'] = r.apply(winner, axis=1) if len(r) else []
        P(f'race {v} stage {st}:'); P(r.to_string(index=False))
        P('success counts:', d[d.stage == st].groupby(['mult', 'cond'])[sk].sum().unstack().to_string())
for st in 'BA':
    P(f'thresholds stage {st} (v2):'); P(pd.DataFrame(thr_summary(tb, st)).to_string(index=False))
def h3(d, sk='succ_R0.1'):
    g = d.groupby(['stage', 'Tr_ms', 'mult', 'cond'])[sk].agg(['sum', 'count']).reset_index(); g['succ'] = [wil(a, b) for a, b in zip(g['sum'], g['count'])]; return g
PB = BH(pa)[(BH(pa).stage == 'B') & (BH(pa).Tr_ms == 200) & (BH(pa).mult == 1.2) & (BH(pa).draw < 40)]
for v, sk in (('v1', 'succ_v1_R0.1'), ('v2', 'succ_R0.1')):
    P(f'H3 path success {v} (Stage B, Tr 200 ms, 1.2x, draws 0-39):'); P(h3(PB, sk).pivot_table(index=['stage', 'Tr_ms', 'mult'], columns='cond', values='sum').to_string())
# H2 on paths (v2): easy - hard Ps_on paired
def h2(d, rstar=0.1):
    k, pk = f'succ_R{rstar:g}', f'Ps_on_R{rstar:g}'; rows = []
    for (st, Tr, m), g in d.groupby(['stage', 'Tr_ms', 'mult']):
        e = g[g.start == 'easy'].set_index('draw')
        for c, h in g[g.start == 'hard'].groupby('cond'):
            h = h.set_index('draw'); ok = e[k] & h[k].reindex(e.index).fillna(False).astype(bool); dd = (e[pk] - h[pk].reindex(e.index))[ok]
            rows.append(dict(stage=st, Tr_ms=Tr, mult=m, cond=c, n=int(ok.sum()), easy_lower=int((dd < -1e-6).sum()), equal=int((dd.abs() <= 1e-6).sum()), easy_minus_hard=desc(dd)))
    return pd.DataFrame(rows)
P('H2 (v2) paired onset pressure easy - hard:'); P(h2(PB).to_string(index=False))
# ============ 2. sensitivity ============
P('\n# 2. SENSITIVITY')
SETS = [(2000.0, -0.05), (500.0, -0.05), (1000.0, -0.05), (5000.0, -0.05), (2000.0, -0.02), (2000.0, -0.1)]
easyB_th = th[(th.stage == 'B') & (th.start == 'easy') & (th.rstar == 0.1)]
easyB_dl = BH(dl)[(BH(dl).start == 'easy') & (BH(dl).stage == 'B') & (BH(dl).draw < 20)]; easyB_pa = BH(pa)[(BH(pa).start == 'easy') & (BH(pa).stage == 'B') & (BH(pa).Tr_ms == 200) & (BH(pa).draw < 40)]
sens = []; base_w = None
for K, X in SETS:
    lab = f'K={K:g} X={X:g}mm R*=0.10'
    t = th[(th.K_ADD == K) & (th.X_AD_mm == X) & (th.stage == 'B') & (th.rstar == 0.1)]; t = pd.concat([t[t.start == 'hard'], easyB_th])
    ts = thr_summary(t, 'B', lab)
    d = dl[(dl.K_ADD == K) & (dl.X_AD_mm == X) & (dl.stage == 'B') & (dl.start == 'hard') & dl.mult.isin([1.05, 1.2, 1.5]) & (dl.draw < 20)]
    if K == 1000: d = d.iloc[0:0]
    r = race(d, 'B', easy=easyB_dl); r['w'] = r.apply(winner, axis=1) if len(r) else []
    p_ = pa[(pa.K_ADD == K) & (pa.X_AD_mm == X) & (pa.stage == 'B') & (pa.start == 'hard') & (pa.Tr_ms == 200) & (pa.mult == 1.2) & (pa.draw < 40)]
    hs = {f'{m:g}x {c}': int(g['succ_R0.1'].sum()) for (m, c), g in p_.groupby(['mult', 'cond'])}
    hs.update({f'{m:g}x easy': int(g['succ_R0.1'].sum()) for m, g in easyB_pa[easyB_pa.mult == 1.2].groupby('mult')})
    blow = d.max_gap_while_held_mm.max()
    sens.append(dict(setting=lab, thr=ts, race=r, h3=hs, max_gap_held_mm=blow))
for rs in (0.05, 0.2):
    lab = f'K=2000 X=-0.05mm R*={rs:.2f}'
    t = th[(th.rstar == rs) & (th.stage == 'B')]; ts = thr_summary(t, 'B', lab)
    d = BH(dl)[(BH(dl).stage == 'B') & BH(dl).mult.isin([1.05, 1.2, 1.5]) & (BH(dl).draw < 20)]
    r = race(d, 'B', rstar=rs); r['w'] = r.apply(winner, axis=1) if len(r) else []
    p_ = PB
    hs = {f'{m:g}x {c}': int(g[f'succ_R{rs:g}'].sum()) for (m, c), g in p_.groupby(['mult', 'cond'])}
    sens.append(dict(setting=lab, thr=ts, race=r, h3=hs, max_gap_held_mm=np.nan))
base_race = sens[0]['race'].set_index(['mult', 'Trel_ms'])['w']
srows = []
for s in sens:
    r = s['race'].set_index(['mult', 'Trel_ms']) if len(s['race']) else None
    if r is None:
        row = dict(setting=s['setting'], race_pattern='not run (reduced scale)', race_same_as_base=np.nan, race_cells=0)
    else: row = dict(setting=s['setting'], race_pattern=' '.join(f"{m:g}x/{rel}:{r.loc[(m, rel), 'w'][0].upper()}({r.loc[(m, rel), 'hard_sooner']}-{r.loc[(m, rel), 'easy_sooner']})" for m, rel in r.index),
               race_same_as_base=int(sum(r.loc[i, 'w'] == base_race[i] for i in r.index)), race_cells=len(r))

    for x in s['thr']: row[f"thr {x['cond']}: hard-easy median"] = x['hard_minus_easy'].split(' [')[0].replace('median ', ''); row[f"thr {x['cond']}: lower/within/higher"] = f"{x['hard_lower_gt2']}/{x['within2']}/{x['hard_higher_gt2']}"
    row['H3 success (Tr 200 ms)'] = ', '.join(f'{k}:{v}' for k, v in sorted(s['h3'].items())); row['max_gap_held_mm'] = s['max_gap_held_mm']; srows.append(row)
S = pd.DataFrame(srows); S.to_csv('sensitivity_summary.csv', index=False)
for _, r in S.iterrows(): P('\n' + '\n'.join(f'  {k}: {v}' for k, v in r.items()))
for s in sens: P('\n' + s['setting']); P(s['race'].to_string(index=False) if len(s['race']) else 'race not run'); P(pd.DataFrame(s['thr'])[['cond', 'hard', 'hard_minus_easy', 'hard_lower_gt2', 'within2', 'hard_higher_gt2']].to_string(index=False))
# ============ zero gap / map ============
P('\n# MAPS (displacement criterion, rows x0 = 0, 0.05, 0.2 mm)')
ce = tab('cell', ['stage', 'x0_mm', 'Q', 'crit']); ce = ce[ce.crit == 'X']; em = pd.read_csv(E + 'map.csv')
for st in 'BA':
    m = ce[ce.stage == st].merge(em[em.stage == st].rename(columns={'Ps_th': 'Ps_th_4e_flowR'}), on=['x0_mm', 'Q'], how='left')
    P(f'stage {st}:'); P(m.pivot_table(index='x0_mm', columns='Q', values=['Ps_th', 'Ps_th_4e_flowR'], dropna=False).round(0).to_string())
    nz = m[m.x0_mm != 0]; P(f'  |X - flowR(4e)| on x0>0 cells: {desc((nz.Ps_th - nz.Ps_th_4e_flowR).abs())}')
ce.to_csv('map_4f.csv', index=False)
# ============ non-monotone ============
P('\n# NON-MONOTONE STAGE A HARD SWEEPS')
nm = tab('nonmono', ['stage', 'Trel', 'Ps', 'seed', 'Tafter']); nm['Trel_ms'] = (nm.Trel * 1e3).round().astype(int); nm.to_csv('nonmono_4f.csv', index=False)
for (rel, sd, Ta), g in nm.groupby(['Trel_ms', 'seed', 'Tafter']):
    g = g.sort_values('Ps'); s = g['succ_R0.1'].astype(int).values; flips = int((np.diff(s) < 0).sum())
    first = g.Ps[g['succ_R0.1']].min(); lastoff = g.Ps[~g['succ_R0.1']].max()
    P(f'rel {rel} seed {sd} T_after {Ta}: first on {first}, last off {lastoff}, on->off flips {flips}; ' + ''.join('#' if v else '.' for v in s))
nm_e = pd.read_csv(E + 'nominal_sweep.csv'); P('4e nominal sweep stage A hard near 250-300:'); P(nm_e[(nm_e.stage == 'A') & (nm_e.Ps.between(250, 300))].pivot_table(index='Ps', columns='Trel_ms', values='success').to_string())
# ============ V3 / V7 ============
P('\n# V3'); v3 = tab('V3', ['stage', 'beta']); v3['onset_minus_offset'] = v3.P_on - v3.P_off
v3['ratio_on_vs_beta1'] = v3.P_on / v3.groupby('stage').P_on.transform(lambda s: s[v3.loc[s.index, 'beta'] == 1.0].iloc[0]); v3['1/beta'] = 1 / v3.beta
P(v3[['stage', 'beta', 'Gamma_th', 'P_lin', 'eq5', 'P_on', 'P_off', 'onset_minus_offset', 'ratio_on_vs_beta1', '1/beta', 'high_ok']].round(3).to_string(index=False)); v3.to_csv('V3.csv', index=False)
P('\n# V7'); v7 = tab('V7', ['stage', 'Q']); v7['onset_minus_offset'] = v7.P_on - v7.P_off; P(v7[['stage', 'Q', 'P_lin', 'P_on', 'P_off', 'onset_minus_offset', 'high_ok']].round(2).to_string(index=False)); v7.to_csv('V7.csv', index=False)
nv = tab('normal', ['stage', 'Q', 'Ps']); P(nv.to_string(index=False))
# ============ robustness ============
P('\n# ROBUSTNESS (wide draws)')
rb = tab('robust', ['draw']); rb.to_csv('robust_4f.csv', index=False)
rel = (5, 20, 100); rows = []
for _, r in rb.iterrows():
    if pd.isna(r.Ps_th): continue
    d = dict(draw=r.draw, Ps_th=r.Ps_th, F0=r.F0)
    for m in (1.05, 1.2, 1.5):
        e = r[f'delay_{m:g}_easy0'] if r[f'succ_{m:g}_easy0'] else np.nan
        for t_ in rel:
            h = r[f'delay_{m:g}_hard{t_}'] if r[f'succ_{m:g}_hard{t_}'] else np.nan
            d[f'{m:g}/{t_}'] = 'n/a' if (pd.isna(e) or pd.isna(h)) else ('H' if e - h > 0 else 'E')
    rows.append(d)
RB = pd.DataFrame(rows); base_pat = {f'{m:g}/{t_}': base_race[(m, t_)][0].upper() for m in (1.05, 1.2, 1.5) for t_ in rel}
cells = list(base_pat); RB['n_match'] = [sum(RB.loc[i, c] == base_pat[c] for c in cells) for i in RB.index]; RB['n_valid'] = [sum(RB.loc[i, c] != 'n/a' for c in cells) for i in RB.index]
core = lambda r: r['1.05/5'] == 'H' and r['1.5/20'] == 'E' and r['1.5/100'] == 'E' and r['1.2/100'] == 'E'
RB['core'] = RB.apply(core, axis=1); RB.to_csv('robust_patterns.csv', index=False)
P(f'draws {len(rb)}, with threshold <=1500 Pa: {len(RB)}; threshold {desc(RB.Ps_th)}; F0 {desc(RB.F0)}')
P('baseline (4e-style, nominal-family) pattern:', base_pat)
for c in cells: P(f'  cell {c}: ' + ', '.join(f'{k}={v}' for k, v in RB[c].value_counts().items()) + f' (base {base_pat[c]})')
full = (RB.n_match == 9).sum(); P(f'all 9 cells match: {wil(int(full), len(RB))}; all valid cells match: {wil(int((RB.n_match == RB.n_valid).sum()), len(RB))}; core pattern: {wil(int(RB.core.sum()), len(RB))}')
P('n_match distribution', RB.n_match.value_counts().sort_index().to_dict())
# ============ figures ============
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for st, mk in (('B', 'o'), ('A', 's')):
    v = v3[v3.stage == st].sort_values('beta'); ax[0].plot(v.beta, v.P_on, mk + '-', label=f'Stage {st} onset'); ax[0].plot(v.beta, v.P_off, mk + '--', mfc='none', label=f'Stage {st} offset')
v = v3[v3.stage == 'B'].sort_values('beta'); ax[0].plot(v.beta, v.P_lin, 'k-', lw=2, label='eq. (5) linear (LK 2005)')
ax[0].set_xlabel('size factor β'); ax[0].set_ylabel('threshold Ps (Pa)'); ax[0].set_title('V3: threshold vs laryngeal size'); ax[0].legend(fontsize=7); ax[0].invert_xaxis()
for st, mk in (('B', 'o'), ('A', 's')):
    v = v7[v7.stage == st].sort_values('Q'); ax[1].plot(v.Q, v.P_on, mk + '-', label=f'Stage {st} onset'); ax[1].plot(v.Q, v.P_off, mk + '--', mfc='none', label=f'Stage {st} offset')
v = v7[v7.stage == 'B'].sort_values('Q'); ax[1].plot(v.Q, v.P_lin, 'k-', lw=1, label='linear (no viscosity)')
ax[1].fill_between([0.2, 1.8], [30, 130], [60, 140], color='orange', alpha=0.3, label='A&T 2013 Fig. 2 onset band (read by eye)')
ax[1].set_xlabel('Q'); ax[1].set_ylabel('Ps (Pa)'); ax[1].set_title('V7: onset boundary in (Q, Ps)'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig_v3_v7.png', dpi=130)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, m in zip(ax, (1.05, 1.2, 1.5)):
    labs, vals = [], []
    for s in [s for s in sens if len(s['race'])]:
        r = s['race'].set_index(['mult', 'Trel_ms'])
        for t_ in rel:
            x = r.loc[(m, t_)]; vals.append(x.hard_sooner / max(x.n, 1)); labs.append((s['setting'].replace(' R*=', '\nR*=').replace('K=', 'K').replace(' X=', ' X'), t_))
    SS = [s for s in sens if len(s['race'])]; n = len(SS); arr = np.array(vals).reshape(n, 3); im = a.imshow(arr, vmin=0, vmax=1, cmap='coolwarm_r', aspect='auto')
    a.set_xticks(range(3)); a.set_xticklabels([f'{t} ms' for t in rel]); a.set_yticks(range(n)); a.set_yticklabels([s['setting'] for s in SS], fontsize=6)
    for i in range(n):
        for j in range(3): a.text(j, i, f'{arr[i, j]:.2f}', ha='center', va='center', fontsize=7)
    a.set_title(f'{m}x threshold: fraction hard sooner')
plt.tight_layout(); plt.savefig('fig_sensitivity_race.png', dpi=130)
fig, ax = plt.subplots(figsize=(7, 3.5)); cc = [c for c in cells]
fr = [(RB[c] == 'H').sum() / max((RB[c] != 'n/a').sum(), 1) for c in cc]
ax.bar(range(9), fr, color=['tab:red' if base_pat[c] == 'H' else 'tab:blue' for c in cc]); ax.set_xticks(range(9)); ax.set_xticklabels(cc, rotation=45, fontsize=8)
ax.set_ylabel('fraction of draws: hard sooner'); ax.set_title('Race across wide parameter draws (red = baseline hard wins)'); ax.axhline(0.5, color='k', lw=0.5)
plt.tight_layout(); plt.savefig('fig_robust.png', dpi=130)
fig, ax = plt.subplots(figsize=(8, 3.2))
for i, ((relm, sd, Ta), g) in enumerate(nm.groupby(['Trel_ms', 'seed', 'Tafter'])):
    g = g.sort_values('Ps'); ax.scatter(g.Ps, [i] * len(g), c=np.where(g['succ_R0.1'], 'tab:green', 'tab:gray'), s=12)
ax.set_yticks(range(len(nm.groupby(['Trel_ms', 'seed', 'Tafter'])))); ax.set_yticklabels([f'rel {a} seed {b} T{c}' for a, b, c in nm.groupby(['Trel_ms', 'seed', 'Tafter']).groups], fontsize=6)
ax.set_xlabel('Ps (Pa)'); ax.set_title('Stage A, hard start: sustained oscillation (green) vs not (grey)'); plt.tight_layout(); plt.savefig('fig_nonmono.png', dpi=130)
OUT.close()
