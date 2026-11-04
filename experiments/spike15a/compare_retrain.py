"""spike15a acceptance check of the re-trained L-dec4 (hypothesis card v8.1): holdout 1100-1599 ONLY (no TEST-C). Runs the retrained predictions through the same Lookup pipeline as spike14a
and compares with the stored spike14a per-episode successes (results/spike14a/pipe_hold.json, keys dec4_a0.9_e38_s{0,1,2}). Writes results/spike15a/retrain_check.json.
Accept: |3-seed merged success difference| <= 2 pp AND per-episode agreement of the merged (3-seed mean) success >= 90%."""
import sys, os, json, pickle, time, multiprocessing as mp
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'spike14a')); import c14; from c14 import *
if __name__ == '__main__':
    D = pickle.load(open(os.path.join(DAT, 'raw_hold.pkl'), 'rb')); assert list(D['seed']) == HOLD
    tab = {}; pr = {}
    for s in range(3):
        P = pickle.load(open(os.path.join(DAT, f'pred_r15dec4_s{s}.pkl'), 'rb'))['preds']['hold']; old = np.load(os.path.join(DAT, 'est', f'dec4_a0.9_e38_s{s}_hold.npy')); pr[s] = P
        tab[f'r{s}'] = {k: P[i] for i, k in enumerate(D['key'])}
    tp = os.path.join(ROOT, '.scratch/s15/tab_retrain_hold.pkl'); pickle.dump(tab, open(tp, 'wb')); c14._init_tab(tp); pool = mp.Pool(int(os.environ.get('NW', 8))); new = {}
    for s in range(3): t0 = time.time(); r = pool.map(c14.run14, [(f'r{s}', x) for x in HOLD], chunksize=2); new[s] = np.array([q['success'] for q in r], float); print('retrained seed', s, new[s].mean(), f'{time.time()-t0:.0f}s', flush=True)
    st = json.load(open(os.path.join(OUT14, 'pipe_hold.json'))); old = {s: np.array(st[f'dec4_a0.9_e38_s{s}']['success'], float) for s in range(3)}
    mn, mo = np.mean([new[s] for s in range(3)], 0), np.mean([old[s] for s in range(3)], 0)
    agree = float(np.mean((mn >= 0.5) == (mo >= 0.5))); agree_exact = float(np.mean(mn == mo)); diff = 100 * (mn.mean() - mo.mean())
    rms = {s: float(np.sqrt(np.mean((pr[s] - np.load(os.path.join(DAT, 'est', f'dec4_a0.9_e38_s{s}_hold.npy'))) ** 2, 0)).mean()) for s in range(3)}
    out = dict(new_seed_success=[100 * new[s].mean() for s in range(3)], old_seed_success=[100 * old[s].mean() for s in range(3)], new_merged=100 * mn.mean(), old_merged=100 * mo.mean(), diff_pp=diff,
               per_seed_episode_agreement=[float(np.mean(new[s] == old[s])) for s in range(3)], merged_agreement_majority=agree, merged_agreement_exact=agree_exact, theta_rms_diff=rms, accepted=bool(abs(diff) <= 2 and agree >= 0.9))
    json.dump(out, open(os.path.join(ROOT, 'results/spike15a/retrain_check.json'), 'w'), indent=1); print(out)
