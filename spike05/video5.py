"""Side-by-side clip, same test episode: T-adapt (left) vs R-DR-hist seed 0 (right). usage: video5.py T-adapt 0
Episode selection rule (fixed before looking at episodes, symmetric): among test episodes where seed-median R-DR-hist hits differ from the classical hits, take the one whose
difference is closest to the MEDIAN of those differences (a typical episode, not the extreme best/worst case for either side)."""
import sys, os, pickle, json, subprocess, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V
from eval_all import net, Res
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pickle.load(open('results/spike05/test_raw.pkl', 'rb')); cl = sys.argv[1]; hs = sys.argv[2] if len(sys.argv) > 2 else '0'
Hc = R[cl]['hits']; Hr = np.array([R[k]['hits'] for k in R if k.startswith('R-hist#')]); Hrm = np.median(Hr, 0); diff = Hrm - Hc; nz = np.where(diff != 0)[0]
ep = int(nz[np.argmin(np.abs(diff[nz] - np.median(diff[nz])))]) if len(nz) else 0
print('episode', ep, 'classical', Hc[ep], 'R-hist median over seeds', Hrm[ep], 'params', R[cl]['params'][ep])
J = pickle.load(open('results/spike05/raw_rl_hist.pkl', 'rb'))[int(hs)]; actor = net(J['obs'], J['hid'], 3); actor.load_state_dict(J['state'])
P = V.P_NOM if cl == 'T-nom' else json.load(open(f'results/spike05/{cl[2:]}.json'))['params']; MAXK = int(40 / V.DT)
def rec(P_, pol, hist):
    env = V.JugEnv5(1, P_, hist=hist); fr = []
    def r(env_): d = env_.d[0]; fr.append((d.qpos[:3].copy(), d.qpos[3:5].copy(), int(env_.hits[0])))
    V.run_policy(env, pol, V.init_cond([ep]), [ep], record=r, maxk=MAXK); return fr
FT, FS = rec(P, None, False), rec(V.P_NOM, Res(actor), True); step = 10; n = max(len(FT), len(FS)) // step; pp = R[cl]['params'][ep]
os.makedirs('.scratch/frames5', exist_ok=True)
def draw(ax, fr, k, title, col):
    k = min(k, len(fr) - 1); q, b, h = fr[k]; p1 = V.L1 * np.array([np.cos(q[0]), np.sin(q[0])]); p2 = p1 + V.L2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])]); ph = q.sum(); t = np.array([np.cos(ph), np.sin(ph)]) * 0.12
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], '-', color='#555', lw=5, solid_capstyle='round'); ax.plot([p2[0] - t[0], p2[0] + t[0]], [p2[1] - t[1], p2[1] + t[1]], '-', color=col, lw=7)
    ax.add_patch(plt.Circle((b[0], b[1]), V.E.R_BALL, color='#e8590c', zorder=5)); ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.6, 1.3); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{title}   hits: {h}' + ('  (ball lost)' if k >= len(fr) - 1 and len(fr) < MAXK else ''), fontsize=11)
for j in range(n):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), dpi=80); draw(axs[0], FT, j * step, cl, '#1c7ed6'); draw(axs[1], FS, j * step, 'R-DR-hist (residual RL)', '#2b8a3e')
    fig.suptitle(f'test ep {ep}: e={pp["e"]:.2f}, ball x{pp["mb"]:.1f}, delay {pp["delay_ms"]:.0f} ms | t={j*step*V.DT:.1f}s', fontsize=10); fig.savefig(f'.scratch/frames5/f{j:05d}.png'); plt.close(fig)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '25', '-i', '.scratch/frames5/f%05d.png', '-pix_fmt', 'yuv420p', 'results/spike05/juggle_h8.mp4'], check=True)
import shutil; shutil.rmtree('.scratch/frames5'); print('video', n, 'frames')
