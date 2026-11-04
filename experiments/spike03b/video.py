"""Side-by-side clip: teacher (left) vs residual-PPO student (right), same test episode, same initial condition & sensor noise. usage: video.py COND [RLSEED]"""
import sys, os, pickle, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stu import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cond_name = sys.argv[1]; rs = int(sys.argv[2]) if len(sys.argv) > 2 else 0; cond = load_conds()[cond_name]
rl = pickle.load(open(f'results/spike03b/raw_rl_{cond_name}.pkl', 'rb')); R = rl[('rl', rs)]
actor = mk = None
from run_rl_net import make_actor
actor = make_actor(); actor.load_state_dict(R['state']); pol = ResPolicy(actor)
th, sh = rl['teacher']['hits'], R['hits']; cand = np.where((th < 40))[0]; ep = int(cand[np.argmax(sh[cand] - th[cand])]) if len(cand) else int(np.argmax(sh - th))
print('episode', ep, 'teacher hits', th[ep], 'student hits', sh[ep])
def rec(policy):
    env = E.JugEnv(1, cond); seeds = [ep]; fr = []; hist = []
    def r(env_):
        d = env_.d[0]; fr.append((d.qpos[:3].copy(), d.qpos[3:5].copy(), int(env_.hits[0])))
    E.run_policy(env, policy, E.init_cond(seeds), seeds, record=r, maxk=int(30/E.DT)); return fr
FT, FS = rec(None), rec(pol); step = 8; n = max(len(FT), len(FS)) // step
os.makedirs('.scratch/frames', exist_ok=True)
def draw(ax, fr, k, title, col):
    k = min(k, len(fr) - 1); q, b, h = fr[k]; p1 = E.L1*np.array([np.cos(q[0]), np.sin(q[0])]); p2 = p1 + E.L2*np.array([np.cos(q[0]+q[1]), np.sin(q[0]+q[1])]); ph = q.sum()
    t = np.array([np.cos(ph), np.sin(ph)])*0.12
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], '-', color='#555', lw=5, solid_capstyle='round'); ax.plot([p2[0]-t[0], p2[0]+t[0]], [p2[1]-t[1], p2[1]+t[1]], '-', color=col, lw=7)
    ax.add_patch(plt.Circle((b[0], b[1]), E.R_BALL*1.0, color='#e8590c', zorder=5)); ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.6, 1.3); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{title}   hits: {h}' + ('   (ball lost)' if k >= len(fr) - 1 and len(fr) < n*step else ''), fontsize=12)
files = []
for j in range(n):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), dpi=80); draw(axs[0], FT, j*step, 'mirror-law teacher', '#1c7ed6'); draw(axs[1], FS, j*step, 'teacher + residual RL', '#2b8a3e')
    fig.suptitle(f'Ball juggling, {cond}  |  t = {j*step*E.DT:.1f} s', fontsize=10); f = f'.scratch/frames/f{j:05d}.png'; fig.savefig(f); plt.close(fig)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '25', '-i', '.scratch/frames/f%05d.png', '-pix_fmt', 'yuv420p', f'results/spike03b/juggle_{cond_name}.mp4'], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '25', '-i', '.scratch/frames/f%05d.png', '-vf', 'fps=12,scale=640:-1', f'results/spike03b/juggle_{cond_name}.gif'], check=True)
import shutil; shutil.rmtree('.scratch/frames'); print('video done', n, 'frames')
