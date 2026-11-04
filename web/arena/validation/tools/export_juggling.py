"""Export the racket-juggling task (spike03b mirror-law teacher) for the web arena.
Writes ../../assets/juggling{,_nominal}.xml (+ juggling.json constants) and ../python_juggling.json (+ python_juggling_noise.bin):
  * closed-loop trajectory of CAL seed 1000, nominal plant (first TRAJ_STEPS control steps, with the exact sensor-noise samples used),
  * hit counts / failure mode of 20 CAL seeds (1000-1019) for nominal (e=0.8) and restitution 0.4, untuned teacher,
  * the initial conditions and the sensor noise of every seed, so the JS run can be driven by identical inputs.
Only calibration seeds are used. experiments/spike03b/ is imported read-only. Run from anywhere with the repo env."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "spike03b"))
import numpy as np
import env as E

ASSETS = os.path.abspath(os.path.join(HERE, "..", "..", "assets"))
OUT = os.path.abspath(os.path.join(HERE, ".."))
SEEDS = list(range(1000, 1020))
assert all(1000 <= s <= 1019 for s in SEEDS)
TRAJ_STEPS = 1500

# MJCF: same template as E.build, with the mass scale and the contact damping ratio left as @placeholders@ for the browser
ball = E.BALL.format(r=E.R_BALL, mb=E.M_BALL, tc=E.TC, zeta="@ZETA@")
tmpl = E.XML.format(dt=E.DT_SIM, d1=E.DAMP[0], d2=E.DAMP[1], d3=E.DAMP[2], m1="@M1@", m2="@M2@", m3="@M3@", ht=E.HALF_T, tc=E.TC, zeta="@ZETA@", ball=ball, tm=E.TAU_MAX)
open(os.path.join(ASSETS, "juggling.xml"), "w").write(tmpl)
nom = E.XML.format(dt=E.DT_SIM, d1=E.DAMP[0], d2=E.DAMP[1], d3=E.DAMP[2], m1=E.M1, m2=E.M2, m3=E.M3, ht=E.HALF_T, tc=E.TC, zeta=E.zeta_for(E.E_NOM), ball="", tm=E.TAU_MAX)
open(os.path.join(ASSETS, "juggling_nominal.xml"), "w").write(nom)
zm = json.load(open(os.path.join(ROOT, "experiments", "spike03b", "zeta_map.json")))
cfg = dict(dt=E.DT, substeps=E.SUB, gravity=E.G, L=[E.L1, E.L2], r_ball=E.R_BALL, half_t=E.HALF_T, ball_mass=E.M_BALL, x_star=E.X_STAR, h_star=E.H_STAR,
           wrist_z0=E.WRIST_Z0, tau_max=E.TAU_MAX, hits_cap=E.HITS_CAP, nmax=E.NMAX, kappa_max=E.KAPPA_MAX, e_nom=E.E_NOM, sigma=0.0002, masses=[E.M1, E.M2, E.M3],
           tp=E.TP, init=dict(dx=0.08, h=[0.2, 0.5], vx=0.3), zeta_map=zm, q0=E.Q0.tolist())
json.dump(cfg, open(os.path.join(ASSETS, "juggling.json"), "w"), indent=1)

CASES = {"nominal": E.Cond(), "e0.4": E.Cond(e=0.4, name="e=0.4")}
ics = E.init_cond(SEEDS)
out = dict(seeds=SEEDS, init=ics, cases={}, traj=None, noise_len={}, noise_offset={})
noise_store = {}
for name, cond in CASES.items():
    env = E.JugEnv(len(SEEDS), cond)
    Q = []; TAU = []
    def rec(e):
        if name == "nominal" and len(Q) < TRAJ_STEPS:
            Q.append(np.concatenate([e.d[0].qpos[:5], e.d[0].qvel[:5]]).tolist())
    # run_policy with the pure teacher, recording seed 1000 state before every step and its torque
    env.reset(ics, SEEDS)
    for i, s in enumerate(SEEDS): noise_store[s] = env.noise[i].copy()
    steps = np.zeros(len(SEEDS), int)
    while env.active.any():
        env.measure_update(); tT = env.teacher()
        if name == "nominal" and len(Q) < TRAJ_STEPS:
            Q.append(np.concatenate([env.d[0].qpos[:5], env.d[0].qvel[:5]]).tolist()); TAU.append(tT[0].tolist())
        env.step(tT)
    rows = [dict(seed=s, hits=int(env.hits[i]), fail=env.fail[i], steps=int(env.k[i])) for i, s in enumerate(SEEDS)]
    out["cases"][name] = rows
    hs = [r["hits"] for r in rows]
    print(name, "hits", hs, "median", np.median(hs), "mean", np.mean(hs), flush=True)
    if name == "nominal": out["traj"] = dict(q_qd=Q, tau=TAU, seed=1000, steps=len(TAU))
off = 0
with open(os.path.join(OUT, "python_juggling_noise.bin"), "wb") as f:
    for s in SEEDS:
        n = max(r["steps"] for c in out["cases"].values() for r in c if r["seed"] == s) + 5
        arr = noise_store[s][:n].astype("<f8"); f.write(arr.tobytes())
        out["noise_len"][str(s)] = int(n); out["noise_offset"][str(s)] = off; off += int(n)
json.dump(out, open(os.path.join(OUT, "python_juggling.json"), "w"), separators=(",", ":"))
print("noise samples", off)
