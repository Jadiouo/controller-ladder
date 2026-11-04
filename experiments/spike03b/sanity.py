"""Contact sanity: ball dropped on a FIXED racket; rebound velocity ratio vs set restitution. Builds zeta_map.json (e -> solref damping ratio)."""
import numpy as np, mujoco, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import env as E
def drop(zeta, h):
    m = E.build(zeta); d = mujoco.MjData(m)
    q0 = E.Q0.copy(); d.qpos[:3] = q0; mujoco.mj_forward(m, d)
    pw = d.site_xpos[0].copy(); d.qpos[3] = pw[0]; d.qpos[4] = pw[2] + E.OFF + h
    vin = None; vout = None; g0 = pw[2] + E.OFF
    for _ in range(int(1.5/E.DT_SIM)):
        d.qpos[:3] = q0; d.qvel[:3] = 0          # racket held fixed
        gap = d.qpos[4] - g0; v0 = d.qvel[4]
        if vin is None and gap < 0.003: vin = v0     # velocity just before/at first touch (free fall so far)
        mujoco.mj_step(m, d)
        if vin is not None and d.qvel[4] > 0 and d.qpos[4] - g0 > 0.002: vout = d.qvel[4]; break
    return -vin, vout
if __name__ == '__main__':
    zs = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.33]; es = []
    for z in zs:
        r = [drop(z, h) for h in (0.15, 0.5, 1.0)]; ee = [b/a for a, b in r]; es.append(float(np.mean(ee[:2])))
        print(f'zeta {z:.2f}: e(v ratio) at drop 0.15/0.5/1.0 m = {np.round(ee,3)}')
    if '--write' in sys.argv: json.dump(dict(zeta=zs, e=es), open(os.path.join(os.path.dirname(__file__), 'zeta_map.json'), 'w'))
    print('map saved')
    for e_set in (0.8, 0.7, 0.6, 0.5, 0.4):
        z = E.zeta_for(e_set); r = [drop(z, h) for h in (0.1, 0.25, 0.5, 1.0)]
        print(f'set e={e_set} -> zeta {z:.3f}; measured velocity ratio at drops 0.1/0.25/0.5/1.0 m:', np.round([b/a for a, b in r], 3), ' height ratio:', np.round([(b/a)**2 for a, b in r], 3))
