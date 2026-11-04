"""Quasi-static push model + sampling-based push planner for Push-T (NOMINAL knowledge only: mu, rho^2, pusher radius, goal).

Model (Lynch & Mason / Hogan & Rodriguez): the block moves quasi-statically under a point-like (circular) pusher with Coulomb friction
coefficient mu at the contact and an ELLIPSOIDAL limit surface (twist = A wrench, A = diag(1, 1, 1/rho^2) in the COM frame).
With the contact point p (body frame, relative to the COM) and r = (p_z, -p_x) (omega x p = omega r):
  contact-point velocity  v_c = (I + r r^T / rho^2) f ,  block velocity v = f ,  omega = (f . r) / rho^2     (f = contact force / b)
  sticking : f = (I + r r^T/rho^2)^-1 v_push, valid while f is inside the friction cone (f.n > 0, |f.t| <= mu f.n)
  sliding  : f = fn (n + s mu t) on the cone edge (s = sign(f.t)), fn from the normal velocity condition  v_push.n = fn (B d).n
  separating (v_push.n <= 0) or no contact: the block stays.
Everything is vectorised over a batch of candidate pushes.

Planner: a push primitive = (contact point on the outline, push direction inside the friction cone, push distance). From the current pose a
beam search of depth `depth` expands all primitives with the model, ranks the poses by cost = |dp|/0.02 + |dth|/0.15 + 0.15 * depth and
returns the first primitive of the best leaf. Approach paths (the pusher has to get around the block) are a visibility graph over the
outline offset by the pusher radius.
"""
from __future__ import annotations

import heapq

import numpy as np

from ..tasks import _pusht_geom as G

DS = 0.003              # push model step (m of pusher travel)
GAP = 0.003             # gap between the pusher and the outline at the start of a push
ALPHAS = (-0.35, 0.0, 0.35)       # push direction relative to the inward normal (rad); cone half angle atan(mu)
DISTS = (0.01, 0.025, 0.05, 0.09)
W_P, W_TH, W_PRIM = 0.02, 0.15, 0.15


def pose_cost(pose, goal):
    d = np.asarray(pose)[..., :2] - goal[:2]
    return np.linalg.norm(d, axis=-1) / W_P + np.abs(G.wrap(np.asarray(pose)[..., 2] - goal[2])) / W_TH


class PushModel:
    def __init__(self, mu, rho2, r_p):
        self.mu, self.rho2, self.r_p = float(mu), float(rho2), float(r_p)

    def step(self, pose, P, dP):
        """One quasi-static step for B candidates: pose (B,3), pusher centre P (B,2), pusher displacement dP (B,2) ->
        new pose, new pusher position (pushed out of the block if it would penetrate)."""
        pos, th = pose[:, :2], pose[:, 2]
        R = G.rot_batch(th)
        P1 = P + dP
        pl = np.einsum("bi,bij->bj", P1 - pos, R)                      # new pusher centre in the block frame
        q, d = G.closest_on_polygon(pl, G.POLY)
        contact = (d < self.r_p) & (d > 1e-9)
        if not contact.any():
            return pose, P1
        nrm = (q - pl) / np.maximum(d, 1e-12)[:, None]                  # inward normal (pusher -> block)
        vl = np.einsum("bi,bij->bj", dP, R)                             # push displacement in the block frame
        vn = (vl * nrm).sum(1)
        push = contact & (vn > 1e-9)
        rvec = np.stack([q[:, 1], -q[:, 0]], 1)
        rr = (rvec ** 2).sum(1)
        f = vl - rvec * ((rvec * vl).sum(1) / (self.rho2 + rr))[:, None]   # sticking solution (I + r r^T/rho^2)^-1 vl
        tan = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        fn, ft = (f * nrm).sum(1), (f * tan).sum(1)
        slide = push & ~((fn > 0) & (np.abs(ft) <= self.mu * fn))
        if slide.any():
            sg = np.where(ft >= 0, 1.0, -1.0)[:, None]
            dd = nrm + sg * self.mu * tan
            Bd = dd + rvec * ((rvec * dd).sum(1) / self.rho2)[:, None]
            den = np.maximum((Bd * nrm).sum(1), 0.05)
            fs = dd * (vn / den)[:, None]
            f = np.where(slide[:, None], fs, f)
        om = (f * rvec).sum(1) / self.rho2
        f = np.where(push[:, None], f, 0.0)
        om = np.where(push, om, 0.0)
        new = pose.copy()
        new[:, :2] = pos + np.einsum("bij,bj->bi", R, f)
        new[:, 2] = th + om
        # keep the pusher outside the (moved) block
        R2 = G.rot_batch(new[:, 2])
        pl2 = np.einsum("bi,bij->bj", P1 - new[:, :2], R2)
        q2, d2 = G.closest_on_polygon(pl2, G.POLY)
        pen = push & (d2 < self.r_p) & (d2 > 1e-9)
        if pen.any():
            out = (pl2 - q2) / np.maximum(d2, 1e-12)[:, None]
            pl2 = np.where(pen[:, None], q2 + out * self.r_p, pl2)
            P1 = np.where(pen[:, None], new[:, :2] + np.einsum("bij,bj->bi", R2, pl2), P1)
        return new, P1


def edge_frames():
    """For each outline edge: start, end, outward unit normal (body frame)."""
    a, b = G.POLY, np.roll(G.POLY, -1, 0)
    e = b - a
    ln = np.linalg.norm(e, axis=1)
    out = np.stack([e[:, 1], -e[:, 0]], 1) / ln[:, None]
    return a, b, out, ln


def primitives(r_p):
    """All primitives in the BLOCK frame: arrays start (M,2), dir (M,2) unit, dist (M,), nout (M,2), edge (M,)"""
    a, b, nout, ln = edge_frames()
    S, U, D, N, E = [], [], [], [], []
    for i in range(len(a)):
        fracs = (0.2, 0.5, 0.8) if ln[i] > 0.1 else (0.3, 0.7) if ln[i] > 0.06 else (0.5,)
        for fr in fracs:
            c = a[i] + fr * (b[i] - a[i])
            s0 = c + nout[i] * (r_p + GAP)
            _, d0 = G.closest_on_polygon(s0[None], G.POLY)
            if d0[0] < r_p - 1e-3:
                continue
            ok = True
            for h in (0.03, 0.06, 0.09):                                # the pusher must be able to leave along the normal
                _, dh = G.closest_on_polygon((s0 + nout[i] * h)[None], G.POLY)
                ok &= dh[0] >= r_p - 1e-3
            if not ok:
                continue
            for al in ALPHAS:
                ca, sa = np.cos(al), np.sin(al)
                u = -nout[i]
                u = np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]])
                for dist in DISTS:
                    S.append(s0); U.append(u); D.append(dist); N.append(nout[i]); E.append(i)
    return np.array(S), np.array(U), np.array(D), np.array(N), np.array(E)


class PushPlanner:
    def __init__(self, mu, rho2, r_p, goal, depth=4, beam=10):
        self.model = PushModel(mu, rho2, r_p)
        self.r_p, self.goal, self.depth, self.beam = r_p, np.asarray(goal, float), depth, beam
        self.S, self.U, self.D, self.N, self.E = primitives(r_p)
        self.nsteps = np.maximum(1, np.round(self.D / DS).astype(int))

    def _expand(self, poses):
        """Apply every primitive to every pose: returns new poses (K, M, 3)."""
        K, M = len(poses), len(self.D)
        pose = np.repeat(poses, M, 0)                                   # (K*M, 3)
        R = G.rot_batch(pose[:, 2])
        P = pose[:, :2] + np.einsum("bij,bj->bi", R, np.tile(self.S, (K, 1)))
        dirw = np.einsum("bij,bj->bi", R, np.tile(self.U, (K, 1)))
        nst = np.tile(self.nsteps, K)
        order = np.argsort(-nst, kind="stable")                         # longest pushes first: later steps touch a shrinking prefix
        cur, P, dirw, nst = pose[order], P[order], dirw[order], nst[order]
        for s in range(int(nst[0])):
            cnt = int((nst > s).sum())
            c, p = self.model.step(cur[:cnt], P[:cnt], dirw[:cnt] * DS)
            cur[:cnt], P[:cnt] = c, p
        out = np.empty_like(cur)
        out[order] = cur
        cur = out
        return cur.reshape(K, M, 3)

    def plan(self, pose):
        """Beam search. Returns list of (first primitive index, best total cost reachable through it), best first, plus the best cost."""
        pose = np.asarray(pose, float)
        M = len(self.D)
        best = {}                                                       # first primitive -> best cost
        cur_poses, cur_root, cur_cost = pose[None], np.full(1, -1), np.zeros(1)
        for lvl in range(1, self.depth + 1):
            new = self._expand(cur_poses)                               # (K, M, 3)
            K = len(cur_poses)
            cost = pose_cost(new, self.goal) + W_PRIM * lvl             # (K, M)
            roots = np.where(cur_root[:, None] < 0, np.arange(M)[None], cur_root[:, None]).repeat(1, 1)
            roots = np.broadcast_to(roots, (K, M))
            cflat, rflat = cost.ravel(), roots.ravel()
            for r in np.unique(rflat):                                  # best cost per first primitive (any depth)
                v = cflat[rflat == r].min()
                if v < best.get(int(r), np.inf):
                    best[int(r)] = float(v)
            keep = np.argsort(cflat)[: self.beam]
            cur_poses, cur_root = new.reshape(-1, 3)[keep], rflat[keep]
        return sorted(best.items(), key=lambda kv: kv[1])

    # ---- geometry of one chosen primitive in the world ------------------------------------------------------
    def world_primitive(self, pose, i):
        R = G.rot(pose[2])
        s = np.asarray(pose[:2]) + R @ self.S[i]
        u = R @ self.U[i]
        return s, u, float(self.D[i]), R @ self.N[i]


def offset_polygon(pose, dist):
    """World-frame outline offset outward by `dist` (vertex bisector offset)."""
    poly = G.poly_world(pose)
    n = len(poly)
    out = []
    for i in range(n):
        e0 = poly[i] - poly[i - 1]
        e1 = poly[(i + 1) % n] - poly[i]
        n0 = np.array([e0[1], -e0[0]]) / np.linalg.norm(e0)
        n1 = np.array([e1[1], -e1[0]]) / np.linalg.norm(e1)
        bis = n0 + n1
        bis /= np.linalg.norm(bis)
        out.append(poly[i] + bis * dist / max(float(bis @ n0), 0.3))
    return np.array(out)


def _clear(a, b, poly, clearance, skip_end=0.0):
    ln = np.linalg.norm(b - a)
    m = max(2, int(ln / 0.005) + 1)
    pts = a[None] + np.linspace(0, 1, m)[:, None] * (b - a)[None]
    if skip_end > 0 and ln > 0:
        pts = pts[np.linspace(0, 1, m) * ln <= ln - skip_end]
        if len(pts) == 0:
            return True
    _, d = G.closest_on_polygon(pts, poly)
    return bool((d >= clearance).all())


def approach_path(tip, goal_pt, gate_pt, pose, r_p, margin=0.012):
    """Waypoints from `tip` to `goal_pt` that keep the pusher centre >= r_p + 0.7 margin away from the block; the last leg gate -> goal
    is the straight approach along the outward normal. None if there is no path."""
    poly = G.poly_world(pose)
    clr = r_p + 0.7 * margin
    nodes = [np.asarray(tip, float)]
    _, d_tip = G.closest_on_polygon(nodes[0][None], poly)
    if d_tip[0] < clr:                                                  # start (nearly) touching the block: back off by the shortest free move
        found = None
        for rad in np.arange(0.01, 0.12, 0.005):
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                c = nodes[0] + rad * np.array([np.cos(ang), np.sin(ang)])
                if _clear(c, c, poly, clr) and _clear(nodes[0], c, poly, r_p - 0.002):
                    found = c
                    break
            if found is not None:
                break
        if found is None:
            return None
        nodes.append(found)
    nodes += list(offset_polygon(pose, r_p + margin))
    nodes.append(np.asarray(gate_pt, float))
    iG = len(nodes) - 1
    first = 0 if len(nodes) and d_tip[0] >= clr else 1
    # Dijkstra from `first` (the tip, or the backed-off point reached by a straight retreat)
    dist = {first: float(np.linalg.norm(nodes[first] - nodes[0])) if first else 0.0}
    prev = {}
    pq = [(dist[first], first)]
    done = set()
    while pq:
        dcur, i = heapq.heappop(pq)
        if i in done:
            continue
        done.add(i)
        if i == iG:
            break
        for j in range(len(nodes)):
            if j == i or j in done or j == 0:
                continue
            if not _clear(nodes[i], nodes[j], poly, clr, skip_end=0.0 if j != iG else 0.0):
                continue
            nd = dcur + float(np.linalg.norm(nodes[j] - nodes[i]))
            if nd < dist.get(j, np.inf):
                dist[j], prev[j] = nd, i
                heapq.heappush(pq, (nd, j))
    if iG not in done:
        return None
    path, i = [nodes[iG]], iG
    while i in prev:
        i = prev[i]
        path.append(nodes[i])
    if first == 1:
        path.append(nodes[0])
    path = path[::-1]
    path.append(np.asarray(goal_pt, float))
    return path
