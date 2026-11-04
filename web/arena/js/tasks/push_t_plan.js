// Push-T geometry, quasi-static push model and sampling-based push planner (pure JS, no MuJoCo).
// Line-by-line port of arena/tasks/_pusht_geom.py and arena/controllers/_pusht_plan.py (nominal knowledge only: mu, rho^2, pusher radius, goal).
// Vectors are plain [x, z] arrays; poses are [x, z, th]; the rotation of a pose is R(th) = [[c, s], [-s, c]] (a MuJoCo hinge about +y).

// ---- geometry --------------------------------------------------------------------------------------------------------
export const W = 0.20, B = 0.05, S = 0.05, L = 0.15;                 // bar width / thickness, stem width / length (m)
const A1 = W * B, A2 = S * L;
export const V_BAR = (A2 * (B / 2 + L / 2)) / (A1 + A2);
export const V_STEM = V_BAR - (B / 2 + L / 2);
export const AREA = A1 + A2;
export const RECTS = [[0.0, V_BAR, W / 2, B / 2], [0.0, V_STEM, S / 2, L / 2]];     // (centre u, centre v, half u, half v)
export const POLY = [[-W / 2, V_BAR + B / 2], [-W / 2, V_BAR - B / 2], [-S / 2, V_BAR - B / 2], [-S / 2, V_STEM - L / 2],
  [S / 2, V_STEM - L / 2], [S / 2, V_BAR - B / 2], [W / 2, V_BAR - B / 2], [W / 2, V_BAR + B / 2]];
export const RHO2 = (A1 * ((W ** 2 + B ** 2) / 12 + V_BAR ** 2) + A2 * ((S ** 2 + L ** 2) / 12 + V_STEM ** 2)) / AREA;

export const wrap = (a) => { const t = a + Math.PI, m = t - 2 * Math.PI * Math.floor(t / (2 * Math.PI)); return m - Math.PI; };
/** R(th) p */
export const rotv = (th, p) => { const c = Math.cos(th), s = Math.sin(th); return [c * p[0] + s * p[1], -s * p[0] + c * p[1]]; };
/** R(th)^T v (world -> shape frame) */
export const rotT = (th, v) => { const c = Math.cos(th), s = Math.sin(th); return [c * v[0] - s * v[1], s * v[0] + c * v[1]]; };

function rectCorners(x, z, th, rect) {
  const [cu, cv, hu, hv] = rect;
  return [[cu - hu, cv - hv], [cu + hu, cv - hv], [cu + hu, cv + hv], [cu - hu, cv + hv]].map((p) => { const r = rotv(th, p); return [x + r[0], z + r[1]]; });
}
function clipPoly(subject, a, b) {    // Sutherland-Hodgman, keep the left of a->b
  const out = [], ex = b[0] - a[0], ez = b[1] - a[1];
  const side = (p) => ex * (p[1] - a[1]) - ez * (p[0] - a[0]);
  for (let i = 0; i < subject.length; i++) {
    const p = subject[i], q = subject[(i + 1) % subject.length], sp = side(p), sq = side(q);
    if (sp >= 0) out.push(p);
    if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
  }
  return out;
}
function polyArea(pts) {
  if (pts.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < pts.length; i++) { const j = (i + 1) % pts.length; s += pts[i][0] * pts[j][1] - pts[i][1] * pts[j][0]; }
  return 0.5 * Math.abs(s);
}
function convexOverlap(c1, c2) {
  let poly = c1.slice();
  for (let i = 0; i < c2.length; i++) { if (!poly.length) return 0; poly = clipPoly(poly, c2[i], c2[(i + 1) % c2.length]); }
  return polyArea(poly);
}
export function overlapArea(pa, pb) {
  const ca = RECTS.map((r) => rectCorners(pa[0], pa[1], pa[2], r)), cb = RECTS.map((r) => rectCorners(pb[0], pb[1], pb[2], r));
  let s = 0;
  for (const a of ca) for (const b of cb) s += convexOverlap(a, b);
  return s;
}
/** Diffusion Policy Push-T score: overlap area / area of the goal T */
export const coverage = (pose, goal) => overlapArea(pose, goal) / AREA;
export function polyWorld(pose) {
  return POLY.map((p) => { const r = rotv(pose[2], p); return [pose[0] + r[0], pose[1] + r[1]]; });
}
/** closest point of the polygon `poly` to point `pt`: {q, d} */
export function closestOnPolygon(pt, poly) {
  let bi = -1, bd = Infinity, ox = 0, oz = 0;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n], abx = b[0] - a[0], abz = b[1] - a[1], inv = 1 / (abx * abx + abz * abz);
    const dx = pt[0] - a[0], dz = pt[1] - a[1];
    let t = (dx * abx + dz * abz) * inv; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const fx = dx - t * abx, fz = dz - t * abz, d2 = fx * fx + fz * fz;
    if (d2 < bd) { bd = d2; bi = i; ox = fx; oz = fz; }
  }
  return { q: [pt[0] - ox, pt[1] - oz], d: Math.sqrt(bd) };
}
export function insidePolygon(pt, poly) {
  const [x, y] = pt, n = poly.length;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const [xa, ya] = poly[i], [xb, yb] = poly[(i + 1) % n];
    if ((ya > y) !== (yb > y) && x < xa + ((y - ya) * (xb - xa)) / (yb - ya)) c++;
  }
  return c % 2 === 1;
}

// ---- quasi-static push model ----------------------------------------------------------------------------------------
export const DS = 0.003, GAP = 0.003, ALPHAS = [-0.35, 0.0, 0.35], DISTS = [0.01, 0.025, 0.05, 0.09];
export const W_P = 0.02, W_TH = 0.15, W_PRIM = 0.15;

export const poseCost = (pose, goal) => Math.hypot(pose[0] - goal[0], pose[1] - goal[1]) / W_P + Math.abs(wrap(pose[2] - goal[2])) / W_TH;

export class PushModel {
  constructor(mu, rho2, rp) { this.mu = mu; this.rho2 = rho2; this.rp = rp; }
  /** one quasi-static step; pose [x,z,th] and pusher centre P [x,z] are updated IN PLACE; dP = pusher displacement */
  step(pose, P, dPx, dPz) {
    const { mu, rho2, rp } = this;
    P[0] += dPx; P[1] += dPz;
    const th = pose[2];
    const pl = rotT(th, [P[0] - pose[0], P[1] - pose[1]]);
    const { q, d } = closestOnPolygon(pl, POLY);
    if (!(d < rp && d > 1e-9)) return;                              // no contact
    const dd0 = Math.max(d, 1e-12), nx = (q[0] - pl[0]) / dd0, nz = (q[1] - pl[1]) / dd0;
    const vl = rotT(th, [dPx, dPz]);
    const vn = vl[0] * nx + vl[1] * nz;
    if (!(vn > 1e-9)) return;                                       // separating
    const rx = q[1], rz = -q[0], rr = rx * rx + rz * rz;
    const k0 = (rx * vl[0] + rz * vl[1]) / (rho2 + rr);
    let fx = vl[0] - rx * k0, fz = vl[1] - rz * k0;                // sticking solution
    const tx = -nz, tz = nx;
    const fn = fx * nx + fz * nz, ft = fx * tx + fz * tz;
    if (!(fn > 0 && Math.abs(ft) <= mu * fn)) {                     // sliding on the cone edge
      const sg = ft >= 0 ? 1 : -1;
      const ddx = nx + sg * mu * tx, ddz = nz + sg * mu * tz;
      const k1 = (rx * ddx + rz * ddz) / rho2;
      const Bdx = ddx + rx * k1, Bdz = ddz + rz * k1;
      const den = Math.max(Bdx * nx + Bdz * nz, 0.05);
      fx = ddx * (vn / den); fz = ddz * (vn / den);
    }
    const om = (fx * rx + fz * rz) / rho2;
    const w = rotv(th, [fx, fz]);
    pose[0] += w[0]; pose[1] += w[1]; pose[2] = th + om;
    // keep the pusher outside the moved block
    const pl2 = rotT(pose[2], [P[0] - pose[0], P[1] - pose[1]]);
    const c2 = closestOnPolygon(pl2, POLY);
    if (c2.d < rp && c2.d > 1e-9) {
      const dd = Math.max(c2.d, 1e-12);
      const ox = c2.q[0] + ((pl2[0] - c2.q[0]) / dd) * rp, oz = c2.q[1] + ((pl2[1] - c2.q[1]) / dd) * rp;
      const wp = rotv(pose[2], [ox, oz]);
      P[0] = pose[0] + wp[0]; P[1] = pose[1] + wp[1];
    }
  }
}

function edgeFrames() {
  const n = POLY.length, a = POLY, b = POLY.map((_, i) => POLY[(i + 1) % n]);
  const ln = a.map((p, i) => Math.hypot(b[i][0] - p[0], b[i][1] - p[1]));
  const nout = a.map((p, i) => [(b[i][1] - p[1]) / ln[i], -(b[i][0] - p[0]) / ln[i]]);
  return { a, b, nout, ln };
}

/** all primitives in the BLOCK frame: {S, U, D, N, E} arrays */
export function primitives(rp) {
  const { a, b, nout, ln } = edgeFrames();
  const out = { S: [], U: [], D: [], N: [], E: [] };
  for (let i = 0; i < a.length; i++) {
    const fracs = ln[i] > 0.1 ? [0.2, 0.5, 0.8] : ln[i] > 0.06 ? [0.3, 0.7] : [0.5];
    for (const fr of fracs) {
      const c = [a[i][0] + fr * (b[i][0] - a[i][0]), a[i][1] + fr * (b[i][1] - a[i][1])];
      const s0 = [c[0] + nout[i][0] * (rp + GAP), c[1] + nout[i][1] * (rp + GAP)];
      if (closestOnPolygon(s0, POLY).d < rp - 1e-3) continue;
      let ok = true;
      for (const h of [0.03, 0.06, 0.09]) ok = ok && closestOnPolygon([s0[0] + nout[i][0] * h, s0[1] + nout[i][1] * h], POLY).d >= rp - 1e-3;
      if (!ok) continue;
      for (const al of ALPHAS) {
        const ca = Math.cos(al), sa = Math.sin(al), u0 = -nout[i][0], u1 = -nout[i][1];
        const u = [ca * u0 - sa * u1, sa * u0 + ca * u1];
        for (const dist of DISTS) { out.S.push(s0); out.U.push(u); out.D.push(dist); out.N.push(nout[i]); out.E.push(i); }
      }
    }
  }
  return out;
}

export class PushPlanner {
  constructor(mu, rho2, rp, goal, depth = 4, beam = 10) {
    this.model = new PushModel(mu, rho2, rp);
    Object.assign(this, { rp, goal, depth, beam });
    Object.assign(this, (() => { const p = primitives(rp); return { S: p.S, U: p.U, D: p.D, N: p.N, E: p.E }; })());
    this.nsteps = this.D.map((d) => Math.max(1, Math.round(d / DS)));
    this.M = this.D.length;
  }
  /** apply every primitive to every pose: returns K*M poses (row k*M + m) */
  _expand(poses) {
    const M = this.M, out = new Array(poses.length * M);
    const P = [0, 0];
    for (let k = 0; k < poses.length; k++) {
      for (let m = 0; m < M; m++) {
        const pose = poses[k].slice();
        const sp = rotv(pose[2], this.S[m]), dir = rotv(pose[2], this.U[m]);
        P[0] = pose[0] + sp[0]; P[1] = pose[1] + sp[1];
        const dx = dir[0] * DS, dz = dir[1] * DS;
        for (let s = 0, n = this.nsteps[m]; s < n; s++) this.model.step(pose, P, dx, dz);
        out[k * M + m] = pose;
      }
    }
    return out;
  }
  /** beam search; returns [[first primitive index, best total cost reachable through it], ...] best first */
  plan(pose) {
    const M = this.M, best = new Array(M).fill(Infinity);
    let cur = [pose.slice()], root = [-1];
    for (let lvl = 1; lvl <= this.depth; lvl++) {
      const nw = this._expand(cur), cost = nw.map((p) => poseCost(p, this.goal) + W_PRIM * lvl);
      const roots = nw.map((_, i) => (root[Math.floor(i / M)] < 0 ? i % M : root[Math.floor(i / M)]));
      for (let i = 0; i < cost.length; i++) if (cost[i] < best[roots[i]]) best[roots[i]] = cost[i];
      const idx = cost.map((_, i) => i).sort((u, v) => cost[u] - cost[v] || u - v).slice(0, this.beam);
      cur = idx.map((i) => nw[i]); root = idx.map((i) => roots[i]);
    }
    return best.map((c, i) => [i, c]).filter((x) => Number.isFinite(x[1])).sort((u, v) => u[1] - v[1] || u[0] - v[0]);
  }
  worldPrimitive(pose, i) {
    const s = rotv(pose[2], this.S[i]), u = rotv(pose[2], this.U[i]), n = rotv(pose[2], this.N[i]);
    return { s: [pose[0] + s[0], pose[1] + s[1]], u, dist: this.D[i], nout: n };
  }
}

// ---- approach paths ---------------------------------------------------------------------------------------------------
export function offsetPolygon(pose, dist) {
  const poly = polyWorld(pose), n = poly.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = poly[(i + n - 1) % n], p1 = poly[i], p2 = poly[(i + 1) % n];
    const e0 = [p1[0] - p0[0], p1[1] - p0[1]], e1 = [p2[0] - p1[0], p2[1] - p1[1]], l0 = Math.hypot(...e0), l1 = Math.hypot(...e1);
    const n0 = [e0[1] / l0, -e0[0] / l0], n1 = [e1[1] / l1, -e1[0] / l1];
    let bx = n0[0] + n1[0], bz = n0[1] + n1[1];
    const bl = Math.hypot(bx, bz); bx /= bl; bz /= bl;
    const f = dist / Math.max(bx * n0[0] + bz * n0[1], 0.3);
    out.push([p1[0] + bx * f, p1[1] + bz * f]);
  }
  return out;
}
function clear(a, b, poly, clearance) {
  const ln = Math.hypot(b[0] - a[0], b[1] - a[1]), m = Math.max(2, Math.trunc(ln / 0.005) + 1);
  for (let i = 0; i < m; i++) {
    const t = i / (m - 1);
    if (closestOnPolygon([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], poly).d < clearance) return false;
  }
  return true;
}
/** waypoints from `tip` to `goalPt` keeping the pusher centre >= rp + 0.7 margin from the block; last leg gate -> goal. null if none. */
export function approachPath(tip, goalPt, gatePt, pose, rp, margin = 0.012) {
  const poly = polyWorld(pose), clr = rp + 0.7 * margin;
  const nodes = [tip.slice()];
  const dTip = closestOnPolygon(nodes[0], poly).d;
  if (dTip < clr) {                                    // start (nearly) touching: back off by the shortest free move
    let found = null;
    for (let ri = 0; ri < 22 && !found; ri++) {      // np.arange(0.01, 0.12, 0.005)
      const rad = 0.01 + 0.005 * ri;
      for (let a = 0; a < 24; a++) {
        const ang = (2 * Math.PI * a) / 24, c = [nodes[0][0] + rad * Math.cos(ang), nodes[0][1] + rad * Math.sin(ang)];
        if (clear(c, c, poly, clr) && clear(nodes[0], c, poly, rp - 0.002)) { found = c; break; }
      }
    }
    if (!found) return null;
    nodes.push(found);
  }
  for (const v of offsetPolygon(pose, rp + margin)) nodes.push(v);
  nodes.push(gatePt.slice());
  const iG = nodes.length - 1, first = dTip >= clr ? 0 : 1;
  const dist = new Map([[first, first ? Math.hypot(nodes[first][0] - nodes[0][0], nodes[first][1] - nodes[0][1]) : 0]]);
  const prev = new Map(), done = new Set(), pq = [[dist.get(first), first]];
  while (pq.length) {
    let bi = 0;
    for (let j = 1; j < pq.length; j++) if (pq[j][0] < pq[bi][0] || (pq[j][0] === pq[bi][0] && pq[j][1] < pq[bi][1])) bi = j;
    const [dcur, i] = pq.splice(bi, 1)[0];
    if (done.has(i)) continue;
    done.add(i);
    if (i === iG) break;
    for (let j = 1; j < nodes.length; j++) {
      if (j === i || done.has(j)) continue;
      if (!clear(nodes[i], nodes[j], poly, clr)) continue;
      const nd = dcur + Math.hypot(nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1]);
      if (nd < (dist.get(j) ?? Infinity)) { dist.set(j, nd); prev.set(j, i); pq.push([nd, j]); }
    }
  }
  if (!done.has(iG)) return null;
  const path = [nodes[iG]];
  let i = iG;
  while (prev.has(i)) { i = prev.get(i); path.push(nodes[i]); }
  if (first === 1) path.push(nodes[0]);
  path.reverse();
  path.push(goalPt.slice());
  return path;
}
