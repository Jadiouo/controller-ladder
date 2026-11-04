// Shared helpers of the two ball tasks (catch, toss): drag ballistics (arena/tasks/_ballistics.py), quintic splines,
// scipy-compatible brentq, level IK, and small dense linear algebra. Pure JS (runs in node and the browser).
export const G = 9.81;
const GV = [0, -G];

/** Python round(): half to even */
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// ---- ballistics: a = -k v + g -----------------------------------------------------------------------------------
export const hOf = (t, k) => (k < 1e-9 ? t : (1 - Math.exp(-k * t)) / k);
export function position(t, p0, v0, k, g = G) {
  const gv = [0, -g], h = hOf(t, k);
  if (k < 1e-9) return [p0[0] + v0[0] * t + 0.5 * gv[0] * t * t, p0[1] + v0[1] * t + 0.5 * gv[1] * t * t];
  return [0, 1].map((i) => p0[i] + (gv[i] * t) / k + (v0[i] - gv[i] / k) * h);
}
export function velocity(t, v0, k, g = G) {
  const gv = [0, -g];
  if (k < 1e-9) return [v0[0] + gv[0] * t, v0[1] + gv[1] * t];
  const e = Math.exp(-k * t);
  return [0, 1].map((i) => gv[i] / k + (v0[i] - gv[i] / k) * e);
}
export function launchVelocity(p0, target, T, k, g = G) {
  const gv = [0, -g];
  if (k < 1e-9) return [0, 1].map((i) => (target[i] - p0[i] - 0.5 * gv[i] * T * T) / T);
  return [0, 1].map((i) => gv[i] / k + (target[i] - p0[i] - (gv[i] * T) / k) / hOf(T, k));
}
/** least-squares (p0 at t = 0, v0) of the drag model from samples (centred simple regression = numpy lstsq for 2 unknowns) */
export function fit(ts, ps, k, g = G) {
  const n = ts.length, gv = [0, -g];
  const h = ts.map((t) => (k < 1e-9 ? t : hOf(t, k)));
  const hm = h.reduce((a, b) => a + b, 0) / n;
  let shh = 0;
  for (const x of h) shh += (x - hm) * (x - hm);
  const p0 = [0, 0], v0 = [0, 0];
  for (let c = 0; c < 2; c++) {
    const y = ts.map((t, i) => ps[i][c] - (k >= 1e-9 ? (t * gv[c]) / k : 0.5 * gv[c] * t * t));
    const ym = y.reduce((a, b) => a + b, 0) / n;
    let shy = 0;
    for (let i = 0; i < n; i++) shy += (h[i] - hm) * (y[i] - ym);
    const w = shy / shh;
    p0[c] = ym - w * hm;
    v0[c] = k < 1e-9 ? w : w + gv[c] / k;
  }
  return [p0, v0];
}

// ---- planar 3-link arm (arena/robots/kinematics.py) ------------------------------------------------------------------
export function fk3(L, q) {
  let p = 0, x = 0, z = 0;
  for (let i = 0; i < L.length; i++) { p += q[i]; x += L[i] * Math.cos(p); z -= L[i] * Math.sin(p); }
  return [x, z];
}
/** (2 x n) tip Jacobian */
export function jac3(L, q) {
  const n = L.length, phi = [];
  let p = 0;
  for (let i = 0; i < n; i++) { p += q[i]; phi.push(p); }
  const dx = L.map((l, i) => -l * Math.sin(phi[i])), dz = L.map((l, i) => -l * Math.cos(phi[i]));
  const jx = new Array(n), jz = new Array(n);
  let sx = 0, sz = 0;
  for (let i = n - 1; i >= 0; i--) { sx += dx[i]; sz += dz[i]; jx[i] = sx; jz[i] = sz; }
  return [jx, jz];
}
/** analytic IK, tip xz and total angle phi; returns null if unreachable */
export function levelIk(L, tip, phi = 0, elbowUp = true) {
  const [L1, L2, L3] = L;
  const wx = tip[0] - L3 * Math.cos(phi), wz = tip[1] + L3 * Math.sin(phi);
  const D = (wx * wx + wz * wz - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  if (Math.abs(D) > 1) return null;
  const q2 = Math.acos(D) * (elbowUp ? 1 : -1), th2 = -q2;
  const th1 = Math.atan2(wz, wx) - Math.atan2(L2 * Math.sin(th2), L1 + L2 * Math.cos(th2));
  const q1 = -th1;
  return [q1, q2, phi - q1 - q2];
}

// ---- small dense linear algebra --------------------------------------------------------------------------------------
/** solve A X = B (A n x n, B n x m), Gauss-Jordan with partial pivoting */
export function solve(A, B) {
  const n = A.length, m = B[0].length;
  const M = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let j = 0; j < n + m; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < n + m; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map((r) => r.slice(n));
}

/** quintic between states s = [q, qd, qdd] (arrays) at t0 and t1; returns f(t) -> [q, qd, qdd] (clamped, then constant-velocity extrapolation) */
export function quintic(t0, s0, t1, s1) {
  const T = t1 - t0, n = s0[0].length;
  const [q0, v0, a0] = s0, [q1, v1, a1] = s1;
  const c0 = q0, c1 = v0, c2 = a0.map((a) => a / 2);
  const M = [[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]];
  const rhs = [0, 1, 2].map((r) => Array.from({ length: n }, (_, j) => (
    r === 0 ? q1[j] - c0[j] - c1[j] * T - c2[j] * T ** 2 : r === 1 ? v1[j] - c1[j] - 2 * c2[j] * T : a1[j] - 2 * c2[j])));
  const C = solve(M, rhs);   // 3 x n : c3, c4, c5
  return (t) => {
    const s = clamp(t - t0, 0, T), late = t > t1;
    const q = [], v = [], a = [];
    for (let j = 0; j < n; j++) {
      const c3 = C[0][j], c4 = C[1][j], c5 = C[2][j];
      let qj = c0[j] + c1[j] * s + c2[j] * s ** 2 + c3 * s ** 3 + c4 * s ** 4 + c5 * s ** 5;
      const vj = c1[j] + 2 * c2[j] * s + 3 * c3 * s ** 2 + 4 * c4 * s ** 3 + 5 * c5 * s ** 4;
      let aj = 2 * c2[j] + 6 * c3 * s + 12 * c4 * s ** 2 + 20 * c5 * s ** 3;
      if (late) { qj += vj * (t - t1); aj = 0; }
      q.push(qj); v.push(vj); a.push(aj);
    }
    return [q, v, a];
  };
}

/** scipy.optimize.brentq (C implementation: xtol 1e-4 as used by the Python toss teacher, rtol 4 eps, maxiter 100) */
export function brentq(f, xa, xb, xtol = 1e-4, rtol = 8.881784197001252e-16, maxiter = 100) {
  let xpre = xa, xcur = xb, xblk = 0, fblk = 0, spre = 0, scur = 0;
  let fpre = f(xpre), fcur = f(xcur);
  if (fpre === 0) return xpre;
  if (fcur === 0) return xcur;
  if (Math.sign(fpre) === Math.sign(fcur)) throw new Error("f(a) and f(b) must have different signs");
  const sgn = (x) => x < 0 || Object.is(x, -0);
  for (let i = 0; i < maxiter; i++) {
    if (fpre !== 0 && fcur !== 0 && sgn(fpre) !== sgn(fcur)) { xblk = xpre; fblk = fpre; spre = scur = xcur - xpre; }
    if (Math.abs(fblk) < Math.abs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    const delta = (xtol + rtol * Math.abs(xcur)) / 2, sbis = (xblk - xcur) / 2;
    if (fcur === 0 || Math.abs(sbis) < delta) return xcur;
    if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
      let stry;
      if (xpre === xblk) stry = (-fcur * (xcur - xpre)) / (fcur - fpre);
      else {
        const dpre = (fpre - fcur) / (xpre - xcur), dblk = (fblk - fcur) / (xblk - xcur);
        stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre));
      }
      if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) { spre = scur; scur = stry; }
      else { spre = sbis; scur = sbis; }
    } else { spre = sbis; scur = sbis; }
    xpre = xcur; fpre = fcur;
    if (Math.abs(scur) > delta) xcur += scur; else xcur += sbis > 0 ? delta : -delta;
    fcur = f(xcur);
  }
  return xcur;
}

// ---- MJCF edits of the ball (plant only: weaknesses) -------------------------------------------------------------------
/** set mass of geom `geom` and damping (drag) of the slide joints `joints` in an MJCF string */
export function withBall(xml, geom, joints, mass, drag) {
  let out = xml.replace(new RegExp(`(<geom name="${geom}"[^>]*mass=")[^"]*(")`), `$1${mass}$2`);
  for (const j of joints) out = out.replace(new RegExp(`(<joint name="${j}"[^>]*damping=")[^"]*(")`), `$1${drag}$2`);
  return out;
}

/** geom colour (hex) from the MJCF rgba, or null for the default grey of the arm links */
export function rgbaColor(model, i) {
  const c = [0, 1, 2].map((j) => model.geom_rgba[4 * i + j]);
  if (Math.abs(c[0] - 0.5) < 1e-6 && Math.abs(c[1] - 0.5) < 1e-6 && Math.abs(c[2] - 0.5) < 1e-6) return null;
  return (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8) | Math.round(c[2] * 255);
}
