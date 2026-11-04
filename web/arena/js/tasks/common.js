// Helpers shared by task modules (pure JS, no DOM; runs in the browser and in node).
import { ArmModel } from "../sim.js";

export const clip = (x, lim) => Math.max(-lim, Math.min(lim, x));

/** small deterministic RNG (mulberry32) for episode resets in the browser */
export function rng(seed) {
  let a = (seed >>> 0) + 0x6d2b79f5;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** write an XML string into MuJoCo's virtual FS and load it */
export function loadXml(mj, xml, path) {
  mj.FS.writeFile(path, xml);
  return mj.MjModel.mj_loadXML(path);
}

/** replace the mass of the FIRST sphere geom (the tip payload in every arm model of this repo) */
export function withPayload(xml, mass) {
  return xml.replace(/(<geom type="sphere"[^>]*mass=")[^"]*(")/, `$1${mass}$2`);
}

/** Delay.on_command of arena/perturb/effects.py: returns the command applied this step */
export class DelayLine {
  constructor() { this.hist = []; this.steps = 0; }
  clear() { this.hist = []; }
  push(cmd) {
    this.hist.push(cmd.slice());
    const out = this.hist[Math.max(0, this.hist.length - 1 - this.steps)];
    if (this.hist.length > this.steps + 2) this.hist.shift();
    return out;
  }
}

// ---- planar n=2 kinematics (arena/robots/kinematics.py) ------------------------------------------------------------
export const fk2 = (L, q) => {
  const p0 = q[0], p1 = q[0] + q[1];
  return [L[0] * Math.cos(p0) + L[1] * Math.cos(p1), -(L[0] * Math.sin(p0) + L[1] * Math.sin(p1))];
};
export const jac2 = (L, q) => {
  const p0 = q[0], p1 = q[0] + q[1];
  const dx0 = -L[0] * Math.sin(p0), dx1 = -L[1] * Math.sin(p1), dz0 = -L[0] * Math.cos(p0), dz1 = -L[1] * Math.cos(p1);
  return [[dx0 + dx1, dx1], [dz0 + dz1, dz1]];
};
export const jdotqd2 = (L, q, qd) => {
  const p0 = q[0], p1 = q[0] + q[1], w0 = qd[0], w1 = qd[0] + qd[1];
  return [-(L[0] * Math.cos(p0) * w0 * w0 + L[1] * Math.cos(p1) * w1 * w1), L[0] * Math.sin(p0) * w0 * w0 + L[1] * Math.sin(p1) * w1 * w1];
};
/** inverse of a 2x2 (equals the pseudo-inverse for a non-singular square Jacobian) */
export const inv2 = (J) => {
  const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  return [[J[1][1] / det, -J[0][1] / det], [-J[1][0] / det, J[0][0] / det]];
};
export const mulv2 = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];

// ---- dense matrices (arrays of rows) and a continuous-time algebraic Riccati solver ----------------------------------
const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const mul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
const add = (A, B, s = 1) => A.map((r, i) => r.map((a, j) => a + s * B[i][j]));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
function solveLin(A, B) { // A X = B, Gauss-Jordan with partial pivoting
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
const fro = (A) => Math.sqrt(A.reduce((s, r) => s + r.reduce((t, a) => t + a * a, 0), 0));

/** LQR gain K (u = -K x) for xdot = A x + B u, cost x'Qx + u'Ru, via the matrix sign function of the Hamiltonian.
 *  Matches scipy.linalg.solve_continuous_are to ~1e-12 on the pole problem (checked in validate_pole.mjs). */
export function lqr(A, B, Q, R) {
  const n = A.length;
  const Rinv = solveLin(R, eye(R.length));
  const BRB = mul(mul(B, Rinv), T(B));
  let Z = [];
  for (let i = 0; i < n; i++) Z.push([...A[i], ...BRB[i].map((x) => -x)]);
  for (let i = 0; i < n; i++) Z.push([...Q[i].map((x) => -x), ...T(A)[i].map((x) => -x)]);
  for (let it = 0; it < 200; it++) {
    const Zi = solveLin(Z, eye(2 * n));
    const c = Math.sqrt(fro(Zi) / fro(Z));
    const Zn = add(Z.map((r) => r.map((x) => (x * c) / 2)), Zi.map((r) => r.map((x) => x / (2 * c))));
    const diff = fro(add(Zn, Z, -1));
    Z = Zn;
    if (diff < 1e-13 * Math.max(1, fro(Z))) break;
  }
  // [Z12; Z22 + I] P = -[Z11 + I; Z21]  (overdetermined, normal equations)
  const left = [], right = [];
  for (let i = 0; i < 2 * n; i++) {
    left.push(Z[i].slice(n));
    const row = Z[i].slice(0, n).map((x) => -x);
    if (i < n) row[i] -= 1;
    right.push(row);
  }
  for (let i = 0; i < n; i++) left[n + i][i] += 1;
  const P = solveLin(mul(T(left), left), mul(T(left), right));
  const Ps = P.map((r, i) => r.map((x, j) => (x + P[j][i]) / 2));
  return mul(mul(Rinv, T(B)), Ps)[0];
}

export { ArmModel };
