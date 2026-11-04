// Interactive arena (browser). MuJoCo WASM steps the plant, three.js draws it, controllers come from sim.js.
import * as THREE from "../vendor/three.module.js";
import loadMujoco from "../vendor/mujoco.js";
import { ArmSim, GLYPHS, lqrDoubleIntegrator } from "./sim.js";

const $ = (id) => document.getElementById(id);
const status = (m) => ($("status").textContent = m);

async function main() {
  status("loading MuJoCo WASM ...");
  const [mj, xml, refs] = await Promise.all([
    loadMujoco(),
    fetch("assets/arm2.xml").then((r) => r.text()),
    fetch("assets/tracking_refs.json").then((r) => r.json()),
  ]);
  const sim = new ArmSim(mj, xml, refs);
  const state = { paused: false, loop: true, stepsTotal: 0, grab: null, mouse: null, hold: 0, speed: 1 };

  // ---------------------------------------------------------------- three.js scene
  const host = $("view");
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f7f9);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
  camera.up.set(0, 0, 1);
  camera.position.set(0.38, -2.0, -0.02);
  camera.lookAt(0.38, 0, -0.02);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(1, -2, 3);
  scene.add(sun);
  const grid = new THREE.GridHelper(3, 30, 0xc5cad3, 0xe0e3e8);
  grid.position.set(0.3, 0.2, 0);
  scene.add(grid);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 24), new THREE.MeshStandardMaterial({ color: 0x555b66 }));
  scene.add(base);

  const colors = [0x2563eb, 0x0d9488, 0xea580c];
  const meshes = [];
  const m = sim.model;
  for (let i = 0; i < m.ngeom; i++) {
    const type = m.geom_type[i];
    const sz = [m.geom_size[3 * i], m.geom_size[3 * i + 1], m.geom_size[3 * i + 2]];
    let geo;
    if (type === mj.mjtGeom.mjGEOM_CAPSULE.value) {
      geo = new THREE.CapsuleGeometry(sz[0], 2 * sz[1], 8, 20);
      geo.rotateX(Math.PI / 2);
    } else if (type === mj.mjtGeom.mjGEOM_SPHERE.value) geo = new THREE.SphereGeometry(sz[0], 28, 20);
    else continue;
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: colors[meshes.length % 3], roughness: 0.45 }));
    mesh.matrixAutoUpdate = false;
    mesh.userData.geom = i;
    scene.add(mesh);
    meshes.push(mesh);
  }

  const lineMat = new THREE.LineBasicMaterial({ color: 0x5b6f9c });
  const refLine = new THREE.Line(new THREE.BufferGeometry(), lineMat);
  scene.add(refLine);
  const refDot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12), new THREE.MeshBasicMaterial({ color: 0xdc2626 }));
  scene.add(refDot);
  const TRAIL = 300;
  const trailPos = new Float32Array(TRAIL * 3);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setDrawRange(0, 0);
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x111827 }));
  trail.frustumCulled = false;
  scene.add(trail);
  let trailN = 0;
  const dragGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const dragLine = new THREE.Line(dragGeo, new THREE.LineBasicMaterial({ color: 0xdc2626 }));
  dragLine.visible = false;
  dragLine.frustumCulled = false;
  scene.add(dragLine);

  function setRefLine() {
    const pts = sim.ref.EE.map((p) => new THREE.Vector3(p[0], 0, p[1]));
    refLine.geometry.dispose();
    refLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }

  function resize() {
    const w = host.clientWidth, h = Math.round(w * 0.78);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------------------------------------------------------------- error chart
  const chart = $("chart");
  const cctx = chart.getContext("2d");
  let series = []; // {t, e, c}
  const CTL_COLOR = { pid: "#2563eb", ctc: "#dc2626" };
  const CTL_NAME = { pid: "PID + gravity comp.", ctc: "CTC + LQR" };
  function drawChart() {
    const dpr = window.devicePixelRatio || 1;
    const cw = chart.clientWidth, ch = chart.clientHeight;
    if (chart.width !== cw * dpr) { chart.width = cw * dpr; chart.height = ch * dpr; }
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.clearRect(0, 0, cw, ch);
    const T = sim.refs.duration, L = 44, R = 8, Tp = 8, B = 24;
    let ymax = 0.02;
    for (const s of series) if (s.t > 0.3) ymax = Math.max(ymax, s.e); // the initial-condition error of the first 0.3 s is clipped
    ymax = Math.min(Math.max(ymax * 1.15, 0.02), 1.0);
    cctx.strokeStyle = "#e3e6eb"; cctx.fillStyle = "#6b7280"; cctx.font = "11px system-ui, sans-serif"; cctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Tp + ((ch - Tp - B) * i) / 4;
      cctx.beginPath(); cctx.moveTo(L, y); cctx.lineTo(cw - R, y); cctx.stroke();
      cctx.fillText(((ymax * (4 - i)) / 4 * 1000).toFixed(0), 6, y + 4);
    }
    cctx.fillText("tip error (mm)", 6, ch - 6);
    cctx.fillText(`t (s), 0 .. ${T}`, cw - 90, ch - 6);
    const X = (t) => L + ((cw - L - R) * t) / T, Y = (e) => Math.max(Tp, ch - B - ((ch - Tp - B) * e) / ymax);
    cctx.lineWidth = 2;
    let prev = null;
    for (const s of series) {
      if (prev) {
        cctx.strokeStyle = CTL_COLOR[s.c];
        cctx.beginPath(); cctx.moveTo(X(prev.t), Y(prev.e)); cctx.lineTo(X(s.t), Y(s.e)); cctx.stroke();
      }
      prev = s;
    }
  }

  // ---------------------------------------------------------------- mouse perturbation (xfrc_applied)
  const ray = new THREE.Raycaster();
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const names = [];
  for (let b = 0; b < sim.model.nbody; b++) names.push(b);
  function pointerWorld(ev) {
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -(((ev.clientY - r.top) / r.height) * 2 - 1));
    ray.setFromCamera(ndc, camera);
    const p = new THREE.Vector3();
    return ray.ray.intersectPlane(plane, p) ? p : null;
  }
  const segDist = (p, a, b) => {
    const abx = b[0] - a[0], abz = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((p.x - a[0]) * abx + (p.z - a[1]) * abz) / (abx * abx + abz * abz)));
    return [Math.hypot(p.x - (a[0] + t * abx), p.z - (a[1] + t * abz)), t];
  };
  function bodyXZ() {
    const x = sim.data.xpos; // body positions: world, l1, l2, tip
    return [0, 1, 2, 3].map((b) => [x[3 * b], x[3 * b + 2]]);
  }
  renderer.domElement.addEventListener("pointerdown", (ev) => {
    const p = pointerWorld(ev);
    if (!p) return;
    const B = bodyXZ();
    const cand = [[1, B[1], B[2]], [2, B[2], B[3]]].map(([body, a, b]) => {
      const [d, t] = segDist(p, a, b);
      return { body, d, t, a, b };
    }).sort((u, v) => u.d - v.d)[0];
    if (cand.d > 0.14) return;
    // grab point in the body frame (hinge about y: rotate the offset back with the body's xmat)
    const xm = sim.data.xmat, o = 9 * cand.body;
    const wx = cand.a[0] + cand.t * (cand.b[0] - cand.a[0]) - sim.data.xpos[3 * cand.body];
    const wz = cand.a[1] + cand.t * (cand.b[1] - cand.a[1]) - sim.data.xpos[3 * cand.body + 2];
    // local = R^T * w   (R row-major, columns = body axes)
    const lx = xm[o] * wx + xm[o + 6] * wz, lz = xm[o + 2] * wx + xm[o + 8] * wz;
    state.grab = { body: cand.body, lx, lz };
    state.mouse = p;
    renderer.domElement.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });
  renderer.domElement.addEventListener("pointermove", (ev) => {
    if (state.grab) state.mouse = pointerWorld(ev) || state.mouse;
  });
  const release = () => { state.grab = null; dragLine.visible = false; sim.data.xfrc_applied.fill(0); };
  renderer.domElement.addEventListener("pointerup", release);
  renderer.domElement.addEventListener("pointercancel", release);

  const FMAX = 120, KSPR = 220, KDAMP = 18;
  let gripVel = [0, 0], gripPrev = null;
  function applyMouseForce(s) {
    const d = s.data;
    d.xfrc_applied.fill(0);
    const g = state.grab;
    if (!g) return;
    const b = g.body, xm = d.xmat, o = 9 * b;
    const ax = d.xpos[3 * b] + xm[o] * g.lx + xm[o + 2] * g.lz;
    const az = d.xpos[3 * b + 2] + xm[o + 6] * g.lx + xm[o + 8] * g.lz;
    let fx = KSPR * (state.mouse.x - ax) - KDAMP * gripVel[0];
    let fz = KSPR * (state.mouse.z - az) - KDAMP * gripVel[1];
    const mag = Math.hypot(fx, fz);
    if (mag > FMAX) { fx *= FMAX / mag; fz *= FMAX / mag; }
    const rx = ax - d.xipos[3 * b], rz = az - d.xipos[3 * b + 2];
    d.xfrc_applied[6 * b + 0] = fx;
    d.xfrc_applied[6 * b + 2] = fz;
    d.xfrc_applied[6 * b + 4] = rz * fx - rx * fz; // torque about y at the body COM
    state.gripPoint = [ax, az];
    state.force = mag;
  }
  function updateGripVel(dt) {
    const g = state.grab;
    if (!g) { gripPrev = null; return; }
    const d = sim.data, b = g.body, xm = d.xmat, o = 9 * b;
    const ax = d.xpos[3 * b] + xm[o] * g.lx + xm[o + 2] * g.lz;
    const az = d.xpos[3 * b + 2] + xm[o + 6] * g.lx + xm[o + 8] * g.lz;
    if (gripPrev) gripVel = [(ax - gripPrev[0]) / dt, (az - gripPrev[1]) / dt];
    gripPrev = [ax, az];
  }

  // ---------------------------------------------------------------- UI wiring
  const glyphSel = $("glyph");
  GLYPHS.forEach((g) => glyphSel.add(new Option(g, g)));
  function resetEpisode() {
    sim.reset();
    series = [];
    trailN = 0;
    state.hold = 0;
    release();
    setRefLine();
  }
  glyphSel.onchange = () => { sim.setGlyph(glyphSel.value); resetEpisode(); };
  $("reset").onclick = resetEpisode;
  $("pause").onclick = () => { state.paused = !state.paused; $("pause").textContent = state.paused ? "Resume" : "Pause"; };
  $("loop").onchange = (e) => (state.loop = e.target.checked);
  function selectController(name) {
    sim.select(name);
    document.querySelectorAll(".ctl-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ctl === name)));
    $("panel-pid").hidden = name !== "pid";
    $("panel-ctc").hidden = name !== "ctc";
    $("ctl-name").textContent = CTL_NAME[name];
  }
  document.querySelectorAll(".ctl-btn").forEach((b) => (b.onclick = () => selectController(b.dataset.ctl)));
  const bind = (id, fn, fmt = (v) => v) => {
    const el = $(id), out = $(id + "-v");
    const f = () => { out.textContent = fmt(+el.value); fn(+el.value); };
    el.oninput = f; f();
  };
  bind("pid-wn", (v) => { sim.controllers.pid.wn = v; sim.controllers.pid.kp = null; }, (v) => v.toFixed(0) + " rad/s");
  bind("pid-ki", (v) => (sim.controllers.pid.ki = v), (v) => v.toFixed(0));
  $("pid-grav").onchange = (e) => (sim.controllers.pid.gravity = e.target.checked);
  const ctcUpdate = () => {
    const qp = 10 ** +$("ctc-qp").value, qv = 10 ** +$("ctc-qv").value;
    const [kp, kd] = lqrDoubleIntegrator(qp, qv);
    sim.controllers.ctc.Kp = kp; sim.controllers.ctc.Kd = kd;
    $("ctc-qp-v").textContent = qp.toExponential(1); $("ctc-qv-v").textContent = qv.toExponential(1);
    $("ctc-gains").textContent = `Kp = ${kp.toFixed(0)}, Kd = ${kd.toFixed(1)}`;
  };
  $("ctc-qp").oninput = ctcUpdate; $("ctc-qv").oninput = ctcUpdate; ctcUpdate();
  bind("payload", (v) => sim.setPayload(sim.nominalPayload * (1 + v / 100)), (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %");
  bind("delay", (v) => sim.setDelay(v), (v) => `${v} steps (${(v * sim.dt * 1000).toFixed(0)} ms)`);
  $("speed").onchange = (e) => (state.speed = +e.target.value);
  selectController("pid");
  setRefLine();

  // ---------------------------------------------------------------- main loop
  let acc = 0, last = performance.now(), sumSq = 0, nErr = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    acc += Math.min(0.05, (now - last) / 1000) * state.speed;
    last = now;
    if (!state.paused) {
      let n = Math.min(16, Math.floor(acc / sim.dt));
      acc -= n * sim.dt;
      for (; n > 0; n--) {
        if (sim.k >= sim.nSteps) {
          if (state.loop && ++state.hold > 60) { resetEpisode(); sumSq = 0; nErr = 0; }
          else if (!state.loop) break;
        }
        updateGripVel(sim.dt);
        sim.step(applyMouseForce);
        state.stepsTotal++;
        if (sim.k <= sim.nSteps) {
          const e = sim.errors().tip;
          series.push({ t: sim.k * sim.dt, e, c: sim.active });
          sumSq += e * e; nErr++;
        }
        if (sim.k % 2 === 0) {
          const p = sim.tipPos();
          if (trailN < TRAIL) trailN++; else trailPos.copyWithin(0, 3);
          trailPos.set([p[0], 0, p[1]], (trailN - 1) * 3);
          trailGeo.attributes.position.needsUpdate = true;
          trailGeo.setDrawRange(0, trailN);
        }
      }
    }
    // draw
    const d = sim.data;
    for (const mesh of meshes) {
      const g = mesh.userData.geom, x = d.geom_xpos, R = d.geom_xmat;
      mesh.matrix.set(R[9 * g], R[9 * g + 1], R[9 * g + 2], x[3 * g], R[9 * g + 3], R[9 * g + 4], R[9 * g + 5], x[3 * g + 1],
        R[9 * g + 6], R[9 * g + 7], R[9 * g + 8], x[3 * g + 2], 0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
    const r = sim.referenceAt(sim.k);
    refDot.position.set(r.ee[0], 0, r.ee[1]);
    if (state.grab && state.mouse) {
      const gp = state.gripPoint || [0, 0];
      dragGeo.setFromPoints([new THREE.Vector3(gp[0], 0, gp[1]), new THREE.Vector3(state.mouse.x, 0, state.mouse.z)]);
      dragLine.visible = true;
    }
    renderer.render(scene, camera);
    drawChart();
    const rms = nErr ? Math.sqrt(sumSq / nErr) * 1000 : 0;
    $("readout").textContent = `t = ${(sim.k * sim.dt).toFixed(2)} s   |   tip error ${(sim.errors().tip * 1000).toFixed(1)} mm   |   episode RMS ${rms.toFixed(1)} mm   |   torque ${sim.lastTau ? sim.lastTau.map((x) => x.toFixed(0)).join(", ") : "-"} Nm${state.grab ? `   |   pull ${state.force?.toFixed(0) ?? 0} N` : ""}`;
    status(`MuJoCo ${mj.mj_versionString ? mj.mj_versionString() : ""} running, ${state.stepsTotal} control steps`);
  }
  requestAnimationFrame(frame);
  window.__arena = { sim, state, mj, THREE, camera, renderer, selectController, resetEpisode, series: () => series };
}

main().catch((e) => {
  console.error(e);
  status("failed to start: " + e.message);
});
