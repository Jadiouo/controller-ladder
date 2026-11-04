// Interactive arena (browser). MuJoCo WASM steps the plant, three.js draws it. Everything task-specific comes from a task
// plugin in ./tasks/ (see tasks/README.md); this file only knows the plugin interface.
import * as THREE from "../vendor/three.module.js";
import loadMujoco from "../vendor/mujoco.js";
import { TASKS, getTask, loadAssets } from "./tasks/index.js";

const $ = (id) => document.getElementById(id);
const status = (m) => ($("status").textContent = m);
const PALETTE = [0x2563eb, 0x0d9488, 0xea580c];

async function main() {
  status("loading MuJoCo WASM ...");
  const mj = await loadMujoco();
  const state = { paused: false, loop: true, stepsTotal: 0, grab: null, mouse: null, hold: 0, speed: 1, seed: 1000 };
  let task = null, session = null, series = [], meshes = [], trailN = 0;
  const assetCache = {};

  // ---------------------------------------------------------------- theme (follows the page: prefers-color-scheme / data-theme)
  const isDark = () => document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);

  // ---------------------------------------------------------------- three.js scene (shared by all tasks)
  const host = $("view");
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f7f9);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(1, -2, 3);
  scene.add(sun);
  let grid = null;
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 24), new THREE.MeshStandardMaterial({ color: 0x555b66 }));
  scene.add(base);
  const refLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5b6f9c }));
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
  const dragGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const dragLine = new THREE.Line(dragGeo, new THREE.LineBasicMaterial({ color: 0xdc2626 }));
  dragLine.visible = false;
  dragLine.frustumCulled = false;
  scene.add(dragLine);

  function applyTheme() {
    const dark = isDark();
    scene.background.set(dark ? 0x161a22 : 0xf6f7f9);
    host.style.background = dark ? "#161a22" : "#f6f7f9";
    trail.material.color.set(dark ? 0xe5e7eb : 0x111827);
    if (grid) scene.remove(grid);
    grid = dark ? new THREE.GridHelper(3, 30, 0x3a4252, 0x262c38) : new THREE.GridHelper(3, 30, 0xc5cad3, 0xe0e3e8);
    grid.position.set(0.3, 0.2, 0);
    scene.add(grid);
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);
  applyTheme();

  function buildMeshes() {
    for (const m of meshes) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
    meshes = [];
    const m = session.model;
    for (let i = 0; i < m.ngeom; i++) {
      const type = m.geom_type[i];
      const sz = [m.geom_size[3 * i], m.geom_size[3 * i + 1], m.geom_size[3 * i + 2]];
      let geo;
      if (type === mj.mjtGeom.mjGEOM_CAPSULE.value) {
        geo = new THREE.CapsuleGeometry(Math.max(sz[0], 0.008), 2 * sz[1], 8, 20);   // thin rods stay visible
        geo.rotateX(Math.PI / 2);
      } else if (type === mj.mjtGeom.mjGEOM_SPHERE.value) geo = new THREE.SphereGeometry(sz[0], 28, 20);
      else if (type === mj.mjtGeom.mjGEOM_BOX.value) geo = new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]);
      else continue;
      const color = session.geomColor?.(i) ?? PALETTE[meshes.length % 3];
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.45 }));
      mesh.matrixAutoUpdate = false;
      mesh.userData.geom = i;
      scene.add(mesh);
      meshes.push(mesh);
    }
  }
  function setRefLine() {
    const path = session.refPath();
    refLine.visible = !!path;
    if (!path) return;
    refLine.geometry.dispose();
    refLine.geometry = new THREE.BufferGeometry().setFromPoints(path.map((p) => new THREE.Vector3(p[0], 0, p[1])));
  }
  function setCamera() {
    const { target, distance, fov } = task.camera;
    camera.fov = fov;
    camera.position.set(target[0], -distance, target[1]);
    camera.lookAt(target[0], 0, target[1]);
    camera.updateProjectionMatrix();
  }
  function resize() {
    const w = host.clientWidth, h = Math.round(w * 0.78);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------------------------------------------------------------- chart
  const chart = $("chart");
  const cctx = chart.getContext("2d");
  const ctlColor = (id) => task.controllers.find((c) => c.id === id)?.color ?? "#888";
  function drawChart() {
    const dpr = window.devicePixelRatio || 1;
    const cw = chart.clientWidth, ch = chart.clientHeight;
    if (chart.width !== Math.round(cw * dpr)) { chart.width = Math.round(cw * dpr); chart.height = Math.round(ch * dpr); }
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.clearRect(0, 0, cw, ch);
    const spec = task.chart, S = spec.scale, T = session.duration, L = 44, R = 8, Tp = 8, B = 24;
    let ymax;
    if (spec.fixedMax) ymax = spec.fixedMax / S;
    else {
      ymax = spec.minRange ?? 0;
      for (const s of series) if (s.t > (spec.skip ?? 0)) ymax = Math.max(ymax, Math.abs(s.e));
      ymax = Math.min(Math.max(ymax * 1.15, spec.minRange ?? 0), spec.maxCap ?? Infinity);
    }
    const dark = isDark();
    cctx.strokeStyle = dark ? "#262c38" : "#e3e6eb"; cctx.fillStyle = dark ? "#9aa3b2" : "#6b7280";
    cctx.font = "11px system-ui, sans-serif"; cctx.lineWidth = 1;
    const H = ch - Tp - B;
    const Y = spec.signed
      ? (e) => Math.max(Tp, Math.min(ch - B, Tp + H * (0.5 - e / (2 * ymax))))
      : (e) => Math.max(Tp, ch - B - (H * e) / ymax);
    for (let i = 0; i <= 4; i++) {
      const y = Tp + (H * i) / 4;
      cctx.beginPath(); cctx.moveTo(L, y); cctx.lineTo(cw - R, y); cctx.stroke();
      const v = spec.signed ? ymax * (1 - i / 2) : (ymax * (4 - i)) / 4;
      cctx.fillText((v * S).toFixed(0), 6, y + 4);
    }
    cctx.fillText(spec.label, 6, ch - 6);
    cctx.fillText(`t (s), 0 .. ${T}`, cw - 90, ch - 6);
    for (const g of spec.guides ?? []) {
      cctx.save(); cctx.strokeStyle = "#dc2626"; cctx.setLineDash([4, 4]);
      for (const sgn of spec.signed ? [1, -1] : [1]) { const y = Y((sgn * g) / S); cctx.beginPath(); cctx.moveTo(L, y); cctx.lineTo(cw - R, y); cctx.stroke(); }
      cctx.fillStyle = "#dc2626"; cctx.fillText(spec.guideLabel ?? "", L + 6, Y(g / S) + 12);
      cctx.restore();
    }
    const X = (t) => L + ((cw - L - R) * t) / T;
    cctx.lineWidth = 2;
    let prev = null;
    for (const s of series) {
      if (prev) {
        cctx.strokeStyle = ctlColor(s.c);
        cctx.beginPath(); cctx.moveTo(X(prev.t), Y(prev.e)); cctx.lineTo(X(s.t), Y(s.e)); cctx.stroke();
      }
      prev = s;
    }
  }

  // ---------------------------------------------------------------- mouse perturbation (xfrc_applied)
  const ray = new THREE.Raycaster();
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
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
  renderer.domElement.addEventListener("pointerdown", (ev) => {
    const p = pointerWorld(ev);
    if (!p) return;
    const IA = task.interaction;
    const cand = session.grabSegments().map((s) => {
      const [d, t] = segDist(p, s.a, s.b);
      return { ...s, d, t };
    }).sort((u, v) => u.d - v.d)[0];
    if (!cand || cand.d > IA.hitRadius) return;
    // grab point in the body frame (hinge about y: rotate the offset back with the body's xmat)
    const d = session.data, xm = d.xmat, o = 9 * cand.body;
    const wx = cand.a[0] + cand.t * (cand.b[0] - cand.a[0]) - d.xpos[3 * cand.body];
    const wz = cand.a[1] + cand.t * (cand.b[1] - cand.a[1]) - d.xpos[3 * cand.body + 2];
    state.grab = { body: cand.body, lx: xm[o] * wx + xm[o + 6] * wz, lz: xm[o + 2] * wx + xm[o + 8] * wz,
      fmax: cand.fmax ?? IA.fmax, k: cand.k ?? IA.k, damp: cand.damp ?? IA.damp };
    state.mouse = p;
    renderer.domElement.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });
  renderer.domElement.addEventListener("pointermove", (ev) => {
    if (state.grab) state.mouse = pointerWorld(ev) || state.mouse;
  });
  const release = () => { state.grab = null; dragLine.visible = false; session?.data.xfrc_applied.fill(0); };
  renderer.domElement.addEventListener("pointerup", release);
  renderer.domElement.addEventListener("pointercancel", release);

  let gripVel = [0, 0], gripPrev = null;
  const gripPoint = (d, g) => {
    const b = g.body, xm = d.xmat, o = 9 * b;
    return [d.xpos[3 * b] + xm[o] * g.lx + xm[o + 2] * g.lz, d.xpos[3 * b + 2] + xm[o + 6] * g.lx + xm[o + 8] * g.lz];
  };
  function applyMouseForce(s) {
    const d = s.data;
    d.xfrc_applied.fill(0);
    const g = state.grab;
    if (!g) return;
    const b = g.body, [ax, az] = gripPoint(d, g);
    let fx = g.k * (state.mouse.x - ax) - g.damp * gripVel[0];
    let fz = g.k * (state.mouse.z - az) - g.damp * gripVel[1];
    const mag = Math.hypot(fx, fz);
    if (mag > g.fmax) { fx *= g.fmax / mag; fz *= g.fmax / mag; }
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
    const p = gripPoint(session.data, g);
    if (gripPrev) gripVel = [(p[0] - gripPrev[0]) / dt, (p[1] - gripPrev[1]) / dt];
    gripPrev = p;
  }

  // ---------------------------------------------------------------- UI built from the task definition
  function addRow(parent, labelText, el, outEl) {
    const row = document.createElement("div");
    row.className = "row";
    const lab = document.createElement("label");
    lab.textContent = labelText;
    if (el.id) lab.htmlFor = el.id;
    row.append(lab, el, outEl ?? document.createElement("span"));
    parent.append(row);
  }
  /** slider / checkbox for a param spec; `apply(actualValue)` is called on user input (never at build time: the session starts at the defaults) */
  function addControl(parent, idPrefix, p, apply) {
    if (p.type === "checkbox") {
      const el = document.createElement("input");
      el.type = "checkbox"; el.id = `${idPrefix}-${p.id}`; el.checked = p.value;
      el.onchange = () => apply(el.checked);
      addRow(parent, p.label, el);
      return;
    }
    const el = document.createElement("input"), out = document.createElement("output");
    el.type = "range"; el.id = `${idPrefix}-${p.id}`;
    el.min = p.min; el.max = p.max; el.step = p.scale === "log10" ? "any" : p.step; el.value = p.value;
    const actual = () => (p.scale === "log10" ? 10 ** +el.value : +el.value);
    out.textContent = p.fmt(actual());
    el.oninput = () => { out.textContent = p.fmt(actual()); apply(actual()); };
    addRow(parent, p.label, el, out);
  }
  function refreshInfo() {
    for (const c of task.controllers) if (c.info) $(`info-${c.id}`).textContent = session.controllerInfo(c.id);
  }
  function selectController(id) {
    session.setController(id);
    const c = task.controllers.find((x) => x.id === id);
    document.querySelectorAll(".ctl-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ctl === id)));
    document.querySelectorAll(".ctl-panel").forEach((p) => (p.hidden = p.dataset.ctl !== id));
    $("ctl-name").textContent = c.name;
  }
  function buildUI() {
    $("blurb").textContent = task.summary;
    document.title = `${task.title} - Controller Arena`;
    $("legend").innerHTML = task.controllers.map((c) => `<span style="background:${c.color}"></span>${c.name}`).join(" ") + ` &nbsp; (${task.legendNote})`;
    const seg = $("ctl-seg"), panels = $("ctl-panels");
    seg.innerHTML = ""; panels.innerHTML = "";
    for (const c of task.controllers) {
      const b = document.createElement("button");
      b.className = "ctl-btn"; b.dataset.ctl = c.id; b.textContent = c.label;
      b.onclick = () => selectController(c.id);
      seg.append(b);
      const p = document.createElement("div");
      p.className = "ctl-panel"; p.dataset.ctl = c.id;
      for (const prm of c.params) addControl(p, `p-${c.id}`, prm, (v) => { session.setParam(c.id, prm.id, v); refreshInfo(); });
      const note = document.createElement("p");
      note.className = "small"; note.textContent = c.note;
      p.append(note);
      if (c.info) { const i = document.createElement("p"); i.className = "small"; i.id = `info-${c.id}`; p.append(i); }
      panels.append(p);
    }
    const tc = $("task-controls");
    tc.innerHTML = "";
    for (const o of task.options) {
      const sel = document.createElement("select");
      sel.id = `opt-${o.id}`;
      o.choices.forEach((ch) => sel.add(new Option(ch, ch)));
      sel.value = o.value;
      sel.onchange = () => { session.setOption(o.id, sel.value); resetEpisode(false); };
      addRow(tc, o.label, sel);
    }
    for (const w of task.weaknesses) addControl(tc, "w", w, (v) => session.setWeakness(w.id, v));
    $("weak-note").textContent = task.weaknessNote;
    selectController(task.controllers[0].id);
    refreshInfo();
  }

  function resetEpisode(newSeed = true) {
    if (newSeed) state.seed += 1;
    session.reset(state.seed);
    series = [];
    trailN = 0;
    state.hold = 0;
    release();
    setRefLine();
  }

  async function selectTask(id, push = true) {
    const next = getTask(id);
    if (next === task) return;
    status(`loading ${next.title} ...`);
    release();
    assetCache[next.id] ??= await loadAssets(next);
    session?.dispose();
    task = next;
    session = task.create(mj, assetCache[task.id]);
    state.seed = 1000;
    document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.task === task.id)));
    buildMeshes();
    setCamera();
    buildUI();
    session.reset(state.seed);
    series = []; trailN = 0; state.hold = 0;
    setRefLine();
    if (push) { const u = new URL(location.href); u.searchParams.set("task", task.id); history.replaceState(null, "", u); }
    window.__arena = { sim: session, session, task, state, mj, THREE, camera, renderer, selectController, resetEpisode, selectTask, series: () => series };
  }

  const tabs = $("tabs");
  for (const t of TASKS) {
    const b = document.createElement("button");
    b.setAttribute("role", "tab"); b.dataset.task = t.id; b.textContent = t.title;
    b.onclick = () => selectTask(t.id);
    tabs.append(b);
  }
  $("reset").onclick = () => resetEpisode(true);
  $("pause").onclick = () => { state.paused = !state.paused; $("pause").textContent = state.paused ? "Resume" : "Pause"; };
  $("loop").onchange = (e) => (state.loop = e.target.checked);
  $("speed").onchange = (e) => (state.speed = +e.target.value);
  const wanted = new URLSearchParams(location.search).get("task") || location.hash.replace(/^#(task=)?/, "");
  await selectTask(wanted || TASKS[0].id, false);

  // ---------------------------------------------------------------- main loop
  const banner = $("banner");
  let acc = 0, last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    acc += Math.min(0.05, (now - last) / 1000) * state.speed;
    last = now;
    if (!state.paused) {
      let n = Math.min(16, Math.floor(acc / session.dt));
      acc -= n * session.dt;
      for (; n > 0; n--) {
        if (session.done) {
          if (!state.loop) break;
          if (++state.hold > 60) resetEpisode(true);
        }
        updateGripVel(session.dt);
        session.step(applyMouseForce);
        state.stepsTotal++;
        if (session.recordNow) series.push({ t: session.k * session.dt, e: session.chartValue(), c: session.active });
        if (session.k % 2 === 0) {
          const p = session.trailPoint();
          if (trailN < TRAIL) trailN++; else trailPos.copyWithin(0, 3);
          trailPos.set([p[0], 0, p[1]], (trailN - 1) * 3);
          trailGeo.attributes.position.needsUpdate = true;
          trailGeo.setDrawRange(0, trailN);
        }
      }
    }
    // draw
    const d = session.data;
    for (const mesh of meshes) {
      const g = mesh.userData.geom, x = d.geom_xpos, R = d.geom_xmat;
      mesh.matrix.set(R[9 * g], R[9 * g + 1], R[9 * g + 2], x[3 * g], R[9 * g + 3], R[9 * g + 4], R[9 * g + 5], x[3 * g + 1],
        R[9 * g + 6], R[9 * g + 7], R[9 * g + 8], x[3 * g + 2], 0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
    const tg = session.target();
    refDot.position.set(tg[0], 0, tg[1]);
    if (state.grab && state.mouse) {
      const gp = state.gripPoint || [0, 0];
      dragGeo.setFromPoints([new THREE.Vector3(gp[0], 0, gp[1]), new THREE.Vector3(state.mouse.x, 0, state.mouse.z)]);
      dragLine.visible = true;
    }
    renderer.render(scene, camera);
    drawChart();
    $("metrics").innerHTML = session.metrics().map(([k, v]) => `<div><small>${k}</small><b>${v}</b></div>`).join("") +
      (state.grab ? `<div><small>pull</small><b>${(state.force ?? 0).toFixed(1)} N</b></div>` : "");
    banner.hidden = !(session.done && session.message);
    banner.textContent = session.message;
    banner.className = session.outcome === "failure" ? "fail" : "ok";
    status(`MuJoCo ${mj.mj_versionString ? mj.mj_versionString() : ""} running, ${state.stepsTotal} control steps, episode seed ${state.seed}`);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  status("failed to start: " + e.message);
});
