# Task plugins for the Controller Arena

`app.js` knows nothing about individual tasks. Each task is one module `js/tasks/<name>.js` with a default export (the
**task definition**), registered in `js/tasks/index.js` (`TASKS = [tracking, poleBalance, ...]`). The tab bar, controller
panel, sliders, weakness controls, metrics chips, chart, camera, legend and mouse interaction are all built from it.
Working examples: `tracking.js` (wraps `../sim.js`) and `pole_balance.js` (self-contained). `?task=<id>` selects a task.

## 1. Task definition (default export)

| field | meaning |
|---|---|
| `id`, `title` | URL id (`?task=`) and tab label |
| `summary` | 1-3 sentences for a general reader: goal, classical controllers, what can be tuned |
| `assets` | `{key: path}` relative to `arena/`; fetched by `loadAssets`; `*.json` parsed, else text. Passed to `create` as `assets` |
| `camera` | `{target:[x,z], distance, fov}`; the scene is the x-z plane seen along +y (z up) |
| `chart` | `{label, scale, signed, fixedMax?, guides?, guideLabel?, skip?, minRange?, maxCap?}`: plots `session.chartValue()` (SI) times `scale` over 0..`session.duration`. `signed:true` draws +-`fixedMax` (display units), else 0..auto |
| `legendNote` | text after the colour legend |
| `interaction` | `{fmax, k, damp, hitRadius}` default mouse spring (N, N/m, N s/m, m); a grab segment may override fmax/k/damp |
| `controllers` | `[{id, label, name, color, note, params:[...], info?}]`; `info:true` shows `session.controllerInfo(id)` under the sliders |
| `weaknesses` | param specs for plant-only errors, e.g. `payload` (% of nominal), `delay` (steps) |
| `weaknessNote` | explanation under the controls |
| `options` | `[{id, label, choices, value}]` drop-downs (e.g. figure); change -> `setOption` then a new episode |
| `create(mj, assets)` | returns a **session** |

Param spec: `{id, label, min, max, step, value, fmt(actual)->string}` or `{id, label, type:"checkbox", value}`.
`scale:"log10"`: slider holds the exponent (`value` = log10 of the default); `setParam` gets the actual value 10^x.
Sliders call the session only on user input, so a new session must already start at the defaults listed in the definition.

## 2. Session (what `create` returns)

Pure simulation, no DOM, must run in node (so it can be validated against Python).

Properties: `model`, `data` (MuJoCo WASM, used for drawing and mouse forces), `dt` (control period, s), `duration` (s), `k`
(control step), `done`, `outcome` (`null | "success" | "failure" | "complete"`), `message` (banner text once done), `active`
(controller id), `recordNow` (true if the step just taken belongs to the episode: push a chart point).

Methods:
- `reset(seed)`: new episode (app passes 1000 first, then +1 per restart). Deterministic in `seed`.
- `step(hook)`: one control period in the order of `arena/env.py` (controller, delay, clip, substeps, checks). `hook(session)` runs before every physics substep (mouse force). After `done` physics keeps running, metrics stay latched.
- `setController(id)`, `setParam(controllerId, paramId, value)`, `setWeakness(id, value)`, `setOption(id, value)`, `controllerInfo(id)`.
- Display: `chartValue()`, `metrics()` -> `[[label, string], ...]`, `refPath()` -> `[[x,z],...] | null`, `target()` -> `[x,z]` (red dot), `trailPoint()` -> `[x,z]`, `geomColor(i)` -> hex or `null`, `grabSegments()` -> `[{body, a:[x,z], b:[x,z], fmax?, k?, damp?}]` (mouse grabs a body anywhere on a-b), `dispose()` (free WASM objects; called on task switch).

Rendered geoms: capsule, sphere, box (others skipped).

## 3. Conventions

- Physics must match the Python task: export MJCF from Python (`build_mjcf(spec, task.tip_xml(spec))`) to `assets/<name>.xml` and constants/gains to `assets/<name>.json`. Template: `validation/tools/export_pole.py` (also exports Python reference trajectories).
- Controllers needing the nominal model use `ArmModel` (`../sim.js`; arm-only MJCF of the nominal spec, `mj_inverse`).
- Weaknesses change the plant only. Payload = mass of the FIRST sphere geom (`common.js: withPayload`).
- `common.js`: `DelayLine`, `withPayload`, `loadXml`, `rng(seed)`, planar 2-link `fk2/jac2/jdotqd2/inv2`, `lqr(A,B,Q,R)` (matrix-sign-function Riccati, matches scipy to 1e-12).
- Validate each task: `validate_<name>.mjs` like `validate_pole.mjs` (same initial state, nominal params, fixed steps, max error; plus outcomes over seeds), numbers into `../VALIDATION.md`.
- New task = `tasks/<name>.js` + assets + one line in `tasks/index.js`. `app.js` should not need changes; if the interface lacks something, extend it in `app.js` and document it here.
