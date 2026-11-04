# Release notes (public snapshot, prepared 2026-11-03)

Status: a fresh single-commit export built from the private development repository. Nothing has been pushed anywhere; no remote is configured. The commit author is the neutral placeholder `Author <author@example.com>`.

## Included

* `arena/`, `arena_mjx/`, `tests/`, `pyproject.toml` (code and tests, unchanged except path/tool-name scrubbing in comments and logs).
* All `spike*/` code folders (every spike is referenced by `docs/experiments.tsv` or `claims_evidence.md`); `__pycache__` removed.
* `scripts/` demos and MJX scripts (without `school_sync.sh`).
* `docs/`: `03_hypotheses.md`, `rulers.md`, `experiments.tsv`, `canonical_numbers.md`, `paper/` (skeleton, claims_evidence, prespec_audit, review_r1, review_r2, r2_fixlog, short_2p, citations_check, related_check, `stats/`), new `provenance.md`, new `ARCHITECTURE.md` (old README body, scrubbed).
* `results/`: all tracked files <= 1 MB of type md, json, txt, jsonl, log, plus small `.npy` arrays (316 KB total) and two matplotlib `.png` (metadata: only the matplotlib software tag), and `.test_access.log` (111 lines, no paths).
* New: `README.md` (public), `LICENSE` (MIT placeholder), `CITATION.cff` (placeholder), `.gitignore`.

## Excluded

* Private infrastructure and process: `.venv*`, `.scratch/`, `*.egg-info`, `MUJOCO_LOG.TXT`, `CLAUDE.md`, `docs/progress.md`, `docs/devlog/`, `docs/00_original_request.md`, `docs/01_plan_draft.md`, `docs/02_lit_check.md`, `docs/project_type.md`, `scripts/school_sync.sh`.
* `docs/paper/anonymization.md` (the leak checklist, itself describes leaks), `docs/paper/venues.md` (venue planning / submission strategy).
* `results/spike16a/cmp_*.json|txt` (cross-machine check files containing host strings). The conclusion (30/30 episodes identical) stays in the spike16a README and rulers.md.
* Binary results (not needed to re-build tables; sizes from the private repo): model weights `.pt` 66 MB (learn_demo ckpts 2.2 MB each x 24, spike15c models 1.7 MB each x 9), data/policy `.npz` 35 MB (learn_demo shards 4.2 MB x 8, mjx policies), videos `.mp4` 2.1 MB and `.gif` 0.6 MB. All are each < 20 MB, so including the weights is possible; they were left out because no table in the paper needs them to be regenerated (the per-episode `.npy` arrays are included). Decision pending: include `spike15c/models` (15 MB) and `learn_demo/ckpt` (53 MB)? Pickles inside `.pt` could carry paths; they were checked in the private repo with `git grep -a` (no hits) but would be re-checked with `strings` before inclusion.
* `web/` and `docs/paper/arxiv/` (being prepared separately; to be merged later).

## Rewrites

* Absolute paths `/home/<account>/Documents/robot_arm_control/...` -> relative paths (`./`, `.scratch/...`) or bare `python` (scripts, logs, `selection.json`).
* Machine names: "school machine" / `ssh school` -> "second machine" / `machine-B`; local machine -> `machine-A`.
* GPU queue tool name -> "a GPU job queue". Personal-role words ("user", "user decision", in Chinese docs "使用者") -> "author".
* Docs mentioning the removed internal files (`docs/progress.md`, `CLAUDE.md`, `anonymization.md`) still contain the references as text; they are dangling pointers and harmless.

## Known residual items (for the owner to decide)

1. **Hardware mentions**: "RTX 5070 Ti" remains in `scripts/demo_students.py`, `results/learn_demo/table.md`, `results/mjx/README.md`; "RTX A5000" in `related_check.md` (third-party). A GPU model is a reproducibility detail but can narrow down who owns the machine.
2. **Workflow disclosure**: the docs describe closed-door and white-team "review" rounds (`review_r1/r2.md`) and a "no third round by author decision" workflow; they reveal an AI-assisted or committee-simulated process. Disclose or remove according to the target venue's policy.
3. **Language**: several docs are in Chinese (`03_hypotheses.md`, `rulers.md`, review files).
4. Commit time zone +08:00 appears in `docs/provenance.md` (weak regional signal).
5. License holder, author names in `CITATION.cff`, paper link in `README.md`.
6. Provenance is an assertion until the private repo is shown (see `docs/provenance.md`); consider an external timestamp.

## Audit

See the verification section appended at the end of this file (scan results and pytest).

## Verification (2026-11-03)

* Scan over tracked files for the account names, e-mail domain, `/home/`, machine/school/VPN/session/AI-tool strings, trailers: 0 hits outside this file (this file names the removed items, an intended exception). Generic `ssh machine-B` (placeholder alias) remains in `results/spike16a/README.md` by design. The words `lex` as a substring appear only inside ordinary words.
* `python -m pytest -q` in a fresh `.venv` (install `-e . pytest`, run under `flock` + `systemd-run` MemoryMax=6G): 194 passed, 6 skipped, 343 s.
* Git: one commit, neutral author, no remote.

## Second snapshot (2026-11-03)
* Added `web/` (portfolio page, 11 task videos, MuJoCo WASM arena with vendored mujoco@3.1.16 and three r160; upstream build paths inside the vendored binaries are not ours), `docs/paper/arxiv/` (LaTeX sources, figures script, compiled PDF; build artefacts excluded), and the frozen whip estimator weights `results/spike15c/models/*.pt` (15 MB).
* Removed the internal AI-simulated review records (`review_r1.md`, `review_r2.md`, `r2_fixlog.md`, `review_a1.md`); some documents still mention them by name. The paper's acknowledgement discloses that reviews were simulated with AI agents.
* License: MIT (author confirmed); copyright holder and author names remain placeholders.

## Layout change (branch `reorg`)

Pure move plus path fixes; no experiment logic or number was changed.

* `spike*/` -> `experiments/spike*/` (folder names unchanged; index in `experiments/README.md`).
* `docs/paper/arxiv/` -> `paper/arxiv/`; the other files of `docs/paper/` (claims_evidence, prespec_audit, stats/, ...) -> `paper/support/`.
* Path strings inside code, shell scripts, docs, READMEs, `docs/experiments.tsv` and the generated `table.md` headers were updated. Historical records that quote old paths (for example the md5 keys in `results/spike09b/final_methods.json`, `results/spike10b/final_methods.json`, `results/.test_access.log`, run logs) are left as they were written.
* The entries above this section describe the layout at the time of the snapshot (old paths).
