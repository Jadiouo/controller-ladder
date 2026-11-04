# 2-page extended-abstract skeleton (backup version) -- status: DRAFT v1 from skeleton.md v4

Purpose: ready-to-use fallback if the accepted ICRA 2027 workshop (list 2027-01-31) or the L4DC 2027 late-breaking track (2027-04-05) allows only a 2-page extended abstract (`venues.md` rows for ICRA workshops and L4DC). Not a new result: every number below is copied from `skeleton.md` v4 (abstract, K1-K4, Section 5) and carries the line number of its row in `docs/paper/claims_evidence.md` (written `CE:Lnn`). **No number may be added here; if a number is needed that is not listed, take it from the main draft first.** Page limit and reference treatment of the real venue are unknown until the CFP appears: re-check word budget then.

## 1. Word budget (2 pages, two columns or one, with 1 figure and 1 table)

Target: 900-1000 words of running text including captions; references excluded (check the CFP; if references count, cut K4 and the related-work half sentence first).

| Block | Budget (words) | Draft below (words, counted) | Float |
|---|---|---|---|
| Title + abstract (merged into the Intro if the venue wants no separate abstract) | 110 | 80 (merged abstract/intro) | |
| 1 Intro (question + scope) | (included above) | | |
| 2 Protocol (K1) | 150 | 125 | |
| 3 Cross-setting result (K2) | 190 | 170 | Table 1 (compact, 3-4 rows), about 60 words of caption/notes |
| 4 Where the whip headroom is (K3) | 200 | 136 | Fig 1 (= skeleton Fig 3, TEST-D bars), about 50 words of caption |
| 5 Multimodal teacher (K4), one sentence or move out | 25 | 23 | |
| 6 Limitations (mandatory qualifiers) | 110 | 55 | |
| Sum of running text | about 785 | 589 | 1 figure, 1 table |
| Captions (Table 1 + Fig 1) | 110 | not drafted | |

Reading: 589 draft words + about 110 caption words is about 700, so about 200-300 words of the 900-1000 target are free. Use them first for (a) a standalone 120-word abstract taken from `skeleton.md` Section 1 if the venue asks for one, (b) the Limitations budget (55 of 110 used: add budget ratios 9-17x from skeleton Section 5 item (5) and holdout reuse), (c) one related-work sentence (4-page related work bullets are in `skeleton.md` Section 3.2). Do not use the room for new numbers.

Page cost estimate (inference): Fig 1 single-column 0.25 p, Table 1 0.2 p, text 0.9 p at about 1000 words in a two-column 10 pt template; verify against the real template.

Floats:
- Table 1 (compact): rows = (S3 imitation, S5/S6 PPO, S7 whip: best learned vs strongest classical), columns = paired diff with CI, n, "learned?" flag. Source: `skeleton.md` Section 4 Table 1 sources; full S1-S8 goes to the long version. Do not show a number that is not in the CE rows cited in Section 3.
- Fig 1: success of R0, C-ens, B, R1 on TEST-D with 95% CI and recovered fractions (source `skeleton.md` Section 4 Fig 3, `results/spike15e/table.md:18-26`).
- Dropped from the 2-page version (stay in the 4-page paper / appendix): Fig 1 protocol schematic (replace with 3 lines of text), ladder figure on TEST-C, Table 2 (gate), related work (half a sentence), fragility, dose curve, spike16a mini tables.

## 2. Qualifiers that must appear (do not delete when cutting)

| # | Qualifier | Where | CE line |
|---|---|---|---|
| Q1 | One simulator (MuJoCo), mostly one task (whip); mechanistic claims rest on whip alone | abstract last sentence and Limitations | skeleton Section 5 item (4) (scope statement, no CE row) |
| Q2 | Whip qualification on K2: the learned side is ahead of the strongest classical by +5.6 pp [0.6, 10.6]; 10 pp cannot be excluded (H16b not supported). Never write "no learned method wins by >= 10 pp" without it | K2 | CE:L117, L118, L49 |
| Q3 | C-ens is a rule built first for the learned side and then given to the classical side (added in spike15d after spike15c showed B beating R0; the v9 card was written after C-ens was frozen, before TEST-D) | K2 and K3 | CE:L49, L170 |
| Q4 | RL gate results are single seed (spike15b gates single shot; spike13a gate passed only marginally, +10.2 vs +10 pp, best of 8 runs); the claim is "did not learn under this recipe", not "cannot learn" | K1 and Limitations | CE:L25, L237, L239 |
| Q5 | N = 3 of 8 settings (S5, S6, S7 PPO) had a learner that did not learn, so they cannot show classical is better | K2 | CE:L49 |
| Q6 | "Plan selection" reading of the 0.59 / 0.90 fractions is an interpretation; variance reduction by averaging over nearby plants is not ruled out | K3 | skeleton K3, Section 1 check note; CE:L116 |
| Q7 | Pre-specified (cards committed first) vs post hoc separated; H13b and H16b were not supported; H16b card rule inconsistency (95% CI vs one-sided p) disclosed | K3 (one clause) | CE:L118, L122 |
| Q8 | No unqualified "fair" or "equal" in title or text: say "audited protocol with disclosed budgets" | title, K1 | skeleton Section 0 |
| Q9 | K4 result is post hoc best-student choice, NEW seeds only, below the bar, not an equivalence result | K4 | CE:L79 |

## 3. Draft text (numbers are copied; each carries a CE line)

**Title.** An Audited, Budget-Disclosed Benchmark of Classical Teachers and Learned Students on a Simulated Arm

**Abstract / Intro (merged).**
> Learning-based controllers are often reported to beat classical ones, yet comparisons rarely give the classical side extended tooling, a disclosed budget, and rules fixed before testing. We ask what extra ingredient a learned student needs to beat its classical teacher, and answer under one protocol, in one simulator (MuJoCo), mostly on one task (whip). We contribute (1) an audited comparison protocol, (2) a cross-setting null with one whip qualification, and (3) a locked-set decomposition of where the whip headroom lies.

**2 Protocol (K1).**
> Classical tooling is extended on the same footing as the learner's. Each method has a three-axis budget (episodes, wall-clock, labels) and over-runs are disclosed. Rules are pre-specified in time-stamped, version-controlled hypothesis cards; git order was verified for only some spikes, and a public deviation table lists the rest (`prespec_audit.md`). Seed sets are disjoint, and every arena evaluation on locked sets is logged (111 lines) [CE:L23]. A sanity gate checks that the RL plumbing can learn a trivially representable correction; it passed only marginally (+10.2 pp vs the +10 pp threshold, best of 8 runs, n=500 instead of n=200) [CE:L25, L237], and later gates are single-seed shots [CE:L239]. The gate also taught us that a default exploration scale made PPO unable to learn a learnable correction.

**3 Cross-setting result (K2).**
> In 8 enumerated tracking, juggling and whip settings, no learned method beat the strongest classical baseline by the pre-specified +10 pp bar with a CI excluding 0 [CE:L49]. Learned methods beat only weaker baselines (an untuned teacher, or R0 on whip). In 3 of the 8 settings (S5, S6, S7 PPO) the learner did not learn, so these cannot show classical is better [CE:L49]; with a single-seed-gated exploration scale PPO degraded the base by 17.6-23.8 pp where tested [CE:L88, L89]. Whip is the exception to read carefully: against R0 alone the learned method clears the bar (+16.2 pp [11.0, 21.4]), but a classical ensemble, C-ens, also beats R0 (+10.6 [5.8, 15.4]) [CE:L49]. C-ens was added after the learned method had beaten R0, using a rule built first for the learned side; the card for the final test was written after C-ens was frozen [CE:L49, L170]. Against C-ens the learned side is ahead by +5.6 pp [0.6, 10.6], and an advantage of 10 pp or more cannot be excluded [CE:L117, L118].

**4 Where the whip headroom is (K3).**
> On realistic whip, locked sets (n=500) give three steps. True parameters add +17.6 pp [12.6, 22.6] over estimated ones (TEST-C) [CE:L108]; the probe cost is small but not zero, +4.4 [0.8, 7.8], and its pre-specified bound was not met [CE:L109]. Most of the +17.6 gap is recovered without better parameters: scoring candidate swings across several plausible plants recovers 0.59 [0.36, 0.82] of it with a classical Laplace ensemble and 0.90 [0.68, 1.16] with estimators trained on simulator labels (TEST-D) [CE:L116]. We read this as plan selection mattering more than policy learning; averaging over nearby plants as variance reduction is an untested alternative. Both bounds we pre-specified for the probe cost and for a 10 pp learned advantage failed (H13b, H16b), and the learned estimator alone, without robust selection, gives only +4.4 pp [-1.4, 10.0] [CE:L118, L119].

**5 K4 (one sentence).**
> For a multimodal teacher, the best generative student added +5.0 pp [-0.3, 10.2], below the bar, chosen post hoc on fresh seeds [CE:L79].

**6 Limitations.**
> One simulator and one dominant task: mechanistic claims rest on whip alone. RL results mean "did not learn under this recipe", with single-seed gates. Noise and friction magnitudes were set by us, not derived from hardware. Budget ratios were unequal, Holm correction applies within a hypothesis only, and the 10 pp upper end remains open.

(The "budget ratios unequal / Holm within hypothesis" clause mirrors skeleton Section 5 item (5); the CE row for budget ratios is skeleton Table 1 / CE Section 7 and no new number is stated.)

## 4. Word count of the draft above

Draft body: 589 words (Intro 80, Protocol 125, K2 170, K3 136, K4 23, Limitations 55). Command: `awk '/^> /{sub(/^> /,""); n+=split($0,a," ")} END{print n}' docs/paper/short_2p.md`. The budget table in Section 1 uses these counts; recount after any edit.

## 5. Cut order if over budget
1. K4 sentence (move to a footnote or drop).
2. Table 1 to 3 rows.
3. K1 sentences on over-runs and gate lesson (keep Q4 qualifier).
4. Never cut Q1-Q5 (Q2, Q3 are the review R2-M1 must-fixes).

## 6. Open items before use
- Check real CFP: page limit, reference treatment, template, anonymity (see `anonymization.md`), whether an abstract is separate.
- If the venue wants a "position" framing, the title candidate 2 in `skeleton.md` Section 0 slightly over-promises; keep title 1.
- If the main draft numbers change (e.g. a new locked-set run), regenerate this file from `skeleton.md`; CE line numbers drift whenever `claims_evidence.md` is edited (verify with grep before submitting).
