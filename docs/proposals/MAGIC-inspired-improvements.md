# Proposal: MAGIC-flow inspired improvements to detail_router / placer_rl / constraints

Source of study: anaroute @ 9ef33c2, IdeaPlaceEx @ a727c34, ConstGen @ 6e1532d
(cloned read-only under /tmp/opencode/). Status: proposal, no code yet.

This document maps concrete mechanisms from the MAGIC flow to our codebase and
proposes a staged adoption plan. Items are ordered by (value / risk / effort).

---

## A. `align/pnr/detail_router/` — borrow anaroute's grid & access model

Our router is a **path-based SAT** router (`CandidatePath` enumeration +
cover/mutex/lazy-connectivity); anaroute is **A\* over a single global grid +
history rip-up**. Different solvers, but several of anaroute's *modelling* ideas
transfer cleanly and would have prevented most of the bugs we just fixed
(half-pitch via drift, off-pin access cells, boundary-pin collisions).

### A1. (HIGH, medium effort) Unified global routing grid

**Problem today.** Each layer has its own `pitch` (M1/M3 V=80, M2/M4 H=84),
and `track_index` rounds while `seg_index` floors — so the *same* physical
(x,y) lands on different cell centres on different layers. That mismatch is
the root cause of every `_junction_on_pin` / half-pitch via bug we chased:
via junctions, leg rebuilds, boundary-pin staggering, abutting-seg mutexes
all exist *because* layers don't share a coordinate system.

**anaroute's answer.** One `(gridStep, gridOffsetX, gridOffsetY)` shared by
all routing layers; `gridStep` must be a common multiple of the per-layer
pitches. Vias only allowed on grid points; wire tracks = the layer's own
pitch *subset* of the global grid.

**Proposal.** Introduce a `GlobalGrid` in `grid.py`:
- `step = lcm(metal pitches)` (for FinFET14nm_Mock: lcm(80,84) = 1680; with
  offset 0 this is fine — cells become sparse but correctness dominates).
- Per-layer `ti`/`si` become *global* indices; legality still per-layer
  (`allowed[layer]`) but junction coordinates are exact — `(ti,si)` on any
  layer maps to the same physical point, eliminating the two-pass leg
  rebuild and junction-vs-cell-centre hacks in `candidates.py`.
- Keep per-layer `direction`/`width`/`spacing` metadata; only the indexing
  unifies.

**Risk.** Moderate refactor of `grid.py` + `cell_center`/`track_index` call
sites; candidate count changes (fewer spurious tracks). Worth it — it deletes
an entire class of bugs rather than patching them.

### A2. (HIGH, small effort) Access points = "pin box ∩ grid points", OD-scored

**Today.** `pin_access_cells` samples pin rects (`_sample_rect_points`),
quantises to cells, then filters with `_touches`/`_junction_on_pin` — a chain
of heuristics that still lets a cell centre sit *just outside* the pin rect.

**anaroute's answer (`acsMgr.cpp:46-105`).** For each pin rect, take global
grid indices `ceil(xl)..floor(xh)` × `ceil(yl)..floor(yh)` — only points
strictly inside the box become candidates. Then, per escape direction, build
an extension rect and score by overlap area with active-device (OD) shapes;
keep zero-overlap directions, else the least-overlapping one.

**Proposal.** Replace `_sample_rect_points`+`_touches` with:
- `pin_access_cells`: enumerate global-grid points inside each pin rect
  (needs A1 or a per-layer equivalent), pick the *leg-compatible* layer
  subset directly so the endpoint via bottoms on pin metal.
- Score candidates by "escape rect ∩ obstacles/other-pin rects" so
  obviously-blocked endpoints are pruned before path enumeration.

**Risk.** Low; the function signature already takes `pin_rects`.

### A3. (MEDIUM, small effort) Symmetric-net routing = route one side, mirror the other

**anaroute's answer (`drGridAstar`).** Sym nets decomposed into a symmetric
core + remainder; only one side is searched, then `flipX` mirrors the whole
wire/via set into the partner net. Self-sym nets get a dummy on-axis pin.

**Proposal.** For `SymmetricNets`/`SymmetricBlocks` constraints: solve one
side's paths through the normal SAT model, then emit the mirrored rects
(with `symX = 2*axis − x`) into the partner net's terminals. Add
`drSymmetry`-style graceful degradation: if the mirrored candidate set is
infeasible after N attempts, relax the symmetry assumption rather than
UNSAT.

**Risk.** Needs `symAxis` from placer + constraint plumbing; mirror legality
must be re-checked against obstacles. Defer until A1 (mirror arithmetic is
exact on a unified grid).

### A4. (LOW) Per-layer spatial index for mutex generation

**Today.** `build_mutexes` compares piece bboxes pairwise per layer
(O(n²) after sort-sweep). Fine at our scale (1677 paths → 38k mutexes), but
anaroute's incremental `SpatialMap` (rect `expand(spacing-1)` query) is the
right pattern if regions grow or we move to rip-up/reroute.

**Proposal.** Only if profiling shows mutex generation >10% of runtime.

### What NOT to adopt from anaroute
- Its A*/history-rip-up solver itself — our SAT formulation already solves
  the same problem more globally; keep it.
- Track assignment: doesn't exist at that commit (empty stub).
- Shielding: not implemented there either; out of scope.

---

## B. `align/pnr/placer_rl/` — borrow IdeaPlaceEx's objective engineering

Our placer is `HPWL + sqrt(RUDY)` with SA + RL, symmetry via penalty +
legalization. IdeaPlaceEx is pure analytical NLP (LSE + Adam + subgradient
penalty weights) — we won't adopt the solver, but three *modelling* ideas
transfer.

### B1. (HIGH, small effort) Adaptive penalty weights

**Today.** Symmetry/violation penalties are hand-tuned constants.

**IdeaPlaceEx (`nlpOuterOptm.hpp`).** Initialise each penalty weight by
matching gradient norms to the HPWL term, then update multipliers by
subgradient ascent `λ += step × raw_violation`; record *raw* (unweighted)
sub-objectives separately for the stop criterion.

**Proposal.** In `sa_solver`/`cost_model`:
- Normalise each energy sub-term at init (scale so its initial contribution
  ≈ HPWL's), then adapt `λ` proportional to the *violation magnitude*
  each temperature step.
- Track raw sub-terms for a violation-driven early-stop (e.g. overlap ≤1%
  area, asymmetry ≤ 0.05·√area) instead of fixed `--placer_sa_iterations`.

**Risk.** Low; purely inside `cost_model.py`/`sa_solver.py`.

### B2. (MEDIUM) Symmetry axis as an explicit variable, paired legalisation

**Today.** SymmetricBlocks enforced by energy penalty + `_legalize_symmetry`
post-pass; cells snap to grid independently, then we re-impose symmetry.

**IdeaPlaceEx.** Axis `a` is a decision variable (`2·numCells + numGroups`
placement vector); GP uses soft penalty `(x_i+x_j+w−2a)²+(y_i−y_j)²`,
legalisation uses LP hard equality, and grid-snap moves **symmetric pairs
together** (`GridAligner::adjustSymPair`).

**Proposal.** Add `axis` per `SymmetricBlocks` group to the SA/RL state;
energy term includes it (gradient flows to `a`); legalisation snaps pairs
jointly (snap one, derive partner `x_j = 2a − x_i − w`) rather than
independently. Keeps direction/ratio flexibility our `SymmetricBlocks`
already supports — do *not* copy IdeaPlaceEx's "all groups share one global
axis" simplification (`MULTI_SYM_GROUP` is off there for a reason).

### B3. (LOW-MEDIUM) Virtual-pin boundary escape in the energy function

**Today.** HPWL only counts internal pins; routability to die edge is only
modelled at the router via our synthetic boundary pins.

**IdeaPlaceEx (`VirtualPinAssigner`).** Sprinkle virtual pins along a
virtual boundary (extension ~200, interval ~400, mirrored L/R), assign each
IO net to a pin via min-cost max-flow each outer iteration, and feed that
escape length into HPWL.

**Proposal.** Optional term: for each top-level net, add
`escape_len = Manhattan(pin → nearest boundary virtual pin)` to the energy;
re-assign pins cheaply (greedy or small MCMF) each K iterations. Cheap
routability feedback that complements RUDY.

### What NOT to adopt from IdeaPlaceEx
- The analytical NLP solver itself — our SA/RL is a deliberate choice for
  this codebase (and IdeaPlaceEx has *no* congestion model; our `sqrt(RUDY)`
  is strictly stronger there).

---

## C. `align/compiler/` + `align/schema/constraint.py` — borrow ConstGen's detection

ALIGN's `find_constraint.py` seeds symmetry from port-pair combinations.
ConstGen seeds from **functional motifs** and walks the product graph — more
semantic, more testable.

### C1. (HIGH, medium effort) Motif-driven symmetric-pair detection (new channel)

**ConstGen (`SymDetect.cpp`, `Pattern.cpp`).** Seeds = `DIFF_SOURCE` /
`CROSS_LOAD` device pairs sharing a source net; expands pairwise DFS where
both sides must arrive via the *same* pin type, forbidden to expand through
gate (prevents merging two diff-pairs); terminates on shared-net PASSIVE /
new DIFF_SOURCE / LOAD. Each motif (`Pattern::pattern`) is one testable
predicate with a hard `matchedType && matchedSize` gate.

**Proposal.** Add `find_constraint` pass that:
1. Classifies each MOS by connection (`S==D→CAP`, `G==D→DIODE`, `G==S→DUMMY`,
   else `DIFF` — trivial from netlist).
2. Seeds diff-pair / cross-coupled pairs on signal nets, runs pairwise DFS
   with ConstGen's accept/terminate predicates.
3. **Vote** with the existing port-seeded results: emit
   `SymmetricBlocks` only for pairs found by both channels (or flag the
   single-channel ones as lower-confidence).

**Risk.** Medium — needs the motif classifier + DFS; but it's additive
(doesn't replace the existing pass) and directly testable per motif.

### C2. (HIGH, small effort) Self-symmetric device expansion

**Today.** `find_constraint.py:443` — "Skipping self symmetry for single
device"; tail-current/bias devices that should sit on the axis get missed.

**ConstGen (`selfSymSrch`).** From each diff-pair, walk the drain-connected
`DIFF` devices along the source net and mark them self-symmetric.

**Proposal.** Same one-line-of-thought expansion: after a pair is found,
emit `SymmetricBlocks.pairs=[["MNx"]]` for drain-chain devices on the shared
source net.

### C3. (MEDIUM) Bias / current-mirror group identification

**ConstGen (`Bias.cpp`).** Common-gate net + a `DIODE`-connected driver +
same-`InstType` filter ⇒ `driver ↔ each same-type load` star pairs (BIAS).

**Proposal.** Detect diode-connected devices and their shared-gate same-type
loads; emit `SymmetricBlocks` pairs. **Also implement ratio pairing** —
ConstGen's `matchedSize` requires exact W/L/nf equality so 1:N mirrors are
missed; our `pairs_validator` (`constraint.py:996`) currently also forces
parameter equality and would need relaxing to "proportional" to accept
these.

### C4. (MEDIUM, small effort) Feasibility-aware conflict resolution

**Today.** `add_or_revert_const` swallows conflicts with `except: pass` —
dropped constraints are invisible.

**ConstGen's lesson.** Serialise-time global `one-pair-per-instance`
dedup (learned the hard way: commits 7332c33/47f9aa0 broke `existPair`
because conflicting constraints made the downstream ILP infeasible).

**Proposal.** Build an explicit instance→pair occupancy table during
generation; on conflict resolve deterministically (user constraint >
auto > larger group) and **log the loser + reason** (as a `Generator`
-marked constraint or warning) instead of silently dropping.

### C5. (LOW) Post-place symmetry linter

**MAGIC chain.** Placer `Database::checkSym()` warns when pair midpoints
aren't co-axial / self-syms aren't centred; router reports
degree-of-symmetry.

**Proposal.** After `detail_route_module`/`placer` finishes, run a checker
that verifies each `SymmetricBlocks`/`SymmetricNets` against the emitted
geometry; report violations as warnings (and optionally feed back as a
confidence score on auto-generated constraints).

---

## D. Suggested order of implementation

| Phase | Item | Why first |
|---|---|---|
| 1 | **A1** unified grid | Deletes the bug class we just fought; precondition for A3 |
| 1 | **B1** adaptive weights | Small, self-contained, immediate robustness gain |
| 1 | **C2** self-sym expansion | Tiny; fills a known coverage gap |
| 2 | **A2** box∩grid access points | Builds on A1; simplifies candidate gen |
| 2 | **C1** motif detection | Medium effort, big constraint-coverage win |
| 2 | **C4** conflict-resolution logging | Small; improves debuggability |
| 3 | **B2** axis-as-variable | Needs care interacting with RL state |
| 3 | **A3** symmetric mirror routing | Needs A1 + constraint plumbing |
| 3 | **B3** virtual-pin escape | Optional routability feedback |
| — | A4, C3(ratio), C5 | Opportunistic / as-needed |

---

## E. Validation plan

- A1/A2: rerun `five_transistor_ota` + `telescopic_ota` e2e; `validate()`
  must stay `ok:True` *without* the junction/track workarounds (delete them).
- B1: SA energy convergence on the same examples; check symmetry violations
  in `__placer_dump__.json` go down with adaptive vs fixed weights.
- C1/C2: unit tests per motif (diff pair, cross-coupled load, bias chain)
  on `examples/`; verify generated `SymmetricBlocks` match hand-written
  `.const.json` where present.
- Always: `pytest tests/detail_router/ tests/placer_rl/` (add constraint
  tests under `tests/compiler/`).
