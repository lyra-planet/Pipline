# Control Layer Contract

This contract defines the behavior before implementation. It is intentionally
independent of Aurora's model classes so that the state topology can be tested
without a GPU.

## Core Records

### RootContext

| Field | Meaning | Invariant |
| --- | --- | --- |
| `run_id` | Unique run identifier | Stable for all states in a run. |
| `raw_instruction` | Original user instruction | Never rewritten in place. |
| `root_artifact` | Original model-input artifact | Never replaced by an intermediate candidate. |
| `source_artifact` | Original benchmark source artifact | Retained even when the Actor sees a sampled input. |
| `metadata` | Protocol and input provenance | Must not contain protected target-side benchmark fields. |

### PrivateChecklist

The private checklist is an immutable source-only plan. The fixed MSRAMIE
Instructor is the local adapter-free base Qwen3-VL model; Aurora's Type-1 LoRA
is excluded because its released training target is a different, one-shot
editor-plan schema. The Analyzer emits only an ordered atomic instruction and
one visual Y/N question for each editable unit. It emits no source-description
field, since source descriptions are not schedulable. The controller assigns
ordered `r*` IDs and maps only earlier numeric dependencies into those IDs. A
unit begins with one canonical edit verb (`Add`, `Apply`, `Change`, `Decrease`,
`Increase`, `Keep`, `Move`, `Remove`, `Replace`, `Set`, or `Transform`) and cannot contain a
second edit verb. A requested position, visibility, or time condition of an
action/added/replaced object is normally part of that same unit. `Keep` is
reserved for a requested, independently checkable temporal or state condition
after a prior explicit edit, and must directly depend on exactly one earlier
non-`Keep` requirement. A `Keep` item has exactly one such predecessor, and the
controller must schedule it in the same batch whenever that predecessor is
selected and the `Keep` item remains unmet. Each explicit edit can own at most
one direct `Keep` item, so this coupling always fits the paper-default `IV=2`.
It cannot represent generic source preservation.
Generic unchanged-content clauses use separate `p*` IDs; they enter every
compiled Actor prompt but cannot be selected and do not consume Instruction
Volume.

The controller enumerates every dependency-valid, unmet batch with size at most
IV as `b1`, `b2`, and so on. The Instruction Generator does not return IDs,
reasoning, or an Actor prompt. It is constrained to emit exactly one JSON value,
for example `{"selected_batch_id":"b2"}`, from that finite menu. Before Actor
execution, the controller revalidates the chosen batch against the current Y/N
answers and deterministically compiles its atomic instructions plus preservation
clauses. No model-generated text can add an unselected edit.

### EditState

| Field | Meaning | Invariant |
| --- | --- | --- |
| `state_id` | State identifier | Unique and stable. |
| `parent_state_id` | Transition parent | Exactly one for non-root states; absent for root only. |
| `depth` | Distance from root | Equals parent depth plus one. |
| `input_artifact` | Actor input | Equals the parent output artifact. |
| `output_artifact` | Actor output | Immutable after Actor success. |
| `thought` | Current editor-facing instruction | Bounded by the configured instruction volume. |
| `evaluation` | Evaluator response | Has instruction, preservation, and quality values in `[0, 1]`. |
| `reference_state_ids` | GoR edges | All IDs precede this state and never include the state itself. |
| `scheduler_event` | Decision after evaluation | Records continue, prune, backtrack, completion, or selection. |

### Evaluation

The generic control layer has three mandatory dimensions:

| Field | Meaning | Scheduler role |
| --- | --- | --- |
| `instruction_following` | Fraction/score for the frozen private checklist | Primary ranking dimension. |
| `preservation` | Root-relative preservation score | Secondary ranking dimension. |
| `quality` | Visual or temporal quality score | Gate and tertiary tie-breaker. |

An Aurora evaluator may attach per-unit answers, temporal stability, rationales,
or other observations, but it cannot omit these three normalized values.

## Component Interfaces

| Component | Inputs | Output | Cannot do |
| --- | --- | --- | --- |
| `InstructionGenerator` | Root context, parent state, private checklist, reference summaries, controller-enumerated legal batches | One bounded thought compiled from one selected batch | Read protected CoVEBench annotations or emit an arbitrary Actor prompt. |
| `EditingActor` | Root context, parent state, thought, output location | Output artifact and generation metadata | Mutate an already-completed parent artifact. |
| `StateEvaluator` | Root context, state, private checklist | Normalized evaluation and per-unit result | Query official CoVEBench scores. |
| `SimilarityScorer` | Candidate state and parent state | Similarity in `[0, 1]` | Change state media or selection history. |
| `GraphRetriever` | Parent state and prior topology | Ordered reference summaries | Return future states or out-of-range states. |
| `DepthFirstScheduler` | Topology, evaluations, fixed configuration | Next action and final state | Exceed maximum state, child, or depth budgets. |

## Scheduler Semantics

The default paper-faithful scheduler uses the following order:

1. Add the candidate state and update the temporary best state lexicographically
   by `(instruction_following, preservation, quality, shallower_depth)`.
2. Stop when the state budget is exhausted.
3. Stop on completion only when the temporary best state meets all completion
   thresholds and minimum depth.
4. Continue from the current state only when it meets stay thresholds, is below
   maximum depth, does not degrade beyond tolerance against its parent, and is
   below the completion upper bound.
5. Otherwise, backtrack to the deepest earlier ancestor with remaining child
   capacity. If no ancestor is expandable, select the temporary best state.

State ranking is evidence-only: a state is not silently excluded because a later
branch failed a stay condition. Stay conditions control expansion and
backtracking; the ranking rule controls terminal selection.

The scheduler is depth-first: when the current state may continue, that child is
selected before an unexpanded shallower branch. A state can have at most
`max_children`; all non-root states count against `max_steps`.

## Graph-of-References Semantics

The paper retrieves states in a predefined topology search range, measures
visual similarity between a candidate reference state input and the parent state
output, discards scores below a threshold, and selects Top-K.

The core implementation therefore requires:

- a nonnegative `search_range`; zero disables retrieval;
- a positive `top_k`;
- a threshold in `[0, 1]`;
- a similarity scorer that returns `[0, 1]`;
- deterministic ordering: higher similarity, then shallower temporal creation
  order, then state ID;
- deduplication by state ID and explicit exclusion of root and parent unless an
  adapter requests root metadata separately.

The expected production scorer is a normalized blend of DISTS and LPIPS as
described by the paper. The core does not hard-code either image model; an
Aurora adapter supplies that scorer. Lightweight feature scorers are permitted
only for test/smoke use and must be recorded in run metadata.

## Persistence and Resume

`RunJournal` is append-only. A persisted run contains the root context,
configuration, private checklist, state records, reference edges, scheduler
events, and terminal selection. Resuming a run must:

- validate schema version and configuration fingerprint;
- preserve existing state IDs and output artifacts;
- never regenerate a completed state;
- reject a root-context mismatch;
- continue only from an action that is valid under the saved topology.

A planning response that exhausts its bounded validation retries is instead
recorded as a terminal `planning_failure` event. It contains rejected raw
responses and deterministic validation errors but creates no Actor state or
artifact. Resume must fail closed on that event.

The current protocol uses ControlConfig schema `2` and journal schema `3`.
Earlier journal/checklist formats are rejected rather than migrated or guessed.

## CoVEBench Information Allowlist

For a CoVEBench execution, component payloads may contain only:

- task ID;
- original source-video path;
- protocol-sampled model-input path and sampling metadata;
- raw `editing_instruction`;
- candidate state media and generation metadata;
- private plan/checklist generated from those inputs.

They must not contain `target_video_description`, `evaluation_groups`, question
answers, category hierarchy, official metric values, or any content derived
from them. The implementation will validate protected key names recursively at
runtime before invoking an Instructor, Actor, Evaluator, or selector.
