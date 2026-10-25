<h1 align=center>TraceSpec</h1>
<p align=center><strong>Trajectory-Level Evaluation for Speculative Decoding in LLM Agents</strong></p>
<p align=center>Dynamic execution and fixed-prefix replay with unified acceptance, throughput, and episode-cost measurements</p>

## 🔎 Why TraceSpec

Speculative decoding is usually evaluated on isolated prompts. Agent serving presents a different workload: every action extends the history, tool feedback changes the next prediction, and calls within one task can have very different contexts and purposes. A single average accepted length cannot show where a draft succeeds or how acceptance affects task completion.

TraceSpec evaluates speculative decoding at the trajectory level. It combines dynamic interaction with controlled replay, then resolves the measurements by context length, execution phase, workload, and task progress.

* **For inference and serving researchers:** Measure both draft agreement and the cost of processing complete agent histories.
* **For speculative decoding researchers:** Compare methods on identical decision states and inspect where their context and phase profiles diverge.
* **For agent system developers:** Connect per-call decoding behavior with model time, tool time, and end-to-end episode latency.

## ✨ Core Contributions

1. **Dynamic agent evaluation.**
   TraceSpec runs multi-step episodes on BrowseComp-Plus and SWE-bench Verified. Browsing agents search and read benchmark documents. Repository agents inspect real checkouts, while edit and test operations receive simulated feedback.

2. **Controlled replay with TraceSpec-Traj.**
   The included test set contains 2,500 fixed prefixes across Web research and software engineering, four execution phases, and five nominal context strata. Every method receives the same saved history and generates one assistant continuation.

3. **Diagnostics beyond aggregate averages.**
   Reports resolve context length and episode progress jointly, with separate views for planning, action, reflection, and answer generation. This makes it possible to distinguish a change in agreement from a change in workload composition.

4. **A shared comparison protocol.**
   DFlash, EAGLE-3, DSpark, DLite, and LongSpark use common trajectory records and metric definitions. An autoregressive target baseline and a deterministic mock backend are also included.

## 📊 What the Evaluation Reveals

* **DSpark leads the overall comparisons.** It has the highest overall accepted length and verification yield in all 18 workload, target size, and temperature combinations. All 72 paired accepted-length comparisons have positive lower bounds on their pointwise 95% intervals.
* **Higher acceptance does not translate one to one into faster tasks.** Across six greedy dynamic configurations, DSpark improves accepted length over DFlash by 10.1% to 13.8%, while mean episode latency decreases by 2.7% to 7.1%.
* **Aggregate context curves can conceal conditional degradation.** For DSpark with Qwen3-8B at temperature 0, accepted length in matched late browsing actions falls from 6.194 at 8k to 16k tokens to 5.322 at 16k to 32k tokens. The change is -0.872 with a pointwise 95% interval of [-1.205, -0.473].
* **Context behavior depends on the workload.** In DSpark replay with Qwen3-8B at temperature 0, Web research acceptance falls from 5.242 below 2k tokens to 3.442 at 16k to 32k. Software engineering acceptance instead rises from 3.073 to 4.321 across the same intervals.
* **Sampling temperature changes draft agreement.** Increasing temperature from 0 to 1 lowers accepted length in all 45 matched comparisons.

## 🧪 Evaluation at a Glance

| Protocol | Workload | Evaluation units | Purpose |
|---|---|---:|---|
| Dynamic execution | BrowseComp-Plus | 100 tasks | Measure retrieval episodes as histories evolve |
| Dynamic execution | SWE-bench Verified | 100 tasks | Measure repository episodes with tool feedback |
| Fixed-prefix replay | TraceSpec-Traj | 2,500 test states | Compare methods on identical decision inputs |

The full study covers 90 configurations: three workloads, five speculative methods, Qwen3-4B, Qwen3-8B, and Qwen3-14B targets, and temperatures 0 and 1. Dynamic episodes allow 50 action steps and 32,768 context tokens. Replay uses a 36,864-token context limit. Both protocols allow up to 4,096 new tokens per model call.

Native PyTorch throughput comparisons cover DFlash, EAGLE-3, DSpark, and DLite. LongSpark uses SGLang and is included in acceptance comparisons. Its runtime measurements should not be compared directly with native PyTorch measurements without controlling for the serving stack.

## 🛠️ Quick Start

### Installation

Use Python 3.10 or newer. Run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation compatible with the target hardware and checkpoints. Model weights are not bundled. The command-line interface accepts local model paths or Hugging Face identifiers.

### Data Preparation

* **TraceSpec-Traj:** The compressed test set is included at `datasets/tracespec-traj/test.jsonl.gz` and needs no additional preparation.
* **BrowseComp-Plus:** Place the document collection at `datasets/browsecomp-plus/corpus/documents.jsonl`. The SQLite FTS5 index is built on first use and cached under the dataset root.
* **SWE-bench Verified:** Place full-history repository clones under `datasets/swe-bench-verified/repos/`. Directory names replace the slash in each repository identifier with an underscore. A preflight check verifies the selected base commits before model loading.

Upstream benchmark data, repository snapshots, and model weights remain subject to their original terms.

### Pipeline Check Without a GPU

```bash
python run_agent_eval.py --dataset tracespec-traj --backend mock \
  --max-samples 2 --max-new-tokens 32 \
  --output-dir outputs/smoke
```

The mock backend checks data loading, replay, record writing, aggregation, and report generation. Its metrics have no physical interpretation.

### Fixed-Prefix Replay

```bash
CUDA_VISIBLE_DEVICES=0 python run_agent_eval.py \
  --dataset tracespec-traj --backend spec \
  --target /models/Qwen3-8B \
  --draft /models/dspark_qwen3_8b_block7 \
  --max-context-tokens 36864
```

The default replay run uses all 2,500 test states. Use `--traj-phase`, `--traj-bucket`, and `--traj-mode` to select phases, nominal context buckets, and workloads. The default task-sampling seed is 980406.

### Dynamic Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run_agent_eval.py \
  --dataset browsecomp-plus --backend spec \
  --target /models/Qwen3-8B \
  --draft /models/dspark_qwen3_8b_block7 \
  --max-samples 100 --max-steps 50
```

Use `--dataset swe-bench-verified` for repository episodes. The backend detects the native draft architecture from its checkpoint configuration.

### Batch Evaluation

Run one native dynamic job with the batch launcher:

```bash
MODEL_ROOT=/models DRAFT_ROOT=/models \
GPUS=0 METHODS=dspark TARGETS=Qwen3-8B DATASETS=browsecomp-plus \
bash eval.sh
```

Run one native replay job with the replay launcher:

```bash
MODEL_ROOT=/models DRAFT_ROOT=/models \
GPUS_SPEC=0 METHODS=dspark TARGETS=Qwen3-8B \
bash eval_traj.sh
```

Both scripts accept space-separated GPU, method, target, and dataset lists through environment variables. Native jobs use one GPU each. LongSpark replay uses one target GPU and one draft GPU per worker.

Before using LongSpark, extract `agentspec/vendor/sglang.tar.gz` into `agentspec/vendor/` to restore the required `sglang/` source tree.

LongSpark requires a separate environment because its PyTorch ABI requirements differ from the native evaluators. Install it with:

```bash
bash scripts/install_longspark_env.sh
```

Set `LONGSPARK_PYTHON_BIN` to the resulting interpreter when the replay method list includes LongSpark. Run `python run_agent_eval.py --help` for all model, decoding, filtering, and runtime options.

## 📏 Metrics and Reports

For call `i`, `A_i` is the number of emitted accepted tokens including the correction or bonus token, `V_i` is the number of verification rounds, `P_i` is the number of proposed draft tokens, `N_i` is the number of output tokens, and `t_i` is model-call time.

| Metric | Definition | Measurement scope |
|---|---|---|
| Accepted length | `sum(A_i) / sum(V_i)` | Emitted accepted tokens per verification round |
| Verification yield | `sum(A_i) / sum(P_i + V_i)` | Accepted output relative to proposal and verification positions |
| Generation throughput | `sum(N_i) / sum(t_i)` | Output tokens per second, including prefill and decoding |
| Episode latency | Full episode wall time | Complete interaction, including model calls and tool execution |

Ratios use summed counters rather than averages of per-request averages. The autoregressive baseline has accepted length 1 by definition.

Each run writes a directory under `outputs/` containing:

* `records/`: One JSONL shard per worker, with generation-level counters and episode-level outcomes.
* `summary.json`: Overall, context, context by progress, phase, step, and latency aggregates.
* `report.txt`: Formatted tables for interactive inspection.
* `config.json`: The complete command-line configuration for the run.

## 📁 Repository Structure

```text
TraceSpec/
├── run_agent_eval.py          # Evaluation entry point
├── eval.sh                    # Dynamic batch launcher
├── eval_traj.sh               # Fixed-prefix replay launcher
├── agentspec/
│   ├── agents/                # Dynamic agents and trajectory replay
│   ├── backends/              # Native, SGLang, autoregressive, and mock backends
│   ├── tools/                 # Repository and retrieval tools
│   ├── datasets.py            # Dataset loading and replay filters
│   ├── metrics.py             # Records and global aggregation
│   ├── report.py              # Report rendering
│   └── vendor/                # Vendored evaluator and runtime components
├── datasets/                  # Evaluation metadata and TraceSpec-Traj test states
├── scripts/                   # Runtime installation and backend smoke checks
└── tests/                     # Offline unit tests
```

## ⚠️ Scope and Limitations

* TraceSpec measures inference behavior, not task correctness. It does not score SWE-bench patches or BrowseComp answers.
* Dynamic repository episodes execute allow-listed read commands on real checkouts. Edit and test operations are simulated, so their observations do not represent actual repository changes or test outcomes.
* Native evaluators process one request at a time and prefill the complete history on every call. They do not provide persistent cross-call KV reuse or continuous batching.
* Native PyTorch and SGLang measurements reflect different serving stacks. Compare their runtime results only under an explicitly controlled protocol.

## 📄 License

The TraceSpec code is released under the [Apache License 2.0](LICENSE). Vendored components retain their original licenses and notices. See [NOTICE](NOTICE) for attribution and data terms.
