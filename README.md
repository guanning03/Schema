<div align="center">

# Schema: Discovering Unknown Environments via Agentic Program Induction

</div>

This is the official implementation of our paper "<strong>Schema: Discovering Unknown Environments via Agentic Program Induction</strong>" by Guanning Zeng*, Jiani Wang, Wenjie Ma, Shaofeng Yin, Chenyang Wang, Shichen Liu, Angjoo Kanazawa, Wode Ni, Xiuyu Li*, Andrea Zanette*, and Haiwen Feng* (*project leads).

<div align="center">
<a href="https://guanning03.github.io/schema-webpage/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge"></a>
<a href="https://arxiv.org/abs/2609.39140">
    <img src="https://img.shields.io/badge/Paper-%23FF2442?style=for-the-badge"></a>
<a href="https://github.com/guanning03/Schema">
    <img src="https://img.shields.io/badge/Code-%2300B4D8?style=for-the-badge"></a>
<a href="https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces">
    <img src="https://img.shields.io/badge/Traces-%236C5CE7?style=for-the-badge"></a>
<a href="https://schema-harness.github.io/">
    <img src="https://img.shields.io/badge/Blog-%23F39C12?style=for-the-badge"></a>
</div>

## Overview

Schema is an agent harness that learns unfamiliar environments through *interactive program induction*. The agent writes its understanding of the environment as an executable program that predicts how the environment responds to each action, backtests it against the full interaction history, plans inside it (BFS / A*), and executes plans under step-by-step verification. With the same base models, Schema raises ARC-AGI-3 RHAE from 58.7% to 99.2%, solves all 21 public DiG-bench games, and reaches the median performance of the top-50 human players on MazeBench.

| Directory    | Benchmark | Entry point |
|--------------|-----------|-------------|
| `arc-agi-3/` | ARC-AGI-3 (25 public games bundled under `env/assets/`) | `python -m agent.world_model.solve` |
| `digbench/`  | DiG-bench: Schema (`schema/`) and a baseline harness (`basic_harness/`) | `run.sh` |
| `mazebench/` | MazeBench | `python -m agent.world_model.solve` |

## Installation

```
conda create -n arc3 python=3.12 && conda activate arc3
for d in arc-agi-3 mazebench digbench/schema digbench/basic_harness; do pip install -r $d/requirements.txt; done
```

Then install and log in to the coding CLI: `claude` for Claude Code (uses `CLAUDE_CONFIG_DIR`, else `~/.claude`), or `codex login` for Codex (writes `~/.codex/auth.json`).

## Running

Each run writes to a workdir (`--workdir`, default `tmp/` next to `agent/`) holding the agent's program (`world_model.py`), its notes and the event log (`events.jsonl`). Continue an interrupted run with `--resume <workdir>`; see `--help` for all options.

**ARC-AGI-3.** The games run offline, so no ARC API key is needed. Default: Claude Code, `claude-opus-4-8`, `--effort max`; use `--provider codex-cli --model <model> --reasoning <effort>` for Codex. `--jail` runs Claude Code and the agent's code in containers (podman by default, or `--jail-runtime enroot|docker`).

```
cd arc-agi-3
python -m agent.world_model.solve --game ls20
for g in $(ls env/assets); do sbatch run.sbatch --game $g; done   # all 25 games on Slurm
```

**DiG-bench.** Get a token at <https://digbench.ai/account/tokens>. `run.sh` runs Schema and the baseline side by side with GPT-6 Astra via Codex; logs go to `digbench/logs/`.

```
export DIGBENCH_API_TOKEN=...
cd digbench && ./run.sh medium P-1 P-2
```

**MazeBench.** Needs Linux, bubblewrap (`bwrap` on `PATH`), the MazeBench engine (see [mazebench.com](https://mazebench.com); set `MAZEBENCH_ENGINE_ROOT`, or place it at `vendor/MazeBenchEngine`) and Node.js (`MAZEBENCH_NODE_BIN` if `node` is not on `PATH`). The `run_python` / `run_shell` tools use bubblewrap; install it with `sudo apt-get install bubblewrap` on Debian/Ubuntu and ensure the host permits its user namespaces. The solver checks the platform and executable before starting a run. Default: Codex, `gpt-6-astra`, `--reasoning high`.

```
cd mazebench
python -m agent.world_model.solve --workdir runs/maze --max-hours 48 --service-tier ultrafast
sbatch run.sbatch runs/maze "2026-10-08 12:00" --pool 1    # <run_dir> <deadline> [solve args]
```

## Acknowledgements

We thank the [ARC Prize Foundation](https://arcprize.org) for ARC-AGI-3, the authors of [DiG-bench](https://digbench.ai) for the benchmark and the baseline harness in `digbench/basic_harness/` (MIT License), and the authors of [MazeBench](https://mazebench.com) for the benchmark and engine.

## License

This repository is released under the [MIT License](LICENSE). `digbench/basic_harness/` keeps its own MIT License.

## Citation

```
@misc{schema2026,
      title={Schema: Discovering Unknown Environments via Agentic Program Induction},
      author={Guanning Zeng and Jiani Wang and Wenjie Ma and Shaofeng Yin and Chenyang Wang and Shichen Liu and Angjoo Kanazawa and Wode Ni and Xiuyu Li and Andrea Zanette and Haiwen Feng},
      year={2026},
      eprint={2609.39140},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.39140},
}

@misc{schema2026blog,
      title={{[schema]: Frontier Models with the Right Harness Achieve $\sim$99\% on ARC-AGI-3 Public}},
      author={Zeng, Guanning and Wang, Jiani and Ma, Wenjie and Yin, Shaofeng and Wang, Chenyang and Liu, Shichen and Kanazawa, Angjoo and Ni, Wode and Li, Xiuyu and Zanette, Andrea and Feng, Haiwen},
      year={2026},
      howpublished={Impossible Research. \url{https://schema-harness.github.io/}},
      url={https://schema-harness.github.io/},
}
```
