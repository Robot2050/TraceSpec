"""Render the aggregated metrics as PrettyTable reports."""
from __future__ import annotations

from prettytable import PrettyTable

from .config import AGENT_MODES, CONTEXT_BUCKETS, STEP_BUCKETS


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report(summary: dict, *, title: str = "") -> str:
    parts = []
    if title:
        parts.append(title)

    overall = summary["overall"]
    table = PrettyTable(["#episodes", "#gen calls", "accept_len", "verify_rate",
                         "avg out tok", "avg ctx tok", "max ctx tok"])
    table.add_row([
        summary["num_episodes"],
        overall["num_generations"],
        _fmt(overall["acceptance_length"]),
        _fmt(overall["verify_rate"]),
        _fmt(overall["mean_output_tokens"], 1),
        _fmt(overall["mean_input_tokens"], 0),
        overall["max_input_tokens"],
    ])
    parts.append("== overall ==\n" + table.get_string())

    table = PrettyTable(["context", "#calls", "accept_len", "avg out tok",
                         "avg gen time (s)", "tok/s"])
    for _, _, label in CONTEXT_BUCKETS:
        row = summary["by_context_bucket"].get(label, {})
        if not row or row.get("num_generations", 0) == 0:
            continue
        table.add_row([
            label,
            row["num_generations"],
            _fmt(row["acceptance_length"]),
            _fmt(row["mean_output_tokens"], 1),
            _fmt(row["mean_gen_time_s"], 2),
            _fmt(row["mean_tokens_per_s"], 1),
        ])
    parts.append("== accepted length by context length ==\n" + table.get_string())

    # Cells contain accepted length followed by verification-round support.
    matrix = summary.get("by_context_step", {})
    bands = [label for label, lo, _ in STEP_BUCKETS if lo >= 0]
    table = PrettyTable(["context \\ step"] + bands)
    for _, _, label in CONTEXT_BUCKETS:
        row_data = matrix.get(label)
        if not row_data:
            continue
        cells = []
        nonempty = False
        for band in bands:
            cell = row_data.get(band) or {}
            if not cell.get("num_generations", 0):
                cells.append("-")
                continue
            nonempty = True
            cells.append(
                f"{_fmt(cell['acceptance_length'])} ({cell['verify_rounds']})"
            )
        if nonempty:
            table.add_row([label] + cells)
    parts.append(
        "== accepted length by context × episode progress (plan excluded) ==\n"
        + table.get_string()
    )

    table = PrettyTable(["mode", "#calls", "accept_len", "avg ctx tok", "avg out tok"])
    for mode in AGENT_MODES:
        row = summary["by_mode"].get(mode, {})
        if not row or row.get("num_generations", 0) == 0:
            continue
        table.add_row([
            mode,
            row["num_generations"],
            _fmt(row["acceptance_length"]),
            _fmt(row["mean_input_tokens"], 0),
            _fmt(row["mean_output_tokens"], 1),
        ])
    parts.append("== accepted length by agent mode ==\n" + table.get_string())

    table = PrettyTable(["step", "#calls", "accept_len", "avg ctx tok", "avg out tok"])
    for label, _, _ in STEP_BUCKETS:
        row = summary["by_step"].get(label, {})
        if not row or row.get("num_generations", 0) == 0:
            continue
        table.add_row([
            label,
            row["num_generations"],
            _fmt(row["acceptance_length"]),
            _fmt(row["mean_input_tokens"], 0),
            _fmt(row["mean_output_tokens"], 1),
        ])
    parts.append("== accepted length by step ==\n" + table.get_string())

    latency = summary["latency"]
    table = PrettyTable(["metric", "value"])
    table.add_rows([
        ["mean episode time (s)", _fmt(latency["mean_episode_time_s"], 1)],
        ["median episode time (s)", _fmt(latency["median_episode_time_s"], 1)],
        ["mean model time (s)", _fmt(latency["mean_model_time_s"], 1)],
        ["mean tool time (s)", _fmt(latency["mean_tool_time_s"], 2)],
        ["mean steps / episode", _fmt(latency["mean_steps_per_episode"], 1)],
        ["mean replans / episode", _fmt(latency.get("mean_replans_per_episode", 0.0), 2)],
        ["overall decode tok/s", _fmt(latency["overall_tokens_per_s"], 1)],
        ["finish reasons", summary["finish_reasons"]],
    ])
    parts.append("== end-to-end latency ==\n" + table.get_string())
    return "\n\n".join(parts)
