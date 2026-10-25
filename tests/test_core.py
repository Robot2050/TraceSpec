from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from agentspec.config import context_bucket
from agentspec.datasets import load_traj_tasks
from agentspec.metrics import EpisodeRecord, GenerationRecord, aggregate


class DatasetTests(unittest.TestCase):
    def test_compressed_replay_data(self):
        row = {
            "sample_id": "state-1",
            "split": "test",
            "mode": "browse",
            "phase": "act",
            "length_bucket": "2k",
            "prefix_tokens_est": 2000,
            "num_tool_rounds": 1,
            "source": {},
            "input": [{"role": "user", "content": "Inspect the evidence."}],
            "output": {"role": "assistant", "content": "Next action."},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "tracespec-traj"
            data_dir.mkdir()
            with gzip.open(data_dir / "test.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            tasks = load_traj_tasks(str(root))

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "state-1")
        self.assertEqual(tasks[0].meta["phase"], "act")

    def test_context_bucket_boundaries(self):
        self.assertEqual(context_bucket(0), "0-2k")
        self.assertEqual(context_bucket(2048), "2-4k")
        self.assertEqual(context_bucket(32768), "32k+")


class MetricTests(unittest.TestCase):
    @staticmethod
    def _episode(episode_id, *, accepted, rounds, proposed, output, seconds):
        generation = GenerationRecord(
            episode_id=episode_id,
            step=0,
            mode="act",
            num_input_tokens=100,
            context_bucket="0-2k",
            num_output_tokens=output,
            verify_count=rounds,
            accepted_sum=accepted,
            proposal_sum=proposed,
            gen_time_s=seconds,
            backend="spec",
        )
        return EpisodeRecord(
            episode_id=episode_id,
            dataset="tracespec-traj",
            task_id=episode_id,
            finish_reason="replayed",
            num_steps=0,
            total_time_s=seconds,
            model_time_s=seconds,
            tool_time_s=0.0,
            truncations=0,
            generations=[generation],
        )

    def test_global_counter_ratios(self):
        records = [
            self._episode("a", accepted=8, rounds=2, proposed=12, output=10, seconds=2.0),
            self._episode("b", accepted=2, rounds=1, proposed=6, output=5, seconds=1.0),
        ]
        summary = aggregate(records)
        self.assertAlmostEqual(summary["overall"]["acceptance_length"], 10 / 3)
        self.assertAlmostEqual(summary["overall"]["verify_rate"], 10 / 21)
        self.assertAlmostEqual(summary["latency"]["overall_tokens_per_s"], 5.0)


if __name__ == "__main__":
    unittest.main()
