# experiment_logger.py

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: dict[str, Any],
    ):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.perf_counter()

        self.metrics_path = self.log_dir / "metrics.csv"
        self.config_path = self.log_dir / "config.json"
        self.notes_path = self.log_dir / "experiment_log.md"

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "wall_clock_sec",
                        "split",
                        "loss",
                        "lr",
                    ],
                )
                writer.writeheader()

        if not self.notes_path.exists():
            with open(self.notes_path, "w") as f:
                f.write(make_experiment_log_template(experiment_name, config))

    def log_metric(
        self,
        step: int,
        split: str,
        loss: float,
        lr: float,
    ):
        wall_clock_sec = time.perf_counter() - self.start_time

        row = {
            "step": step,
            "wall_clock_sec": wall_clock_sec,
            "split": split,
            "loss": loss,
            "lr": lr,
        }

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "wall_clock_sec",
                    "split",
                    "loss",
                    "lr",
                ],
            )
            writer.writerow(row)

        print(
            f"[{split}] "
            f"step={step} "
            f"time={wall_clock_sec:.1f}s "
            f"loss={loss:.4f} "
            f"lr={lr:.3e}"
        )


def make_experiment_log_template(
    experiment_name: str,
    config: dict[str, Any],
) -> str:
    config_lines = "\n".join(
        f"- `{k}`: `{v}`"
        for k, v in config.items()
    )

    return f"""# Experiment Log: {experiment_name}

## Goal

Describe what this experiment is trying to test.

## Configuration

{config_lines}

## Runs

| Run | Change | Train Loss | Val Loss | Notes |
|---|---|---:|---:|---|
| 1 | Baseline |  |  |  |

## Observations

Write observations here.

## Problems Encountered

- 

## Next Steps

- 
"""