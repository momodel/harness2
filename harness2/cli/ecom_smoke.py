"""Run a short real MerchantBench episode with the upstream rule-based agent.

No model credentials or Docker required. This validates the public simulator,
agent protocol, settlement and native scoring, NOT model-backed harness quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .. import config
from ..benchmarks.ecommerce import RUNTIME, EcomAdapter
from ..benchmarks.merchantbench_runtime import LocalEpisode, run_process


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("runs/ecom-smoke.json"))
    args = parser.parse_args(argv)
    if args.days < 1 or args.days > 365 or args.master_seed < 0 or args.timeout < 1:
        parser.error("days must be 1..365, seed nonnegative and timeout positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    try:
        EcomAdapter.check_environment()
        with tempfile.TemporaryDirectory(prefix="harness2-ecom-smoke-") as temp:
            root = Path(temp)
            task = {
                "id": "smoke/rule-based",
                "master_seed": args.master_seed,
                "horizon_steps": args.days * 24,
            }
            with LocalEpisode(
                config.merchantbench_repo(), config.MERCHANTBENCH_PYTHON, root / "server", task
            ) as episode:
                episode.start()
                with (root / "agent.log").open("w") as log:
                    run_process(
                        [
                            config.MERCHANTBENCH_PYTHON,
                            config.merchantbench_repo() / "agent/baselines/rule_based.py",
                            "--selection-mode",
                            "random",
                            "--selection-seed",
                            str(args.master_seed),
                            "--max-steps",
                            str(args.days * 24 + 1),
                            "--quiet",
                        ],
                        cwd=root,
                        env=episode.agent_environment(),
                        stdout=log,
                        stderr=log,
                        timeout=args.timeout,
                    )
                record = episode.finish(timeout=args.timeout)
                raw = root / "episode.json"
                raw.write_text(json.dumps(record))
                native = json.loads(
                    subprocess.check_output(
                        [
                            str(config.MERCHANTBENCH_PYTHON),
                            str(RUNTIME),
                            "score",
                            "--repo",
                            str(config.merchantbench_repo()),
                            "--input",
                            str(raw),
                        ],
                        text=True,
                        timeout=30,
                    )
                )
            result = {
                "dataset": "public_synthetic",
                "policy": "upstream-rule-based-random",
                "upstream_pin": config.DEPENDENCY_PINS["merchantbench"],
                "master_seed": args.master_seed,
                "horizon_steps": args.days * 24,
                "terminal_status": record["terminal_status"],
                **native,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            print(json.dumps(result, indent=2))
            print(
                f"Saved {args.output}; synthetic rule-based smoke (may terminate early), not a paper reproduction."
            )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"smoke failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
