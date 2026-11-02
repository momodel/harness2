"""External fine attribution and frozen paired evaluation without paid APIs."""
import argparse
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks.ecom_analysis import episode_audit, paired_summary
from harness2.benchmarks.ecommerce import EcomAdapter
from harness2.cli import ecom_pair


def episode(root, *, seed=100, profit=10):
    (root / "measurement").mkdir(parents=True)
    db = root / "server-state/runs/native/state.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE events(run_id,t,event_type,entity_id,agent_id,payload)")
        c.execute("CREATE TABLE orders(run_id,agent_id,total_penalty,current_status,realized_cost,realized_revenue)")
        for t, kind, penalty, reason in [(8, "order_stockout_violation", 10, "supplier_delisted"),
                                        (18, "order_insufficient_balance_violation", 50, None),
                                        (95, "order_settled_bad_review", 20, None),
                                        (171, "order_settled_refund", 8, None)]:
            c.execute("INSERT INTO events VALUES(?,?,?,?,?,?)",
                      ("native", t, kind, "sku", "agent_0", json.dumps({"penalty": penalty, "reason": reason})))
        c.execute("INSERT INTO orders VALUES('native','agent_0',88,'settled_normal',0,0)")
        c.execute("INSERT INTO orders VALUES('native','agent_0',0,'settled_only_refund',38.63,0)")
    record = {"terminal_status": "finished", "run_id": "native", "master_seed": seed,
              "horizon_steps": 24, "merchant_section": {"agent_id": "agent_0", "is_alive": True,
              "series": {"net_assets": [[214, 3000 + profit]], "cum_net_profit": [[214, profit]],
                         "cum_fine": [[214, 88]]}}}
    (root / "measurement/episode.json").write_text(json.dumps(record))
    return root


class PairedTests(unittest.TestCase):
    def test_fines_reconcile_and_only_refund_is_not_fine(self):
        with tempfile.TemporaryDirectory() as temp:
            d = episode_audit(episode(Path(temp)))
            self.assertEqual(d["fine_breakdown"], {"supplier_delisted": 10, "insufficient_balance": 50,
                                                 "bad_review": 20, "refund": 8})
            self.assertEqual(d["cumulative_fine"], 88)
            self.assertEqual(d["refund_only_cost_loss"], 38.63)
            self.assertEqual([e["phase"] for e in d["fine_events"]], ["operating", "operating", "drain", "drain"])

    def test_changed_attribution_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = episode(Path(temp))
            with sqlite3.connect(root / "server-state/runs/native/state.db") as c:
                c.execute("DELETE FROM events WHERE t=18")
            with self.assertRaisesRegex(ValueError, "reconciliation failed"):
                episode_audit(root)

    def test_statistics_exclude_incomplete_pairs_and_cluster_seeds(self):
        rows = []
        for seed in (100, 101):
            for repeat in range(3):
                for arm in ("base", "agg"):
                    rows.append({"master_seed": seed, "repeat": repeat, "arm": arm, "status": "ok",
                                 "metrics": {"net_profit": 10 + (5 if arm == "agg" else 0),
                                             "cumulative_fine": 88, "is_alive": True,
                                             "fine_breakdown": {"refund": 8}}})
        rows.append({"master_seed": 102, "repeat": 0, "arm": "agg", "status": "failed"})
        d = paired_summary(rows, planned_pairs=9)
        self.assertEqual(d["complete_pairs"], 6)
        self.assertEqual(d["technical_failures"]["agg"], 1)
        self.assertEqual(d["not_run_cells"], 5)
        self.assertEqual(d["arms"]["base"]["mean_net_profit"], 10)
        self.assertEqual(d["paired"]["net_profit"]["seed_count"], 2)
        self.assertEqual(d["paired"]["net_profit"]["seed_bootstrap_95ci"], [5, 5])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_summary(rows + [rows[0]], planned_pairs=9)

    def test_plan_freezes_harnesses_and_bounded_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "ENV_MODE", "legacy"):
            root = Path(temp) / "paired"
            args = argparse.Namespace(output=root, seeds=10, seed_start=100, repeats=3, horizon=24,
                                      base=None, agg=config.ROOT / "examples/ecom-budget-harness/oc")
            plan = ecom_pair.prepare(args)
            self.assertEqual(plan["planned_rollouts"], 60)
            self.assertFalse(plan["library_updates"])
            self.assertEqual(len(list(ecom_pair.cells(plan))), 60)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                ecom_pair.prepare(args)
            calls = []

            def execute(adapter, task_id, harness, run_id):
                calls.append(run_id)
                return episode(root / "fake-episodes" / run_id.replace("/", "__"))

            run_args = argparse.Namespace(output=root, limit_rollouts=2, retry_failed=False)
            with (patch.object(EcomAdapter, "preflight"),
                  patch.object(EcomAdapter, "execute", execute),
                  patch.object(EcomAdapter, "score", return_value={"score": 3010}),
                  patch.object(EcomAdapter, "artifacts_intact", return_value=True), redirect_stdout(io.StringIO())):
                self.assertEqual(ecom_pair.run(run_args)["attempted"], 2)
                self.assertEqual(ecom_pair.run(run_args)["attempted"], 2)
                report = ecom_pair.report(root)
            self.assertEqual(len(set(calls)), 4)
            self.assertEqual(report["complete_pairs"], 2)
            self.assertEqual(report["not_run_cells"], 56)
            with patch.object(config, "SOLVER_MODEL_OC", "changed/model"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    ecom_pair.validate(root)
            (root / "agg/systemprompt.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen agg"):
                ecom_pair.validate(root)

    def test_failed_cell_stops_run_and_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "ENV_MODE", "legacy"):
            root = Path(temp) / "paired"
            ecom_pair.prepare(argparse.Namespace(output=root, seeds=10, seed_start=100, repeats=3,
                horizon=24, base=None, agg=config.ROOT / "examples/ecom-budget-harness/oc"))
            with (patch.object(EcomAdapter, "preflight"),
                  patch.object(EcomAdapter, "execute", side_effect=TimeoutError("probe")) as execute):
                with self.assertRaisesRegex(RuntimeError, "inspect evidence"):
                    ecom_pair.run(argparse.Namespace(output=root, limit_rollouts=60, retry_failed=False))
                self.assertEqual(execute.call_count, 1)
                row = json.loads(next((root / "results").glob("*.json")).read_text())
                self.assertEqual(row["status"], "failed")
                with self.assertRaisesRegex(RuntimeError, "unresolved failure"):
                    ecom_pair.run(argparse.Namespace(output=root, limit_rollouts=60, retry_failed=False))
                self.assertEqual(execute.call_count, 1)
                with self.assertRaisesRegex(RuntimeError, "inspect evidence"):
                    ecom_pair.run(argparse.Namespace(output=root, limit_rollouts=60, retry_failed=True))
                row = json.loads(next((root / "results").glob("*.json")).read_text())
                self.assertEqual(len(row["attempt_history"]), 2)
                report = ecom_pair.report(root)
                self.assertEqual(report["failed_attempts"]["base"], 2)
                self.assertEqual(report["technical_failure_rate"]["base"], 1)


if __name__ == "__main__":
    unittest.main()
