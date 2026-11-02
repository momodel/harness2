"""Pure budget checks plus a real merchant-visible bridge integration, no models."""
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness2 import config
from harness2.benchmarks.merchantbench_runtime import LocalEpisode

HELPER = config.ROOT / "examples/ecom-budget-harness/oc/skills/listing-budget/scripts/budget.py"
spec = importlib.util.spec_from_file_location("budget_helper", HELPER)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def plan(self, count=4, stress=2):
        return {"window_hours": 12, "rationale": "cold start conservative forecast", "reserve": 500,
                "items": [{"product_id": f"p{i}", "sale_price": 150,
                           "expected_units": 1, "stress_units": stress} for i in range(count)]}

    def evaluate(self, plan, cash=2000, listings=()):
        products = {p["product_id"]: {"price": 100, "supplier_available": True, "quantity": 20}
                    for p in plan["items"]}
        return budget.evaluate(plan, {"balance": cash, "in_transit": 700}, listings, products)

    def test_checked_budget_records_all_required_fields(self):
        d = self.evaluate(self.plan())
        self.assertTrue(d["approved"])
        self.assertEqual((d["available_cash"], d["reserve"], d["procurement_budget"], d["sku_count"]),
                         (2000, 500, 1500, 4))
        self.assertEqual((d["expected_occupancy"], d["stress_occupancy"]), (400, 800))
        self.assertEqual(d["stress_remaining_cash"], 1200)

    def test_repeated_demand_not_one_unit_sum_controls_budget(self):
        # 4 x 100 fits, but 4 x 100 x 4 does not.
        d = self.evaluate(self.plan(stress=4))
        self.assertFalse(d["approved"])
        self.assertIn("Stress procurement exceeds cash minus reserve", d["errors"])

    def test_existing_shelf_must_be_budgeted_or_retired(self):
        plan = self.plan()
        listings = [{"product_id": "existing"}]
        with self.assertRaisesRegex(ValueError, "omits active"):
            self.evaluate(plan, listings=listings)
        plan["retire"] = ["existing"]
        self.assertTrue(self.evaluate(plan, listings=listings)["approved"])

    def test_no_cash_and_invalid_forecasts_fail_closed(self):
        self.assertFalse(self.evaluate(self.plan(), cash=0)["approved"])
        for value in (-1, float("nan"), float("inf"), True):
            plan = self.plan()
            plan["items"][0]["stress_units"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(plan)

    @unittest.skipUnless(config.MERCHANTBENCH_PYTHON.is_file(), "install MerchantBench")
    def test_helper_records_live_decision_in_operating_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            shutil.copy(config.PKG / "benchmarks/merchantbench_tool.py", workspace / "merchantbench.py")
            shutil.copy(config.merchantbench_repo() / "agent/sdk/merchantbench_tool_client.py", workspace)
            task = {"id": "synthetic/budget", "master_seed": 42, "horizon_steps": 24}
            with LocalEpisode(config.merchantbench_repo(), config.MERCHANTBENCH_PYTHON,
                              root / "state", task) as episode:
                env = episode.agent_environment()
                episode.start()
                subprocess.run([str(config.MERCHANTBENCH_PYTHON), "merchantbench.py", "observe"],
                               cwd=workspace, env=env, capture_output=True, check=True, timeout=30)
                plan = {"window_hours": 12, "rationale": "cold-start stress check", "reserve": 500,
                        "items": [{"product_id": "P00127", "sale_price": 31.39,
                                   "expected_units": 2, "stress_units": 5}]}
                (workspace / "plan.json").write_text(json.dumps(plan))
                result = subprocess.run([str(config.MERCHANTBENCH_PYTHON), str(HELPER), "plan.json"],
                                        cwd=workspace, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                record = json.loads(result.stdout)
                self.assertTrue(record["approved"])
                self.assertEqual(record["observed_step"], 0)
                self.assertAlmostEqual(record["stress_occupancy"], 19.39200192832126 * 5)
                # Rejected plans also remain in durable merchant-visible history.
                plan["items"][0]["stress_units"] = 1000
                (workspace / "plan.json").write_text(json.dumps(plan))
                result = subprocess.run([str(config.MERCHANTBENCH_PYTHON), str(HELPER), "plan.json"],
                                        cwd=workspace, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1)
                records = [json.loads(line) for line in (workspace / "operations.jsonl").read_text().splitlines()]
                decisions = [r["payload"] for r in records if r["kind"] == "budget_decision"]
                self.assertEqual([r["approved"] for r in decisions], [True, False])


if __name__ == "__main__":
    unittest.main()
