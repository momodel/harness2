"""Quota-reservation and reflection-outcome regressions; no model calls."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.benchmarks.merchantbench_runtime import LocalEpisode, clean_environment
from harness2.core import selection
from harness2.core.proposer import ProposerError, Reply
from harness2.demo import DemoAdapter

BRIDGE = Path(__file__).resolve().parents[1] / "harness2/benchmarks/merchantbench_tool.py"


class HTTPFailure(Exception):
    def __init__(self, status, error):
        self.response = Mock(status_code=status)
        self.response.json.return_value = {"error": error}
        super().__init__(f"HTTP {status}")


class BridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = types.SimpleNamespace(turns=0, step=0, calls=[], error=None)
        server = self.server

        class Client:
            def __init__(self, *_):
                self._latest_env_t = None

            def register(self, **_):
                return {"ok": True}

            def observation(self, **_):
                self._latest_env_t = server.step
                return {"tick": {"step": server.step}, "turn_count": server.turns, "text": "public"}

            def latest_env_t(self):
                return self._latest_env_t

            def act(self, call):
                name = call["tool_calls"][0]["function"]["name"]
                server.calls.append(name)
                if server.error:
                    raise server.error
                if server.turns >= 30:
                    raise HTTPFailure(429, "max_turns_per_step=30 reached")
                turn = server.turns
                server.turns += 1
                if name == "end_of_step":
                    server.turns = 0
                    server.step += 1
                return {"ok": True, "turn_idx": turn, "step_done": name == "end_of_step"}

        requests = types.ModuleType("requests")
        requests.HTTPError = HTTPFailure
        requests.RequestException = HTTPFailure
        sdk = types.ModuleType("merchantbench_tool_client")
        sdk.MerchantBenchToolClient = Client
        with patch.dict(sys.modules, {"requests": requests, "merchantbench_tool_client": sdk}):
            spec = importlib.util.spec_from_file_location("quota_bridge_under_test", BRIDGE)
            self.bridge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.bridge)
        self.bridge.STATE = self.root / "step.json"
        self.bridge.TRACE = self.root / "operations.jsonl"
        self.env = patch.dict(
            os.environ,
            {"MERCHANTBENCH_BASE_URL": "http://127.0.0.1:1", "MERCHANTBENCH_RUN_ID": "fixture"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.call("observe")

    def call(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = self.bridge.main(list(args))
        values = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
        return code, values[-1] if values else {}

    def test_reserves_last_turn_without_automatic_step_advance(self):
        self.server.turns = 29
        code, value = self.call("act", "query_balance", "{}")
        self.assertEqual(code, 1)
        self.assertEqual(value["error"], "end_of_step_reserved")
        self.assertEqual(self.server.calls, [])
        self.assertEqual(self.server.step, 0)
        self.assertEqual(self.call("act", "end_of_step", "{}")[0], 0)
        self.assertEqual(self.server.calls, ["end_of_step"])
        self.assertEqual(self.server.step, 1)

    def test_first_29_actions_are_allowed(self):
        for _ in range(29):
            self.assertEqual(self.call("act", "query_balance", "{}")[0], 0)
        self.assertEqual(self.server.turns, 29)
        self.assertEqual(self.call("act", "query_balance", "{}")[0], 1)
        self.assertEqual(len(self.server.calls), 29)

    def test_budget_resets_only_after_explicit_end_and_new_observation(self):
        self.server.turns = 29
        self.assertEqual(self.call("act", "end_of_step", "{}")[0], 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.call("act", "query_balance", "{}")
        _, packet = self.call("observe")
        self.assertEqual(packet["action_budget"]["remaining_business_turns"], 29)
        self.assertEqual(self.call("act", "query_balance", "{}")[0], 0)
        self.assertEqual(self.server.turns, 1)

    def test_exhausted_quota_is_not_misreported_as_a_retryable_network_error(self):
        self.server.turns = 30
        code, value = self.call("act", "end_of_step", "{}")
        self.assertEqual(code, 1)
        self.assertEqual(value["error"], "turn_quota_exhausted")
        self.assertFalse(value["recoverable"])
        self.assertEqual(self.server.calls, [])

    def test_stale_state_never_silently_refreshes_the_agents_decision(self):
        self.server.step = 1
        code, value = self.call("act", "query_balance", "{}")
        self.assertEqual(code, 1)
        self.assertEqual(value["error"], "stale_step")
        self.assertEqual(self.server.calls, [])
        self.assertNotIn("step", json.loads(self.bridge.STATE.read_text()))

    def test_observation_advertises_remaining_business_budget(self):
        self.server.turns = 28
        code, value = self.call("observe")
        self.assertEqual(code, 0)
        self.assertEqual(value["action_budget"]["remaining_business_turns"], 1)
        self.assertEqual(value["action_budget"]["limit"], 30)

    def test_http_quota_error_is_classified_and_action_has_an_error_result(self):
        self.server.error = HTTPFailure(429, "max_turns_per_step=30 reached")
        code, value = self.call("act", "query_balance", "{}")
        self.assertEqual(code, 1)
        self.assertEqual(value["error"], "turn_quota_exhausted")
        events = [json.loads(line) for line in self.bridge.TRACE.read_text().splitlines()]
        self.assertEqual([e["kind"] for e in events][-2:], ["action", "result"])
        self.assertFalse(events[-1]["payload"]["ok"])


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root / "demo")
        self.context = self.root / "reflection"
        self.context.mkdir()
        self.sel = selection.Selection("ledger-1", "agg", 1, chosen_label="agg1", ok=True)

    def distil(self, text=None, error=None):
        proposer = Mock()
        if error:
            proposer.call.side_effect = error
        else:
            proposer.call.return_value = Reply("experience.agg1", text, True)
        return selection._distil(
            "ledger-1", self.sel, {}, adapter=self.adapter, prop=proposer, root=self.context
        )

    def test_intentional_no_entry_is_persisted_distinctly(self):
        self.assertEqual(self.distil("<edits>comparison</edits>\nNO_ENTRY"), {})
        self.assertEqual(self.sel.to_dict()["reflection_status"], "no_entry")
        self.assertEqual(self.sel.reflection_error, "")

    def test_transport_failure_is_not_no_entry(self):
        self.assertEqual(self.distil(error=ProposerError("fixture transport failure")), {})
        self.assertEqual(self.sel.reflection_status, "error")
        self.assertIn("transport", self.sel.reflection_error)

    def test_malformed_or_embedded_no_entry_is_not_an_abstention(self):
        for index, text in enumerate(
            (
                "<edits>NO_ENTRY</edits>",
                "NO_ENTRY plus commentary",
                "<experience>cue: x</experience>",
            )
        ):
            with self.subTest(index=index):
                self.assertEqual(self.distil(text), {})
                self.assertEqual(self.sel.reflection_status, "error")
                self.assertIn("invalid", self.sel.reflection_error)

    def test_valid_experience_is_distilled_not_assumed_written(self):
        result = self.distil("<experience>\ncue: c\nlesson: l\nevidence: e\n</experience>")
        self.assertEqual(result["lesson"], "l")
        self.assertEqual(self.sel.reflection_status, "distilled")


class NativeBridgeTests(unittest.TestCase):
    def test_unmodified_native_30_turn_limit_still_rejects_exhausted_end(self):
        repo = config.merchantbench_repo()
        with tempfile.TemporaryDirectory() as directory:
            task = {
                "id": "synthetic/native-quota-proof",
                "domain": "synthetic",
                "master_seed": 0,
                "horizon_steps": 1,
            }
            with LocalEpisode(
                repo, config.MERCHANTBENCH_PYTHON, Path(directory) / "state", task
            ) as episode:
                episode.start()
                env = episode.agent_environment(clean_environment())
                env["PYTHONPATH"] = str(repo / "agent/sdk")
                script = """
import os,requests
from merchantbench_tool_client import MerchantBenchToolClient
c=MerchantBenchToolClient(os.environ['MERCHANTBENCH_BASE_URL'],os.environ['MERCHANTBENCH_RUN_ID'],'agent_0')
c.register(framework='native-quota-proof');c.observation()
def act(name):
 return c.act({'role':'assistant','content':'','tool_calls':[{'id':f'{name}_{__import__("time").time_ns()}','type':'function','function':{'name':name,'arguments':'{}'}}]})
for _ in range(30):act('query_balance')
assert c.observation(nowait=True)['turn_count']==30
try:act('end_of_step')
except requests.HTTPError as exc:
 assert exc.response.status_code==429
 assert exc.response.json()['error']=='max_turns_per_step=30 reached'
else:raise AssertionError('Native quota changed; bridge budget must be revisited')
print('ORIGINAL_NATIVE_LIMIT_UNCHANGED')
"""
                result = subprocess.run(
                    [str(config.MERCHANTBENCH_PYTHON), "-c", script],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ORIGINAL_NATIVE_LIMIT_UNCHANGED", result.stdout)

    def test_real_native_limit_preserved_and_explicit_end_reaches_terminal(self):
        repo = config.merchantbench_repo()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "agent"
            workspace.mkdir()
            shutil.copy(BRIDGE, workspace / "merchantbench.py")
            shutil.copy(repo / "agent/sdk/merchantbench_tool_client.py", workspace)
            task = {
                "id": "synthetic/reserve-proof",
                "domain": "synthetic",
                "master_seed": 0,
                "horizon_steps": 1,
            }
            with LocalEpisode(repo, config.MERCHANTBENCH_PYTHON, root / "state", task) as episode:
                episode.start()
                script = """
import contextlib,io,json,merchantbench,sys,subprocess
from concurrent.futures import ThreadPoolExecutor
with contextlib.redirect_stdout(io.StringIO()):
 assert merchantbench.main(['observe'])==0
 for i in range(28):assert merchantbench.main(['act','query_balance','{}'])==0
 def call(_):
  return subprocess.run([sys.executable,'merchantbench.py','act','query_balance','{}'],capture_output=True,text=True,timeout=15).returncode
 with ThreadPoolExecutor(2) as pool:
  codes=list(pool.map(call,range(2)))
 assert sorted(codes)==[0,1],codes
 assert merchantbench.main(['act','query_balance','{}'])==1
 assert merchantbench.main(['act','end_of_step','{}'])==0
print('RESERVED_AND_RELEASED')
"""
                result = subprocess.run(
                    [str(config.MERCHANTBENCH_PYTHON), "-c", script],
                    cwd=workspace,
                    env=episode.agent_environment(clean_environment()),
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("RESERVED_AND_RELEASED", result.stdout)
                self.assertEqual(episode.finish(timeout=30)["terminal_status"], "finished")


if __name__ == "__main__":
    unittest.main()
