"""Configuration pollution regressions; native tests use a local fake LLM, not paid APIs."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks.ecommerce import EcomAdapter
from harness2.cli import check as check_cli
from harness2.core import execution_env as EE
from harness2.core import runtime_tools
from harness2.core.proposer import Proposer, _call_fingerprint
from harness2.core.spec import ChannelSpec
from harness2.substrate.opencode import OpencodeSubstrate


class CleanEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        supplied_rg = runtime_tools.root() / "rg"
        if not supplied_rg.is_file():
            supplied_rg = self.root / "fake-rg"
            supplied_rg.write_text("#!/bin/sh\necho 'ripgrep test-fixture'\n")
            supplied_rg.chmod(0o755)
        patcher = patch.object(runtime_tools, "root", return_value=self.root / "tools")
        patcher.start()
        self.addCleanup(patcher.stop)
        runtime_tools.install(supplied_rg)
        self.profile = self.root / "models.json"
        self.profile.write_text('{"provider":{}}')
        for name, value in (
            ("ENV_MODE", "clean"),
            ("CLEAN_MODEL_CONFIG", self.profile),
            ("SOLVER_MODEL_OC", ""),
            ("IMPROVER_MODEL", ""),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "AGENTS.md").write_text("EXPERIMENT_INSTRUCTION\n")

    def test_check_cli_accepts_ecom_cell(self):
        with (
            patch.object(check_cli, "check", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(check_cli.main(["--bench", "ecom", "--substrate", "oc"]), 0)
        self.assertIn("Ready: ecom/oc", output.getvalue())

    def test_env_whitelist_and_separate_directories(self):
        original = {
            "HOME": "/personal",
            "PATH": "/personal/bin",
            "BASH_ENV": "/evil.sh",
            "OPENCODE_CONFIG_CONTENT": "personal",
            "NODE_OPTIONS": "--require evil",
            "PRIVATE_TOKEN": "secret",
            "HTTPS_PROXY": "http://localhost:1234",
        }
        before = original.copy()
        env = EE.prepare(self.workspace, source=original)
        self.assertEqual(original, before)
        self.assertNotEqual(env["HOME"], original["HOME"])
        for key in ("BASH_ENV", "NODE_OPTIONS", "PRIVATE_TOKEN", "OPENCODE_CONFIG_CONTENT"):
            self.assertNotIn(key, env)
        self.assertNotIn("/personal/bin", env["PATH"])
        self.assertEqual(env["HTTPS_PROXY"], original["HTTPS_PROXY"])
        for key in (
            "HOME",
            "TMPDIR",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
        ):
            self.assertTrue(Path(env[key]).is_relative_to(self.workspace))
        self.assertTrue((self.workspace / ".git").is_dir())
        self.assertNotIn("OPENCODE_DISABLE_PROJECT_CONFIG", env)

    def test_runtime_is_gitignored_but_harness_is_not(self):
        env = EE.prepare(self.workspace, source={})
        harness = self.workspace / "context/harness/systemprompt.md"
        harness.parent.mkdir(parents=True)
        harness.write_text("EXPERIMENT_HARNESS\n")
        runtime = self.workspace / ".harness2-env/data/probe.txt"
        runtime.write_text("runtime state\n")
        result = subprocess.run(
            ["/usr/bin/git", "check-ignore", "--no-index", "--stdin"],
            input=".harness2-env/data/probe.txt\ncontext/harness/systemprompt.md\n",
            cwd=self.workspace, env=env, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [".harness2-env/data/probe.txt"])

    def test_runtime_exclusion_preserves_existing_rules_and_is_idempotent(self):
        subprocess.run(
            ["/usr/bin/git", "-c", "init.templateDir=", "init", "-q", str(self.workspace)],
            check=True, capture_output=True, timeout=15,
        )
        exclude = self.workspace / ".git/info/exclude"
        exclude.parent.mkdir(exist_ok=True)
        original = b"# Keep operator rules\n*.local"
        exclude.write_bytes(original)
        EE.prepare(self.workspace, source={})
        expected = original + b"\n/.harness2-env/\n"
        self.assertEqual(exclude.read_bytes(), expected)
        EE.prepare(self.workspace, source={})
        self.assertEqual(exclude.read_bytes(), expected)
        self.assertFalse((self.workspace / ".gitignore").exists())

    def test_runtime_exclusion_overrides_an_earlier_negation(self):
        env = EE.prepare(self.workspace, source={})
        exclude = self.workspace / ".git/info/exclude"
        original = b"/.harness2-env/\n!/.harness2-env/\n"
        exclude.write_bytes(original)
        EE.prepare(self.workspace, source={})
        self.assertEqual(exclude.read_bytes(), original + b"/.harness2-env/\n")
        result = subprocess.run(
            ["/usr/bin/git", "check-ignore", ".harness2-env/models.json"],
            cwd=self.workspace, env=env, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_exclusion_supports_linked_git_worktrees(self):
        env = EE.prepare(self.workspace, source={})
        for args in (
            ["-c", "user.name=Snapshot Test", "-c", "user.email=snapshot@example.invalid",
             "commit", "--allow-empty", "--no-gpg-sign", "-qm", "test fixture"],
            ["worktree", "add", "--detach", str(self.root / "linked"), "HEAD"],
        ):
            subprocess.run(
                ["/usr/bin/git", *args], cwd=self.workspace, env=env,
                check=True, capture_output=True, timeout=15,
            )
        exclude = self.workspace / ".git/info/exclude"
        exclude.write_bytes(b"# Existing linked worktree rules\n*.local\n")
        linked = self.root / "linked"
        self.assertTrue((linked / ".git").is_file())
        linked_env = EE.prepare(linked, source={})
        self.assertEqual(
            exclude.read_bytes(), b"# Existing linked worktree rules\n*.local\n/.harness2-env/\n"
        )
        result = subprocess.run(
            ["/usr/bin/git", "check-ignore", ".harness2-env/models.json"],
            cwd=linked, env=linked_env, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rg_snapshot_is_materialised_and_tampering_rejected(self):
        env = EE.prepare(self.workspace, source={"PATH": "/personal/bin"})
        resolved = Path(env["PATH"].split(os.pathsep)[0]) / "rg"
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.read_bytes(), (runtime_tools.root() / "rg").read_bytes())
        self.assertEqual(EE.identity()["ripgrep"], runtime_tools.identity())
        (runtime_tools.root() / "rg").write_text("tampered")
        with self.assertRaisesRegex(config.ConfigError, "ripgrep missing or changed"):
            EE.prepare(self.root / "next", source={})

    def test_rejects_capability_and_secret_injection(self):
        for data in [
            {"provider": {}, "mcp": {}},
            {"provider": {}, "plugin": []},
            {"provider": {"test": {"npm": "file:///evil.ts"}}},
            {"provider": {"test": {"env": ["BASH_ENV"]}}},
            {"provider": {"test": {"options": {"apiKey": "literal-secret"}}}},
            {"provider": {"test": {"options": {"baseURL": "https://user:secret@example.org"}}}},
            {"provider": {"test": {"models": {"x": {"name": "{file:/secret}"}}}}},
            {"provider": {"test": {"options": {"baseURL": "https://example.org?key=secret"}}}},
            *[
                {"provider": {"test": {"models": {"x": model}}}}
                for model in [
                    {"options": {"mcp": {}}},
                    {"options": {"max_tokens": 99}},
                    {"options": {"thinking": {"type": "auto"}}},
                    {"options": {"reasoningEffort": "bad"}},
                    {"variants": {"high": {"plugin": []}}},
                    {"limit": {"output": True}},
                ]
            ],
        ]:
            with self.subTest(data=data):
                self.profile.write_text(json.dumps(data))
                with self.assertRaises(config.ConfigError):
                    EE.profile()

    def test_only_declared_model_credentials_and_cache_identity(self):
        self.profile.write_text(
            json.dumps(
                {
                    "provider": {
                        "test": {
                            "env": ["TEST_MODEL_KEY"],
                            "options": {"apiKey": "{env:TEST_MODEL_KEY}"},
                        }
                    }
                }
            )
        )
        env = EE.prepare(
            self.workspace, source={"TEST_MODEL_KEY": "hidden", "OTHER_API_KEY": "other"}
        )
        self.assertEqual(env["TEST_MODEL_KEY"], "hidden")
        self.assertNotIn("OTHER_API_KEY", env)
        self.assertNotIn("hidden", json.dumps(EE.identity()))
        before = _call_fingerprint("label", "mission", "system", None)
        self.profile.write_text('{"provider":{}}')
        self.assertNotEqual(before, _call_fingerprint("label", "mission", "system", None))
        with patch.object(config, "ENV_MODE", "legacy"):
            self.assertEqual(EE.identity(), {"mode": "legacy"})
        with patch.object(config, "ENV_MODE", "typo"):
            with self.assertRaises(config.ConfigError):
                EE.enabled()

    def test_missing_gateway_key_fails_before_cli_launch(self):
        self.profile.write_text(
            json.dumps(
                {
                    "provider": {
                        "gateway": {
                            "env": ["TEST_MODEL_KEY"],
                            "options": {"apiKey": "{env:TEST_MODEL_KEY}"},
                        }
                    }
                }
            )
        )
        with self.assertRaisesRegex(config.ConfigError, "TEST_MODEL_KEY"):
            EE.prepare(self.workspace, model="gateway/model", source={})
        self.assertFalse((self.workspace / ".harness2-env").exists())
        env = EE.prepare(
            self.workspace, model="gateway/model", source={"TEST_MODEL_KEY": "test-only"}
        )
        self.assertEqual(env["TEST_MODEL_KEY"], "test-only")

    def test_model_reasoning_controls_and_output_ceiling(self):
        self.profile.write_text(
            json.dumps(
                {
                    "provider": {
                        "test": {
                            "models": {
                                "solver": {
                                    "reasoning": True,
                                    "limit": {"context": 1048576, "output": 393216},
                                    "options": {
                                        "thinking": {"type": "enabled"},
                                        "reasoningEffort": "high",
                                    },
                                    "variants": {"max": {"reasoningEffort": "max"}},
                                }
                            }
                        }
                    }
                }
            )
        )
        env = EE.prepare(
            self.workspace,
            model="test/solver",
            source={"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "1"},
        )
        self.assertEqual(env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"], "393216")
        rendered = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        definition = rendered["provider"]["test"]["models"]["solver"]
        self.assertEqual(definition["options"]["reasoningEffort"], "high")
        self.assertEqual(definition["variants"]["max"]["reasoningEffort"], "max")

    def test_deepseek_profile_capacities_and_output_budget(self):
        public_profile = config.ROOT / "config/ecom-models.json"
        with patch.object(config, "CLEAN_MODEL_CONFIG", public_profile):
            data, keys = EE.profile()
        model = data["provider"]["ecom_deepseek"]["models"]["DeepSeek-V4.1-Flash"]
        self.assertEqual(model["limit"], {"context": 1048576, "output": 65536})
        self.assertTrue(model["reasoning"])
        self.assertEqual(
            model["options"], {"thinking": {"type": "enabled"}, "reasoningEffort": "high"}
        )
        self.assertEqual(set(model["variants"]), {"low", "high", "max"})
        self.assertEqual(keys, {"ECOM_LLM_API_KEY"})

    def test_vertex_never_falls_back_to_personal_adc(self):
        with self.assertRaisesRegex(config.ConfigError, "GOOGLE_APPLICATION_CREDENTIALS"):
            EE.prepare(self.workspace, model="google-vertex/test", source={})

    def test_rejects_uncertified_executor_and_improver(self):
        with self.assertRaisesRegex(config.ConfigError, "OpenCode"):
            EcomAdapter("cx").preflight()
        with self.assertRaisesRegex(config.ConfigError, "OpenCode"):
            Proposer(ChannelSpec("claude_cli", "test"))


NATIVE = (config.BUN.is_file() and (config.OPENCODE_DIR / "node_modules").is_dir()
          and (runtime_tools.root() / "rg").is_file())


@unittest.skipUnless(NATIVE, "run harness2-setup ecom for native OpenCode pollution tests")
class NativeOpenCodeTests(unittest.TestCase):
    setUp = CleanEnvironmentTests.setUp

    def hostile_home(self):
        host = self.root / "personal-home"
        for folder in (".agents/skills/host-forbidden", ".claude/skills/claude-forbidden"):
            skill = host / folder / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(
                f"---\nname: {skill.parent.name}\ndescription: HOST_FORBIDDEN_SKILL\n---\nHOST_FORBIDDEN_BODY\n"
            )
        (host / ".claude/CLAUDE.md").write_text("HOST_FORBIDDEN_INSTRUCTION")
        cfg = host / "xdg-config/opencode/opencode.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            json.dumps(
                {
                    "mcp": {"host_forbidden_mcp": {"type": "local", "command": ["false"]}},
                    "instructions": [str(host / ".claude/CLAUDE.md")],
                }
            )
        )
        return {
            "HOME": str(host),
            "XDG_CONFIG_HOME": str(host / "xdg-config"),
            "OPENCODE_CONFIG_CONTENT": cfg.read_text(),
            "BASH_ENV": str(host / "evil.sh"),
            "NODE_OPTIONS": "--require=/must-not-exist.js",
            "TEST_MODEL_KEY": "private-test-key",
        }

    def experiment_skill(self):
        skill = self.workspace / ".opencode/skills/experiment-skill/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: experiment-skill\ndescription: EXPERIMENT_SKILL\n---\nUse the experiment policy.\n"
        )

    def test_multistep_snapshots_exclude_runtime_and_preserve_harness_changes(self):
        package = config.OPENCODE_DIR / "packages/opencode"
        script = r'''
import fs from "node:fs/promises";
import path from "node:path";
import { bootstrap } from "./src/cli/bootstrap";
import { AppRuntime } from "./src/effect/app-runtime";
import { Snapshot } from "./src/snapshot";
import { Global } from "./src/global";
const workspace = process.env.PROBE_WORKSPACE;
const harness = path.join(workspace, "context/harness/systemprompt.md");
const rows = [];
let changeSeen = false;
await bootstrap(workspace, async () => {
  let previous;
  for (let iteration = 0; iteration < 12; iteration++) {
    if (iteration === 6) await fs.appendFile(harness, "VERIFIED_CHANGE\n");
    const hash = await AppRuntime.runPromise(Snapshot.Service.use(s => s.track()));
    if (iteration === 6) {
      const patch = await AppRuntime.runPromise(Snapshot.Service.use(s => s.patch(previous)));
      changeSeen = patch.files.includes(harness);
    }
    previous = hash;
    const base = path.join(Global.Path.data, "snapshot", "global");
    const dirs = await fs.readdir(base);
    const gitdir = path.join(base, dirs[0]);
    const proc = Bun.spawnSync(["/usr/bin/git", "--git-dir", gitdir, "--work-tree", workspace, "ls-files", "-z"]);
    if (proc.exitCode !== 0) throw new Error(proc.stderr.toString());
    const files = proc.stdout.toString().split("\0").filter(Boolean);
    rows.push({
      runtime: files.filter(f => f.startsWith(".harness2-env/")).length,
      ownObjects: files.filter(f => f.includes("/snapshot/global/") && f.includes("/objects/")).length,
      hasHarness: files.includes("context/harness/systemprompt.md"),
    });
  }
});
console.log(JSON.stringify({rows, changeSeen}));
process.exit(0);
'''
        results = {}
        for excluded in (False, True):
            workspace = self.root / ("excluded" if excluded else "positive-control")
            workspace.mkdir()
            harness = workspace / "context/harness/systemprompt.md"
            harness.parent.mkdir(parents=True)
            harness.write_text("EXPERIMENT_HARNESS\n")
            env = EE.prepare(workspace, source={}, tool_dirs=(config.BUN.parent,))
            if not excluded:
                # Positive control: the same native component must expose the
                # old defect when only the new exclusion is removed.
                exclude = workspace / ".git/info/exclude"
                if exclude.exists():
                    exclude.write_bytes(b"\n".join(
                        line for line in exclude.read_bytes().splitlines()
                        if line != b"/.harness2-env/"
                    ) + b"\n")
            env["PROBE_WORKSPACE"] = str(workspace)
            proc = subprocess.run(
                [str(config.BUN), "--cwd", str(package), "--conditions=browser", "-e", script],
                env=env, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            results[excluded] = json.loads(proc.stdout.strip().splitlines()[-1])
        control, clean = results[False], results[True]
        self.assertGreater(control["rows"][-1]["runtime"], control["rows"][0]["runtime"])
        self.assertGreater(control["rows"][-1]["ownObjects"], 0)
        self.assertTrue(all(row["runtime"] == 0 and row["ownObjects"] == 0 for row in clean["rows"]))
        self.assertTrue(all(row["hasHarness"] for row in clean["rows"]))
        self.assertTrue(clean["changeSeen"], "Excluding runtime must retain real harness diffs")

    def test_real_cli_detects_pollution_and_preserves_experiment_skills(self):
        self.experiment_skill()
        hostile = self.hostile_home()
        env = EE.prepare(self.workspace, source=hostile)
        audit = self.root / "audit.json"
        result = EE.audit(self.workspace, env, audit)
        self.assertEqual([s["name"] for s in result["skill"]], ["experiment-skill"])
        self.assertIn(str(self.workspace / "AGENTS.md"), result["instruction_paths"])
        self.assertNotIn("HOST_FORBIDDEN", audit.read_text())
        # Positive control: the actual CLI really does discover the hostile
        # global configuration when the isolation boundaries are removed.
        polluted = {
            **env,
            "HOME": hostile["HOME"],
            "XDG_CONFIG_HOME": hostile["XDG_CONFIG_HOME"],
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "0",
            "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "0",
        }
        with self.assertRaisesRegex(config.ConfigError, "External|MCP"):
            EE.audit(self.workspace, polluted, self.root / "must-not-exist.json")
        self.assertFalse((self.root / "must-not-exist.json").exists())

    def test_clean_merchantbench_rollout_and_audit_integrity(self):
        if not config.MERCHANTBENCH_PYTHON.is_file():
            self.skipTest("install the MerchantBench environment")
        manifest = self.root / "tasks.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "tasks": [
                        {
                            "id": "synthetic/clean",
                            "domain": "synthetic",
                            "master_seed": 42,
                            "horizon_steps": 24,
                        }
                    ],
                }
            )
        )
        fake = self.root / "executor.py"
        fake.write_text("""import json, os, subprocess, sys
from pathlib import Path
assert Path(os.environ["HOME"]).is_relative_to(Path.cwd())
assert "NODE_OPTIONS" not in os.environ
assert "BASH_ENV" not in os.environ
assert Path(os.environ["OPENCODE_CONFIG"]).is_file()
assert os.environ["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "393216"
while True:
    packet = json.loads(subprocess.check_output([sys.executable, "merchantbench.py", "observe"], text=True))
    if packet.get("terminal"): break
    subprocess.check_call([sys.executable, "merchantbench.py", "act", "end_of_step", "{}"], stdout=subprocess.DEVNULL)
print(json.dumps({"type":"text", "part":{"text":"Clean episode complete"}}))
""")
        self.profile.write_text(
            json.dumps(
                {
                    "provider": {
                        "ecom_test": {
                            "models": {
                                "test-model": {"limit": {"context": 1048576, "output": 393216}}
                            }
                        }
                    }
                }
            )
        )
        adapter = EcomAdapter(tasks_file=manifest, store_root=self.root / "episodes")
        with (
            patch.dict(os.environ, self.hostile_home()),
            patch.object(config, "SOLVER_MODEL_OC", "ecom_test/test-model"),
            patch.object(
                adapter.substrate,
                "run_argv",
                return_value=[str(config.MERCHANTBENCH_PYTHON), str(fake)],
            ),
        ):
            out = adapter.execute("synthetic/clean", None, "clean-native", timeout=120)
        self.assertTrue(adapter.delivered(out))
        self.assertEqual(adapter.score(out, "synthetic/clean")["final_net_assets"], 3000)
        audit = out / "environment.json"
        self.assertEqual(json.loads(audit.read_text())["environment"]["mode"], "clean")
        audit.write_text("tampered")
        self.assertFalse(adapter.artifacts_intact(out))

    def test_actual_executor_and_improver_with_local_fake_llm(self):
        captures = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captures.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                has_bash = any(
                    t.get("function", {}).get("name") == "bash" for t in body.get("tools", [])
                )
                saw_result = "CLEAN_TOOL_OK" in json.dumps(body.get("messages", []))
                has_skill = "experiment-skill" in json.dumps(body.get("tools", []))
                saw_skill = "<skill_content" in json.dumps(body.get("messages", []))
                if has_skill and not saw_skill:
                    chunks = (({"role": "assistant", "tool_calls": [{
                        "index": 0, "id": "skill-call", "type": "function",
                        "function": {"name": "skill", "arguments": '{"name":"experiment-skill"}'},
                    }]}, None), ({}, "tool_calls"))
                elif has_bash and not saw_result:
                    delta = {
                        "role": "assistant",
                        "reasoning_content": "LOCAL_REASONING_PROBE",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "isolation-call",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {
                                            "command": "printf CLEAN_TOOL_OK",
                                            "description": "Verify clean shell tool",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    chunks = ((delta, None), ({}, "tool_calls"))
                else:
                    chunks = (
                        ({"role": "assistant", "content": "ISOLATION_OK"}, None),
                        ({}, "stop"),
                    )
                for delta, finish in chunks:
                    event = {
                        "id": "local-test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "test-model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        model = "ecom_test/DeepSeek-V4.1-Flash"
        self.profile.write_text(
            json.dumps(
                {
                    "provider": {
                        "ecom_test": {
                            "npm": "@ai-sdk/openai-compatible",
                            "env": ["TEST_MODEL_KEY"],
                            "options": {
                                "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                                "apiKey": "{env:TEST_MODEL_KEY}",
                            },
                            "models": {
                                "DeepSeek-V4.1-Flash": {
                                    "id": "test-model",
                                    "limit": {"context": 1048576, "output": 65536},
                                    "tool_call": True,
                                    "reasoning": True,
                                    "options": {
                                        "thinking": {"type": "enabled"},
                                        "reasoningEffort": "low",
                                    },
                                    "variants": {"high": {"reasoningEffort": "high"}},
                                }
                            },
                        }
                    }
                }
            )
        )
        self.experiment_skill()
        hostile = self.hostile_home()
        with (
            patch.dict(os.environ, hostile),
            patch.object(config, "SOLVER_MODEL_OC", model),
            patch.object(config, "IMPROVER_MODEL", model),
        ):
            substrate = OpencodeSubstrate()
            env = EE.prepare(self.workspace, model=model, tool_dirs=(config.BUN.parent,))
            EE.audit(self.workspace, env, self.root / "executor-environment.json")
            env = substrate.run_env(
                env,
                self.workspace,
                output_token_max=int(env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"]),
            )
            result = subprocess.run(
                substrate.run_argv("Reply ISOLATION_OK. Do not use tools.", self.workspace, model),
                env=env,
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            self.assertIn("ISOLATION_OK", result.stdout)
            executor_captures = list(captures)
            captures.clear()
            prop = Proposer(
                ChannelSpec("opencode", model, effort="high", steps=3),
                audit_dir=self.root / "improver-audit",
                diagnostics=True,
            )
            reply = prop.call(
                "Reply ISOLATION_OK; do not use tools.", None, "isolation", max_attempts=1
            )
            self.assertIn("ISOLATION_OK", reply.text)
            self.assertTrue(
                list((self.root / "improver-audit/environments").rglob("environment.json"))
            )
            attempt = next((self.root / "improver-audit").glob("*.attempts/*"))
            transport = (attempt / "transport.jsonl").read_text()
            events = [json.loads(line) for line in transport.splitlines()]
            self.assertEqual(events[0]["event"], "preload_ready")
            self.assertIn("first_body_chunk", [e["event"] for e in events])
            self.assertIn("service=llm", (attempt / "stderr.log").read_text())
            self.assertTrue(list(attempt.glob("native/opencode*.db")),
                            "Retain the channel-specific native database too")
            self.assertEqual(json.loads((attempt / "process.json").read_text())
                             ["runtime_evidence"]["errors"], [])
            for private in (attempt / "stderr.log", attempt / "transport.jsonl"):
                self.assertNotIn("private-test-key", private.read_text())
            self.assertNotIn("LOCAL_REASONING_PROBE", transport)
            self.assertNotIn("ISOLATION_OK", transport)
        self.assertTrue(executor_captures)
        self.assertTrue(captures)
        text = json.dumps(executor_captures + captures)
        self.assertNotIn("HOST_FORBIDDEN", text)
        self.assertNotIn("host_forbidden_mcp", text)
        self.assertIn("EXPERIMENT_INSTRUCTION", json.dumps(executor_captures))
        self.assertIn("experiment-skill", json.dumps(executor_captures))
        self.assertTrue(any(
            m.get("role") == "tool" and "<skill_content" in json.dumps(m)
            and "Use the experiment policy." in json.dumps(m)
            for request in executor_captures for m in request.get("messages", [])
        ), "Must execute the real skill tool, not just discover the skill")
        for requests, effort in ((executor_captures, "low"), (captures, "high")):
            for request in requests:
                self.assertEqual(request["max_tokens"], 65536)
                # OpenCode's small/background title calls skip the user's
                # variant and retain model.options; main tool-agent calls
                # must honor the improver's explicit --variant high.
                expected_effort = effort if request.get("tools") else "low"
                self.assertEqual(request["reasoning_effort"], expected_effort)
                self.assertEqual(request["thinking"], {"type": "enabled"})
            self.assertTrue(
                any(
                    m.get("reasoning_content") == "LOCAL_REASONING_PROBE"
                    for request in requests
                    for m in request.get("messages", [])
                    if m.get("role") == "assistant"
                )
            )
            self.assertTrue(
                any(
                    m.get("role") == "tool" and "CLEAN_TOOL_OK" in json.dumps(m)
                    for request in requests
                    for m in request.get("messages", [])
                )
            )
        for audit in self.root.rglob("*environment.json"):
            self.assertNotIn("private-test-key", audit.read_text())


if __name__ == "__main__":
    unittest.main()
