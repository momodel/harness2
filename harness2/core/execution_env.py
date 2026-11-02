"""Explicit OpenCode configuration hygiene, not an OS security sandbox.

Only child environments are changed. Personal configuration is never edited.
The shared policy is used by ecom rollouts and all OpenCode improver sessions
when HARNESS2_ENV_MODE=clean. Other executor integrations are not certified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import config
from . import runtime_tools

VERSION = 1
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
SDK_PACKAGES = {
    "@ai-sdk/openai-compatible",
    "@ai-sdk/openai",
    "@ai-sdk/anthropic",
    "@ai-sdk/google",
    "@ai-sdk/google-vertex",
}
NETWORK_ENV = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
}
BUILTIN_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_GENERATIVE_AI_API_KEY",),
    "google-vertex": ("GOOGLE_APPLICATION_CREDENTIALS",),
}
ENV_REF = re.compile(r"\{env:([A-Z][A-Z0-9_]*)\}\Z")


def enabled():
    if config.ENV_MODE not in ("legacy", "clean"):
        raise config.ConfigError("HARNESS2_ENV_MODE must be legacy or clean")
    return config.ENV_MODE == "clean"


def _credential_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise config.ConfigError("Credential env names must be uppercase identifiers")
    if name in {"HOME", "PATH", "BASH_ENV", "ENV", "SHELL", "TMPDIR"} or name.startswith(
        (
            "OPENCODE_",
            "XDG_",
            "NODE_",
            "BUN_",
            "PYTHON",
            "LD_",
            "DYLD_",
            "GIT_",
            "HARNESS2_",
            "MERCHANTBENCH_",
        )
    ):
        raise config.ConfigError(f"Not a credential variable: {name}")
    return name


def _reasoning_options(options):
    """Permit reasoning controls only, not arbitrary provider/body overrides."""
    if not isinstance(options, dict) or options.keys() - {"thinking", "reasoningEffort"}:
        raise config.ConfigError("Clean model options allow thinking/reasoningEffort only")
    if "reasoningEffort" in options and options["reasoningEffort"] not in (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    ):
        raise config.ConfigError("Unsupported reasoningEffort")
    if "thinking" in options:
        thinking = options["thinking"]
        if (
            not isinstance(thinking, dict)
            or thinking.keys() != {"type"}
            or thinking["type"] not in ("enabled", "disabled")
        ):
            raise config.ConfigError("thinking must contain type enabled/disabled only")


def profile():
    """Accept model definitions only; reject executable/configuration escape hatches."""
    path = config.CLEAN_MODEL_CONFIG
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise config.ConfigError(
            f"Invalid clean model profile: {path} ({type(exc).__name__})"
        ) from exc
    if (
        not isinstance(data, dict)
        or data.keys() - {"provider"}
        or not isinstance(data.get("provider"), dict)
    ):
        raise config.ConfigError("Clean model profile must contain only a provider object")
    keys = set()
    for name, provider in data["provider"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not isinstance(provider, dict):
            raise config.ConfigError("Invalid clean provider")
        if provider.keys() - {"npm", "name", "env", "options", "models"}:
            raise config.ConfigError(f"Unsupported fields in provider {name}")
        if provider.get("npm", "@ai-sdk/openai-compatible") not in SDK_PACKAGES:
            raise config.ConfigError("Clean mode permits bundled model SDKs only")
        names = provider.get("env", [])
        if not isinstance(names, list):
            raise config.ConfigError("provider.env must be a list")
        keys.update(_credential_name(n) for n in names)
        options = provider.get("options", {})
        if not isinstance(options, dict) or options.keys() - {
            "baseURL",
            "apiKey",
            "timeout",
            "includeUsage",
        }:
            raise config.ConfigError("Unsupported clean provider options")
        if "apiKey" in options:
            ref = ENV_REF.fullmatch(str(options["apiKey"]))
            if not ref:
                raise config.ConfigError(
                    "apiKey must reference an environment variable, never a literal secret"
                )
            keys.add(_credential_name(ref[1]))
        if "baseURL" in options:
            url = urlparse(str(options["baseURL"]))
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise config.ConfigError(
                    "baseURL must be HTTP(S), without credentials, query or fragment"
                )
        models = provider.get("models", {})
        if not isinstance(models, dict):
            raise config.ConfigError("provider.models must be an object")
        for model in models.values():
            if not isinstance(model, dict) or model.keys() - {
                "id",
                "name",
                "tool_call",
                "reasoning",
                "temperature",
                "limit",
                "modalities",
                "cost",
                "options",
                "variants",
            }:
                raise config.ConfigError("Unsupported fields in clean model definition")
            limits = model.get("limit", {})
            if not isinstance(limits, dict) or limits.keys() - {"context", "input", "output"}:
                raise config.ConfigError("Unsupported clean model limits")
            if any(type(value) is not int or value <= 0 for value in limits.values()):
                raise config.ConfigError("Clean model limits must be positive integers")
            _reasoning_options(model.get("options", {}))
            variants = model.get("variants", {})
            if not isinstance(variants, dict):
                raise config.ConfigError("Clean model variants must be an object")
            for name, options in variants.items():
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                    raise config.ConfigError("Invalid clean model variant name")
                _reasoning_options(options)
    # OpenCode expands {file:...} and {env:...} before parsing. Only credential
    # references may expand; metadata/model names must not become secret carriers.
    copy = json.loads(json.dumps(data))
    for provider in copy["provider"].values():
        provider.get("options", {}).pop("apiKey", None)
    if re.search(r"\{(?:file|env):", json.dumps(copy)):
        raise config.ConfigError(
            "Only apiKey may use an environment reference; file expansion is forbidden"
        )
    return data, keys


def validate_credentials(*, model="", source=None):
    """Check selected providers' declared credentials without contacting their API."""
    source = os.environ if source is None else source
    data, _ = profile()
    models = {m for m in (model, config.SOLVER_MODEL_OC, config.IMPROVER_MODEL) if m}
    for provider_id in {m.split("/", 1)[0] for m in models}:
        provider = data["provider"].get(provider_id, {})
        ref = provider.get("options", {}).get("apiKey")
        if ref:
            name = ENV_REF.fullmatch(ref)[1]
            if not source.get(name):
                raise config.ConfigError(
                    f"{name} is required for {provider_id}; set it securely in the invoking environment"
                )
        else:
            names = provider.get("env", BUILTIN_KEYS.get(provider_id, ()))
            if names and not any(source.get(name) for name in names):
                raise config.ConfigError(
                    f"Set a credential for {provider_id}: {' or '.join(names)}"
                )


def identity():
    if not enabled():
        return {"mode": "legacy"}
    data, keys = profile()
    return {
        "mode": "clean",
        "version": VERSION,
        "opencode_pin": config.DEPENDENCY_PINS["opencode"],
        "improver_sha256": hashlib.sha256(
            (Path(__file__).parent / "proposer.py").read_bytes()
        ).hexdigest(),
        "policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile": data,
        "credential_names": sorted(keys),
        "system_path": SYSTEM_PATH,
        "ripgrep": runtime_tools.identity(),
        "runtime_tools_sha256": hashlib.sha256(Path(runtime_tools.__file__).read_bytes()).hexdigest(),
    }


def prepare(workspace: Path, *, model: str = "", tool_dirs=(), source=None):
    """Return a new env; do not mutate the parent or inherit arbitrary PATH entries."""
    if not enabled():
        raise config.ConfigError("prepare requires clean mode")
    source = os.environ if source is None else source
    validate_credentials(model=model, source=source)
    workspace = workspace.resolve()
    root = workspace / ".harness2-env"
    dirs = {
        key: root / name
        for key, name in {
            "HOME": "home",
            "XDG_CONFIG_HOME": "config",
            "XDG_DATA_HOME": "data",
            "XDG_STATE_HOME": "state",
            "XDG_CACHE_HOME": "cache",
            "TMPDIR": "tmp",
            "OPENCODE_CONFIG_DIR": "opencode-config",
        }.items()
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    tool_bin = runtime_tools.materialise(workspace)
    data, keys = profile()
    models = [m for m in (model, config.SOLVER_MODEL_OC, config.IMPROVER_MODEL) if m]
    providers = sorted({m.split("/", 1)[0] for m in models})
    for provider in providers:
        keys.update(BUILTIN_KEYS.get(provider, ()))
    env = {k: source[k] for k in (keys | NETWORK_ENV | {"LANG", "LC_ALL", "TZ"}) if k in source}
    bypass = ",".join(
        filter(None, [env.get("NO_PROXY"), env.get("no_proxy"), "127.0.0.1,localhost"])
    )
    env["NO_PROXY"] = env["no_proxy"] = bypass
    env.update({key: str(value) for key, value in dirs.items()})
    env.update(
        {
            "PATH": os.pathsep.join([str(tool_bin), *(str(Path(p).resolve()) for p in tool_dirs), SYSTEM_PATH]),
            "SHELL": "/bin/sh",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
        }
    )
    if "google-vertex" in providers:
        if not env.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise config.ConfigError(
                "Clean Vertex runs require explicit GOOGLE_APPLICATION_CREDENTIALS; personal gcloud HOME is not inherited"
            )
        env.update(
            {
                "GOOGLE_CLOUD_PROJECT": config.VERTEX_PROJECT,
                "GOOGLE_VERTEX_PROJECT": config.VERTEX_PROJECT,
                "GOOGLE_CLOUD_LOCATION": config.VERTEX_REGION,
                "GOOGLE_VERTEX_LOCATION": config.VERTEX_REGION,
                "VERTEXAI_LOCATION": config.VERTEX_REGION,
            }
        )
    # OpenCode independently caps model.limit.output at 32K (64K in the
    # executor launcher). Honor the selected model's explicit profile limit
    # for both roles, without inheriting a personal OPENCODE_* override.
    provider_id, _, model_id = model.partition("/")
    definition = data["provider"].get(provider_id, {}).get("models", {}).get(model_id, {})
    output_limit = definition.get("limit", {}).get("output")
    if output_limit:
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(output_limit)
    rendered = {**data, "enabled_providers": providers}
    if model:
        rendered["model"] = model
    filename = root / "models.json"
    filename.write_text(json.dumps(rendered))
    env["OPENCODE_CONFIG"] = str(filename)
    # Stop upward project discovery at this rollout, without disabling the
    # experiment's own AGENTS.md, .opencode skills, subagents or plugins.
    if not (workspace / ".git").exists():
        subprocess.run(
            ["/usr/bin/git", "-c", "init.templateDir=", "init", "-q", str(workspace)],
            env=env,
            check=True,
            capture_output=True,
            timeout=15,
        )
    # OpenCode snapshots use a Git object store under XDG_DATA_HOME. Without
    # this exclusion the snapshot stages its own objects/runtime on every turn,
    # growing even when the harness is unchanged. Use Git's path resolution so
    # existing linked worktrees (.git files) work too; never edit .gitignore.
    result = subprocess.run(
        ["/usr/bin/git", "rev-parse", "--git-path", "info/exclude"],
        cwd=workspace,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    exclude = workspace / result.stdout.strip()
    exclude.parent.mkdir(parents=True, exist_ok=True)
    contents = exclude.read_bytes() if exclude.exists() else b""
    rule = b"/.harness2-env/"
    # Keep the rule last so an earlier negation cannot re-include runtime files.
    if not contents.splitlines() or contents.splitlines()[-1] != rule:
        with exclude.open("ab") as stream:
            if contents and not contents.endswith(b"\n"):
                stream.write(b"\n")
            stream.write(rule + b"\n")
    return env


def audit(workspace: Path, env: dict, destination: Path):
    """Ask the real pinned CLI what config/skills it resolved; never call a model."""
    # Unlike `run --dir`, debug commands use process.cwd(). Do not pass Bun's
    # --cwd here: it would audit the OpenCode source checkout, not this rollout.
    package = config.OPENCODE_DIR / "packages/opencode"
    command = [
        str(config.BUN),
        "run",
        "--tsconfig-override",
        str(package / "tsconfig.json"),
        "--preload",
        str(package / "node_modules/@opentui/solid/scripts/preload.ts"),
        "--conditions=browser",
        str(package / "src/index.ts"),
        "debug",
    ]
    resolved = {}
    for kind in ("config", "skill"):
        result = subprocess.run(
            [*command, kind],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        resolved[kind] = json.loads(result.stdout)
    # Instructions are loaded separately from config.json. Inspect the actual
    # service too, or ~/.claude/CLAUDE.md could slip past a config-only audit.
    script = (
        'import {bootstrap} from "./src/cli/bootstrap";'
        'import {AppRuntime} from "./src/effect/app-runtime";'
        'import {Instruction} from "./src/session/instruction";'
        f"await bootstrap({json.dumps(str(workspace.resolve()))}, async () => {{"
        "const paths = await AppRuntime.runPromise(Instruction.Service.use(s => s.systemPaths()));"
        "console.log(JSON.stringify([...paths]));}); process.exit(0);"
    )
    result = subprocess.run(
        [str(config.BUN), "--cwd", str(package), "--conditions=browser", "-e", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    resolved["instruction_paths"] = json.loads(result.stdout)
    for path in resolved["instruction_paths"]:
        if not Path(path).resolve().is_relative_to(workspace.resolve()):
            raise config.ConfigError("External global instruction discovered in clean mode")
    cfg = resolved["config"]
    if cfg.get("mcp"):
        raise config.ConfigError("Clean mode forbids MCP configuration")
    for plugin in cfg.get("plugin", []):
        location = plugin[0] if isinstance(plugin, list) else plugin
        url = urlparse(location)
        if url.scheme != "file" or not Path(unquote(url.path)).resolve().is_relative_to(
            workspace.resolve()
        ):
            raise config.ConfigError("Only experiment-local plugins are allowed in clean mode")
    for skill in resolved["skill"]:
        if not Path(skill["location"]).resolve().is_relative_to(workspace.resolve()):
            raise config.ConfigError("External skill discovered in clean mode")
    for instruction in cfg.get("instructions", []):
        path = Path(instruction)
        if "://" in instruction or not (workspace / path).resolve().is_relative_to(
            workspace.resolve()
        ):
            raise config.ConfigError("External instruction discovered in clean mode")
    # Model credential values must never enter the audit or the cache identity.
    _, names = profile()
    for values in BUILTIN_KEYS.values():
        names.update(values)
    secrets = [env[name] for name in sorted(names) if env.get(name)]

    def redact(value):
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    payload = redact({"environment": identity(), "resolved": resolved})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return resolved
