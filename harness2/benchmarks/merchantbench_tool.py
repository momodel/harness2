"""Fixed agent-facing transport for MerchantBench's public SDK.

Copied next to the unmodified upstream SDK into each executor workspace. This
file is transport, not an editable harness component. Never accesses admin APIs.
The pinned public default scenario permits 30 /act turns per simulated step.
Reserve its last turn for an EXPLICIT end_of_step; never change that native
limit, silently advance the simulator, or treat quota exhaustion as network loss.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from merchantbench_tool_client import MerchantBenchToolClient

STATE = Path(__file__).resolve().parent / ".merchantbench_step.json"
TRACE = Path(__file__).resolve().parent / "operations.jsonl"
TURN_LIMIT = 30  # Protocol of the pinned default scenario, NOT a simulator override.


def emit(kind, payload):
    with TRACE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


def save_state(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(STATE)


def budget(turns):
    return {
        "limit": TURN_LIMIT,
        "turns_used": turns,
        "remaining_business_turns": max(0, TURN_LIMIT - turns - 1),
        "reserved_for": "end_of_step",
    }


def request_failure(exc):
    response = getattr(exc, "response", None)
    status = response.status_code if response is not None else None
    body = {}
    if response is not None:
        try:
            body = response.json()
        except ValueError:
            pass
    reason = body.get("error", "") if isinstance(body, dict) else ""
    if status == 429 and re.fullmatch(r"max_turns_per_step=\d+ reached", str(reason)):
        return {
            "ok": False,
            "error": "turn_quota_exhausted",
            "recoverable": False,
            "http_status": status,
            "retry": "This is a per-step action limit, not a network rate limit. Waiting or observing cannot reset an exhausted step; stop this incomplete rollout.",
        }
    return {
        "ok": False,
        "error": "stale_step" if status == 425 else "request_failed",
        "http_status": status,
        "retry": "observe before replanning",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("observe", "tools", "act"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}")
    args = parser.parse_args(argv)
    # A batch may spawn several bridge processes. Serialize the public budget
    # check and /act so they cannot all spend the same final business turn.
    with STATE.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return dispatch(args, parser)


def dispatch(args, parser):
    try:
        client = MerchantBenchToolClient(
            os.environ["MERCHANTBENCH_BASE_URL"],
            os.environ["MERCHANTBENCH_RUN_ID"],
            os.environ.get("MERCHANTBENCH_AGENT_ID", "agent_0"),
        )
    except requests.HTTPError as exc:
        # Finished runtimes may be unloaded before the SDK fetches the schema.
        if (
            args.command == "observe"
            and exc.response is not None
            and exc.response.status_code == 410
        ):
            save_state({"registered": True, "terminal": True})
            emit("terminal", {"terminal": True})
            return 0
        emit("error", request_failure(exc))
        return 1
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.command == "tools":
        print(json.dumps(client.tools(), ensure_ascii=False))
        return 0
    if args.command == "observe":
        if not state.get("registered"):
            client.register(framework=os.environ.get("HARNESS2_ECOM_FRAMEWORK", "harness2"))
        try:
            packet = client.observation()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 410:
                save_state({"registered": True, "terminal": True})
                emit("terminal", {"terminal": True})
                return 0
            emit("error", request_failure(exc))
            return 1
        turns = packet["turn_count"]
        if type(turns) is not int or turns < 0:
            raise ValueError("Invalid public turn_count; refusing to guess the action budget")
        save_state({"registered": True, "step": client.latest_env_t()})
        packet["action_budget"] = budget(turns)
        emit("observation", packet)
        return 0
    if not args.name:
        parser.error("act requires a tool name and optional JSON object")
    if state.get("terminal") or state.get("step") is None:
        parser.error("observe the current step before acting")
    parsed = json.loads(args.arguments)
    if not isinstance(parsed, dict):
        parser.error("tool arguments must be a JSON object")
    # Public read-only counter, not local guesswork: direct SDK traffic may also
    # have spent turns. Do not silently refresh a stale decision's X-Agent-Step.
    try:
        packet = client.observation(nowait=True)
    except requests.RequestException as exc:
        emit("error", request_failure(exc))
        return 1
    if client.latest_env_t() != state["step"]:
        save_state({"registered": True})
        emit("error", {"ok": False, "error": "stale_step", "retry": "observe before replanning"})
        return 1
    turns = packet["turn_count"]
    if type(turns) is not int or turns < 0:
        raise ValueError("Invalid public turn_count; refusing to guess the action budget")
    if turns >= TURN_LIMIT:
        emit(
            "error",
            {
                "ok": False,
                "error": "turn_quota_exhausted",
                "recoverable": False,
                "action_budget": budget(turns),
                "retry": "The simulated step is exhausted; waiting/observe cannot reset it. Stop this incomplete rollout.",
            },
        )
        return 1
    if turns >= TURN_LIMIT - 1 and args.name != "end_of_step":
        emit(
            "error",
            {
                "ok": False,
                "error": "end_of_step_reserved",
                "action_budget": budget(turns),
                "retry": "Call end_of_step explicitly now. The last /act turn is reserved; waiting cannot replenish this per-step budget.",
            },
        )
        return 1
    client._latest_env_t = state["step"]
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{time.time_ns()}",
                "type": "function",
                "function": {"name": args.name, "arguments": json.dumps(parsed)},
            }
        ],
    }
    emit("action", {"name": args.name, "arguments": parsed})
    try:
        response = client.act(call)
    except requests.RequestException as exc:
        if getattr(exc, "response", None) is not None and exc.response.status_code == 425:
            save_state({"registered": True})
        emit("result", request_failure(exc))
        return 1
    if response.get("hook_released") or response.get("step_done"):
        save_state({"registered": True})
    response["action_budget"] = budget(response["turn_idx"] + 1)
    emit("result", response)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(
            json.dumps({"error": str(exc), "retry": "observe before replanning"}), file=sys.stderr
        )
        raise SystemExit(1)
