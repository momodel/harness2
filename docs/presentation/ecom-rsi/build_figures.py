#!/usr/bin/env python3
"""Build four evidence-backed talk figures. Standard library; no model/network calls.

Only this directory is written. Historical runs are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from harness2.benchmarks.ecom_analysis import paired_summary  # noqa: E402
from harness2.core.runmark import fingerprint  # noqa: E402

TRAIN = ROOT / "runs/ecom-train-20261023-102614-1d6e02"
EARLY = ROOT / "runs/ecom-train-20261022-135821-2bc054"
HELDOUT = ROOT / "runs/ecom-heldout-seed15-20261023-170455"
FIXED = ROOT / "runs/ecom-rsi-snapshot-fixed-20261021-233057"
W, H = 1920, 1080
C = {
    "paper": "#F7F6F0", "ink": "#183134", "muted": "#647475",
    "line": "#D8DED6", "white": "#FFFFFF", "teal": "#087D70",
    "pale": "#E5F1E9", "blue": "#3C657F", "blue_pale": "#EAF0F4",
    "amber": "#A46C1A", "amber_pale": "#F8EFD9", "red": "#B54C3B",
    "red_pale": "#FAEDE7", "gray": "#879391",
}
NAMES = ["01-experiment-timeline", "02-round-by-round-changes",
         "03-profit-with-intervals", "04-evaluated-version-flow"]
LABELS = ["实验时间线", "逐轮修改图", "带区间的收益对比图", "最终被评估版本的流转图"]
INPUTS: dict[str, str] = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_bytes(path):
    path = Path(path)
    content = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
    return content


def load(path):
    return json.loads(source_bytes(path))


def equal_numbers(left, right):
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_numbers(v, right[k]) for k, v in left.items()))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(equal_numbers(a, b) for a, b in zip(left, right)))
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and math.isclose(left, right, abs_tol=1e-8)
    return left == right


def effective_tree(directory):
    # README stubs and empty always-on materializations have no effective text.
    result = {}
    for p in sorted(directory.rglob("*")):
        if (not p.is_file() or p.name in {"README.md", ".adapt_complete", ".DS_Store"}
                or "__pycache__" in p.parts or p.suffix == ".pyc"):
            continue
        data = source_bytes(p)
        if data:
            result[p.relative_to(directory).as_posix()] = data.decode("utf-8")
    return result


def collect():
    INPUTS.clear()
    report = load(HELDOUT / "report.json")
    plan = load(HELDOUT / "plan.json")
    provenance = load(HELDOUT / "freeze-provenance.json")
    evaluation = load(HELDOUT / "evaluation-status.json")
    pilot = load(HELDOUT / "pilot-status.json")
    training = load(TRAIN / "results-summary.json")
    train_config = load(TRAIN / "train.json")
    early = load(EARLY / "results-summary.json")
    repaired = load(FIXED / "verification.json")
    repaired_status = load(FIXED / "job-status.json")
    early_status = load(EARLY / "job-status.json")
    train_status = load(TRAIN / "job-status.json")
    source_bytes(ROOT / "harness2/benchmarks/ecom_analysis.py")
    source_bytes(ROOT / "runs/ecom-holdout-preflight-20261023-170455/DIAGNOSIS.md")
    source_bytes(ROOT / "runs/ecom-rsi-20261021-172043/diagnostics/ROOT_CAUSE.md")
    require(evaluation["verification_passed"] and evaluation["frozen_harnesses_intact"],
            "Final held-out verification did not pass")
    require(plan["library_updates"] is False and plan["repeats"] == 3,
            "This figure requires a frozen 3-repeat protocol")
    require(train_config["k"] == 1 and train_config["mode"] == "sequential"
            and train_config["hours"] == 24,
            "Round descriptions require the saved sequential k=1 / 24h batch")
    require(train_config["solver_model"] == plan["solver_model"], "Model differs between stages")
    require([task["master_seed"] for task in train_config["tasks"]]
            == [11, 16, 18, 13, 19, 15, 12, 17, 10, 14], "Task stream changed")
    require(repaired["passed"] and len(repaired["episodes"]) == 4
            and all(episode["complete"] for episode in repaired["episodes"].values()),
            "Timeline requires four completed repaired rollouts")
    require(early["complete_rollouts"] == 34 and early["graded"] == 8
            and early["ungraded"] == 2 and not early["verification"]["passed"],
            "Early-batch timeline annotation no longer matches")
    require(training["counts"] == {"total": 10, "finished": 10, "complete": 8,
                                   "failed": 1, "fallbacks": 1, "rollouts": 35,
                                   "planned_rollouts": 40}
            and not training["verification"]["passed"], "Latest-batch timeline changed")
    expected_seeds = [t["master_seed"] for t in plan["tasks"]]
    require(expected_seeds == list(range(1000, 1010)), "Unexpected held-out seeds")
    require(all(t["horizon_steps"] == 24 for t in plan["tasks"]), "Unexpected horizon")
    rows = []
    grouped = defaultdict(dict)
    for seed in expected_seeds:
        for repeat in range(plan["repeats"]):
            for arm in ("base", "agg"):
                row = load(HELDOUT / "results" / f"seed{seed}-r{repeat}-{arm}.json")
                require((row["master_seed"], row["repeat"], row["arm"])
                        == (seed, repeat, arm), "Result identity mismatch")
                require(row["status"] == "ok" and row["metrics"]["reconciled"],
                        "Incomplete/unreconciled cell cannot become a plotted score")
                episode = HELDOUT / "episodes" / Path(row["episode"]).name
                measurement = source_bytes(episode / "measurement/episode.json")
                require(hashlib.sha256(measurement).hexdigest()
                        == row["metrics"]["measurement_sha256"], "Measurement hash mismatch")
                native = json.loads(measurement)
                require(native["terminal_status"] == "finished", "Nonterminal native result")
                for key, series_key in [("net_profit", "cum_net_profit"),
                                        ("final_net_assets", "net_assets")]:
                    require(math.isclose(row["metrics"][key],
                                         native["merchant_section"]["series"][series_key][-1][1],
                                         abs_tol=1e-8), "Result disagrees with native measurement")
                rows.append(row)
                grouped[(seed, repeat)][arm] = row
    recomputed = paired_summary(rows, planned_pairs=30)
    require(equal_numbers(recomputed, report), "Saved report disagrees with result records")
    seed_stats = []
    for seed in expected_seeds:
        arm_means = {arm: st.mean(grouped[(seed, r)][arm]["metrics"]["net_profit"]
                                 for r in range(3)) for arm in ("base", "agg")}
        seed_stats.append({"seed": seed, **arm_means,
                           "delta": arm_means["agg"] - arm_means["base"]})
    rng = random.Random(0)
    samples = {arm: [] for arm in ("base", "agg")}
    # Same seed draw for both arms: preserve pairing and all 3 repeats as a cluster.
    for _ in range(2000):
        draw = rng.choices(seed_stats, k=len(seed_stats))
        for arm in samples:
            samples[arm].append(st.mean(x[arm] for x in draw))
    arm_intervals = {arm: [sorted(values)[49], sorted(values)[1949]]
                     for arm, values in samples.items()}
    candidate = (TRAIN / "ecom/adapt/ecom-oc-seq1-s10"
                 / "synthetic__train24h-seed15/harnesses/c1")
    require(fingerprint(candidate) == provenance["source_fingerprint"],
            "Training candidate fingerprint changed")
    for arm in ("base", "agg"):
        require(fingerprint(HELDOUT / arm) == plan["harness_fingerprints"][arm],
                f"Frozen {arm} fingerprint changed")
    require(effective_tree(candidate) == effective_tree(HELDOUT / "agg"),
            "Frozen candidate's effective text changed")
    base_source = candidate.parent / "base"
    require(effective_tree(base_source) == effective_tree(HELDOUT / "base"),
            "Frozen Base differs from the training Base")
    descriptions = {
        11: "现金容量规划；桥接输出解析与执行技能",
        16: "经营周期技能；资金容量与交付风险检查",
        18: "现金与订单核对；供应商事件先映射再下架",
        13: "分窗口经营规则；最终仍扩大货架并亏损",
        19: "经营技能 + 批次检查、历史检查脚本",
        15: "分窗口预算、储备与已成交采购均价检查",
        12: "低采购成本选品；资金容量与订单核对",
        17: "批次构建、历史校验；区分发货与总交期",
        10: "基线经营失败；没有进入候选修改",
        14: "合成回退到 Base；最终没有有效新改动",
    }
    training_by_seed = {row["seed"]: row for row in training["rows"]}
    rounds = []
    for index, task in enumerate(train_config["tasks"], 1):
        seed = task["master_seed"]
        encoded = task["id"].replace("/", "__")
        summary = load(TRAIN / "ecom/adapt/ecom-oc-agg1-s10" / encoded / "summary.json")
        trees = TRAIN / "ecom/adapt/ecom-oc-seq1-s10" / encoded / "harnesses"
        final = effective_tree(trees / "c1") if (trees / "c1").exists() else {}
        base = effective_tree(trees / "base") if (trees / "base").exists() else {}
        changed = sorted(k for k in base.keys() | final.keys() if base.get(k) != final.get(k))
        final_row = training_by_seed[seed]
        delta = final_row.get("delta")
        if base:
            require(base == effective_tree(HELDOUT / "base"), "Training Base changed across tasks")
        if delta is not None:
            # Avoid subtracting two already-rounded asset values (seed13 differs by 0.01).
            selected = summary["selections"]["seq"]["chosen_run_dir"]
            base_episode = summary["base"][0]["run_dir"]
            profits = []
            for episode_path in (base_episode, selected):
                native = load(TRAIN / "ecom/episodes" / Path(episode_path).name
                              / "measurement/episode.json")
                require(native["terminal_status"] == "finished", "Unfinished training pair")
                profits.append(native["merchant_section"]["series"]["cum_net_profit"][-1][1])
            delta = profits[1] - profits[0]
            require(abs(delta - final_row["delta"]) < 0.02,
                    "Training summary differs from native accounting")
        rounds.append({"order": index, "seed": seed, "description": descriptions[seed],
                       "changed_files_vs_base": changed,
                       "status": final_row["status"], "delta": delta,
                       "summary_delta_rounded": final_row.get("delta"),
                       "reflection_status": final_row["reflection_status"],
                       "fallbacks": final_row["fallbacks"]})
        if seed == 15:
            require(summary["selections"]["seq"]["trivial"],
                    "The k=1 candidate unexpectedly used a selector session")
            require(set(final) == {"systemprompt.md", "skills/merchant-operating-cycle/SKILL.md"},
                    "The final candidate is no longer prompt + one skill")
    pair_deltas = [p["agg"]["metrics"]["net_profit"] - p["base"]["metrics"]["net_profit"]
                   for p in grouped.values()]
    return {
        "scope": "Public synthetic MerchantBench; fixed weights; 24h operating window + drain",
        "report": report, "arm_mean_95ci": arm_intervals, "seed_means": seed_stats,
        "bootstrap": {"unit": "seed (retain both arms and all 3 repeats)", "resamples": 2000,
                      "rng_seed": 0, "sorted_indices": [49, 1949],
                      "arm_ci_origin": "New offline computation for this chart",
                      "delta_ci_origin": "Saved report, reproduced with paired_summary"},
        "wins": sum(x > 0 for x in pair_deltas), "losses": sum(x < 0 for x in pair_deltas),
        "positive_seed_means": sum(x["delta"] > 0 for x in seed_stats),
        "rounds": rounds,
        "freeze": {"candidate_training_seed": 15, "candidate_label": "c1",
                   "source_fingerprint": provenance["source_fingerprint"],
                   "frozen_fingerprints": provenance["frozen_fingerprints"],
                   "effective_tree_equal": True, "base_matches_training_base": True,
                   "human_screening": "Engineering screening, not highest-profit selection",
                   "normalization": "Omit README stubs; materialize absent always-on files as empty"},
        "timeline": {
            "repaired": {"started": repaired_status["started_at"],
                         "finished": repaired_status["finished_at"],
                         "passed": repaired["passed"]},
            "early_training": {"created": load(EARLY / "train.json")["created_at"],
                               "last_start": early_status["started_at"],
                               "finished": early_status["finished_at"],
                               "rollouts": early["complete_rollouts"],
                               "graded_pairs": early["graded"],
                               "verification_passed": early["verification"]["passed"]},
            "latest_training": {"started": train_status["started_at"],
                                "finished": train_status["finished_at"],
                                "counts": training["counts"],
                                "verification_passed": training["verification"]["passed"]},
            "freeze_pilot": {"prepared": provenance["prepared_at"],
                             "pilot_finished": pilot["finished_at"], "pilot_pairs": 1},
            "heldout": {"continuation_started": evaluation["started_at"],
                        "finished": evaluation["finished_at"],
                        "complete_pairs": evaluation["complete_pairs"],
                        "recorded_attempts": evaluation["recorded_rollout_attempts"]},
        },
        "input_sha256": dict(sorted(INPUTS.items())),
    }


class SVG:
    def __init__(self, number, title, subtitle):
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                      f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      f'<desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                      'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      '<path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker></defs>',
                      '<style>text{font-family:"PingFang SC","Hiragino Sans GB",'
                      '"Noto Sans CJK SC",sans-serif}.title{font-family:"Songti SC",'
                      '"Noto Serif CJK SC",serif;font-weight:700}.num{font-family:'
                      '"Avenir Next","PingFang SC",sans-serif;font-weight:600}'
                      '.mono{font-family:"SFMono-Regular",Menlo,monospace}</style>']
        self.rect(0, 0, W, H, C["paper"])
        self.text(80, 65, f"HARNESS RSI  /  MERCHANTBENCH     •     实验证据 {number:02d}",
                  20, C["muted"], weight=600)
        self.text(80, 140, title, 54, cls="title")
        self.text(80, 187, subtitle, 25, C["muted"])
        self.line(80, 216, 1840, 216)
        self.text(1840, 65, f"0{number} / 04", 20, C["muted"], anchor="end", cls="mono")

    def add(self, value):
        self.items.append(value)

    def rect(self, x, y, w, h, fill, stroke="none", radius=0):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                 f'fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, value, size=24, color=None, *, weight=400, anchor="start", cls=""):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or C["ink"]}" '
                 f'font-weight="{weight}" text-anchor="{anchor}" class="{cls}">'
                 f'{escape(str(value))}</text>')

    def lines(self, x, y, values, size=24, gap=40, color=None):
        for i, value in enumerate(values):
            self.text(x, y + i * gap, value, size, color)

    def line(self, x1, y1, x2, y2, color=None, width=1.5, dash=None, arrow=False):
        self.path(f"M{x1} {y1}L{x2} {y2}", color, width, dash, arrow)

    def path(self, d, color=None, width=2, dash=None, arrow=False):
        self.add(f'<path d="{d}" fill="none" stroke="{color or C["line"]}" '
                 f'stroke-width="{width}" stroke-linejoin="round" '
                 f'{f"stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""} '
                 f'{"marker-end=" + chr(34) + "url(#arrow)" + chr(34) if arrow else ""}/>')

    def circle(self, x, y, r, fill, stroke="none", width=1):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{width}"/>')

    def footer(self, note, source):
        self.line(80, 980, 1840, 980)
        self.text(80, 1015, note, 20, C["muted"])
        self.text(80, 1049, source, 17, C["muted"])

    def finish(self):
        return "\n".join(self.items + ["</svg>", ""])


def timeline(data):
    s = SVG(1, "从链路跑通，到冻结验证", "这不是一条收益不断上涨的曲线，而是一条证据逐步变强的路径。")
    stages = [
        ("10.21", "修复运行链路", "4 / 4", "经营执行完整", C["blue"], C["blue_pale"],
         ["INV / DOM 曾超时回退", "排除 Snapshot 自包含", "新目录重跑并完成反思", "单任务验收通过"], "链路验证 ≠ 泛化证据"),
        ("10.22–23", "探索训练 A", "8 / 10", "任务有配对评分", C["amber"], C["amber_pale"],
         ["seed 0–9  ·  k=1", "34 / 40 次经营完成", "完整配对：7 胜 1 负", "整批验收未通过"], "保留失败与缺失"),
        ("10.23", "探索训练 B", "8 + 1 + 1", "完整 / 回退 / 失败", C["amber"], C["amber_pale"],
         ["seed 10–19  ·  k=1", "35 / 40 次经营完成", "生成不同任务的候选", "整批验收未通过"], "不是冻结泛化评测"),
        ("10.23", "筛选、冻结、试跑", "seed15", "候选 c1 被冻结", C["teal"], C["pale"],
         ["按工程规则筛选候选", "不按训练利润取最大", "先完成 1 对执行试跑", "未按试跑利润再调策略"], "先冻结，再验证"),
        ("10.26", "全量冻结验证", "30 / 30", "完整配对", C["teal"], C["pale"],
         ["新 seed 1000–1009", "每 seed 重复 3 次", "60 次执行，0 技术失败", "快照完整性验收通过"], "支持本次短窗口效果"),
    ]
    s.line(100, 331, 1810, 331, C["line"], width=4)
    for i, (date, title, value, unit, color, pale, lines, takeaway) in enumerate(stages):
        x = 80 + i * 358
        s.text(x, 278, date, 31, color, weight=600, cls="num")
        s.circle(x + 20, 331, 19, color)
        s.text(x + 20, 338, i + 1, 20, C["white"], anchor="middle", weight=600)
        s.rect(x, 385, 328, 415, C["white"], C["line"], 10)
        s.rect(x, 385, 328, 7, color)
        s.text(x + 22, 437, title, 27, weight=600)
        s.text(x + 22, 511, value, 43, color, cls="num")
        s.text(x + 22, 549, unit, 21, C["muted"])
        s.lines(x + 22, 605, lines, size=22, gap=42)
        s.rect(x, 825, 328, 55, pale, radius=6)
        s.text(x + 164, 860, takeaway, 21, color, anchor="middle", weight=600)
    s.text(80, 939, "阅读顺序：能完成任务  →  能生成改动  →  冻结后还能否在新 seed 上有效", 27)
    s.footer("2026 年；按阶段顺序排列，横向间距不表示耗时。训练评分为事后离线核算。",
             "来源：运行 job-status / verification、两批 results-summary、freeze-provenance、最终 evaluation-status；详见 evidence.json。")
    return s.finish()


def rounds_figure(data):
    s = SVG(2, "逐轮修改：看实际改动，也保留失败", "最新训练批次的真实任务顺序；每个任务执行 k=1，不是同一个版本连续升级 10 次。")
    s.rect(80, 242, 1760, 53, C["ink"], radius=4)
    headers = [(104, "轮次"), (225, "任务 seed"), (390, "最终候选的主要改动 / 观察"),
               (1160, "相对 Base 的有效文件"), (1510, "执行状态"), (1810, "利润差")]
    for x, label in headers:
        s.text(x, 276, label, 21, C["white"], anchor="end" if x == 1810 else "start")
    for i, row in enumerate(data["rounds"]):
        y = 305 + i * 55
        highlight = row["seed"] == 15
        fill = C["pale"] if highlight else (C["white"] if i % 2 == 0 else C["paper"])
        s.rect(80, y, 1760, 53, fill)
        if highlight:
            s.rect(80, y, 5, 53, C["teal"])
        baseline = y + 35
        s.text(104, baseline, f'{row["order"]:02d}', 22, C["muted"], cls="mono")
        s.text(225, baseline, str(row["seed"]), 25, weight=600, cls="num")
        s.text(390, baseline, row["description"], 23)
        files = row["changed_files_vs_base"]
        if row["seed"] == 10:
            label = "—"
        elif not files:
            label = "0（与 Base 相同）"
        else:
            skills = sum(p.endswith("/SKILL.md") for p in files)
            scripts = sum(p.endswith(".py") for p in files)
            label = f"提示 + {skills} 技能" + (f" + {scripts} 脚本" if scripts else "")
        s.text(1160, baseline, label, 21, C["muted"])
        status = {"complete": "完整", "failed": "失败", "warning": "回退"}[row["status"]]
        if highlight:
            status = "完整 · 冻结源"
        elif row["seed"] == 13:
            status = "完整 · 亏损"
        color = C["teal"] if highlight else (C["red"] if row["seed"] in (10, 13) else C["muted"])
        s.text(1510, baseline, status, 20, color, weight=600)
        delta = row["delta"]
        s.text(1810, baseline, "—" if delta is None else f"{delta:+.2f}", 25,
               C["muted"] if delta is None else C["teal"] if delta > 0 else C["red"],
               anchor="end", cls="num")
    s.rect(80, 885, 1760, 65, C["white"], C["line"], 5)
    s.text(104, 926, "共享的是前序经验；表中每次文件差异都相对固定 Base。利润差来自各自任务，不能当作版本升级曲线。", 24)
    s.footer("利润差 = 本任务最终净利润 − Base 净利润，单位为模拟货币；缺失不记为 0，回退不算成功改进。",
             "来源：train.json 的任务顺序、各任务 summary 与实际候选文件、原生 measurement；改动不构成单规则因果归因。")
    return s.finish()


def profit_figure(data):
    report = data["report"]
    s = SVG(3, "冻结候选提高了收益，但不是每次都赢", "固定模型、固定双臂、新 seed；区间按 seed 聚类计算，不把同一 seed 的重复当作独立样本。")
    s.text(80, 265, "10 个 seed  ×  3 次重复  ×  2 个版本   =   60 次完整执行；无技术失败、无缺失配对", 24, C["muted"])
    s.rect(80, 300, 930, 530, C["white"], C["line"], 10)
    s.rect(1050, 300, 790, 530, C["pale"], radius=10)
    s.text(112, 352, "平均净利润与 95% 区间", 29, weight=600)
    s.text(112, 390, "点为均值；须线为均值区间，不是收益标准差", 20, C["muted"])
    def px(value):
        return 295 + value / 1200 * 650
    for tick in (0, 300, 600, 900, 1200):
        x = px(tick)
        s.line(x, 425, x, 690, C["line"], width=1, dash="4 6")
        s.text(x, 732, tick, 20, C["muted"], anchor="middle", cls="num")
    for arm, label, y, color in [("base", "Base", 490, C["gray"]),
                                  ("agg", "冻结候选", 635, C["teal"])]:
        mean = report["arms"][arm]["mean_net_profit"]
        low, high = data["arm_mean_95ci"][arm]
        s.text(112, y + 8, label, 25, color, weight=600)
        s.line(px(low), y, px(high), y, color, width=5)
        for value in (low, high):
            s.line(px(value), y - 12, px(value), y + 12, color, width=3)
        s.circle(px(mean), y, 10, color, C["white"], 3)
        s.text(px(mean), y - 27, f"{mean:.2f}", 34, color, anchor="middle", cls="num")
        s.text(px(mean), y + 44, f"[{low:.2f}, {high:.2f}]", 22, C["muted"], anchor="middle")
    s.text(112, 788, "横轴：模拟货币单位；最终结算后的净利润", 21, C["muted"])
    delta = report["paired"]["net_profit"]["mean_seed_delta_agg_minus_base"]
    low, high = report["paired"]["net_profit"]["seed_bootstrap_95ci"]
    s.text(1086, 352, "主要结论：候选 − Base 的配对增益", 28, weight=600)
    s.text(1086, 457, f"+{delta:.2f}", 86, C["teal"], cls="num")
    lift = delta / report["arms"]["base"]["mean_net_profit"] * 100
    s.text(1090, 526, f"相对 Base 的利润均值提高 {lift:.2f}%", 25, C["teal"])
    def dx(value):
        return 1140 + (value + 100) / 700 * 625
    s.line(dx(0), 550, dx(0), 673, C["muted"], width=2, dash="6 6")
    s.line(dx(-100), 620, dx(600), 620, C["line"], width=2)
    s.line(dx(low), 620, dx(high), 620, C["teal"], width=7)
    for value in (low, high):
        s.line(dx(value), 603, dx(value), 637, C["teal"], width=3)
    s.circle(dx(delta), 620, 12, C["teal"], C["white"], 3)
    for tick in (-100, 0, 200, 400, 600):
        s.text(dx(tick), 703, tick, 20, C["muted"], anchor="middle", cls="num")
    s.text(1086, 766, f"95% CI  [+{low:.2f}, +{high:.2f}]", 31, C["teal"], cls="num")
    for x, big, small in [(80, f'{data["wins"]} / 30', "配对中候选收益更高"),
                          (680, f'{data["positive_seed_means"]} / 10', "seed 的平均收益差为正"),
                          (1280, "30 / 30", "完整配对；没有删除失败样本")]:
        s.text(x, 900, big, 42, C["teal"], cls="num")
        s.text(x + 215, 895, small, 22)
    s.footer("2,000 次 seed 聚类 bootstrap（固定随机种子）；双臂区间为本次离线补算，配对差区间复现最终报告。",
             "来源：最终 report.json 与 60 份 results；24 小时经营窗口含后续订单结算。公开合成、同分布新 seed，不代表真实店铺或全年收益。")
    return s.finish()


def flow_figure(data):
    s = SVG(4, "最终评估的，是哪一个版本？", "把自动生成、人工工程筛选、快照冻结与独立评估分开，才能追溯最后一个数字。")
    for x, width, title, color in [(80, 675, "01  任务内自动生成", C["blue"]),
                                   (805, 280, "02  人工工程筛选", C["amber"]),
                                   (1135, 280, "03  标准化冻结", C["teal"]),
                                   (1465, 375, "04  固定策略评估", C["teal"])]:
        s.rect(x, 244, width, 49, color, radius=4)
        s.text(x + 18, 277, title, 24, C["white"], weight=600)
    s.text(80, 338, "seed15 · k=1 · 模型权重不变", 23, C["muted"])

    def node(x, y, w, h, title, lines, color=C["blue"], pale=C["white"]):
        s.rect(x, y, w, h, pale, C["line"], 8)
        s.text(x + 18, y + 39, title, 27, color, weight=600)
        s.lines(x + 18, y + 77, lines, size=21, gap=34)

    node(80, 440, 180, 110, "Base", ["固定起始提示"], C["gray"])
    node(320, 370, 190, 130, "INV", ["通用流程候选", "校验与执行规则"])
    node(320, 575, 190, 130, "DOM", ["领域经营候选", "预算与经营规则"])
    node(570, 455, 185, 150, "合成 c1", ["提示 + 经营技能", "无新增执行脚本"], C["teal"], C["pale"])
    s.path("M260 477H290V428H312", C["blue"], arrow=True)
    s.path("M260 520H290V632H312", C["blue"], arrow=True)
    s.path("M510 428H540V490H562", C["blue"], arrow=True)
    s.path("M510 632H540V565H562", C["blue"], arrow=True)
    s.text(385, 530, "探测轨迹作为合成证据", 19, C["muted"], anchor="middle")
    node(805, 415, 280, 230, "选择 seed15 / c1", ["按工程规则筛选", "不是利润最高版本", "seed16 训练利润更高", "不声称统计最优"], C["amber"], C["amber_pale"])
    node(1135, 390, 280, 280, "固定候选快照", ["有效编辑内容不变", "仅规范占位与空文件", "不重写经营策略", "此后不再修改候选"], C["teal"], C["pale"])
    s.text(1153, 645, "agg · 20d45ac8d63e55b6", 18, C["muted"], cls="mono")
    node(1465, 365, 375, 325, "Base vs 冻结候选", ["新 seed 1000–1009", "每 seed 重复 3 次", "30 对 / 60 次完整执行", "无提案、选择、反思写回", "每次使用新的经营状态"], C["teal"])
    s.line(755, 530, 797, 530, C["amber"], width=3, arrow=True)
    s.line(1085, 530, 1127, 530, C["teal"], width=3, arrow=True)
    s.line(1415, 530, 1457, 530, C["teal"], width=3, arrow=True)
    s.path("M170 550V790H1127", C["gray"], width=3, arrow=True)
    s.text(270, 772, "同一个训练 Base，独立冻结为对照臂", 24, C["muted"])
    node(1135, 735, 280, 112, "固定 Base 快照", [], C["gray"])
    s.text(1153, 819, "base · 471dfcf713d09935", 17, C["muted"], cls="mono")
    s.path("M1415 790H1652V698", C["gray"], width=3, arrow=True)
    s.rect(80, 885, 1760, 65, C["white"], C["line"], 5)
    s.text(104, 926, "k=1 时 c1 是唯一最终候选，无额外 selector 会话。冻结后的试跑与续跑未改变候选，也未继续训练。", 24)
    s.footer("源指纹与冻结指纹因标准物化不同；已核对有效可编辑内容相同。最终评估的是一个候选，不是所有训练版本。",
             "来源：seed15 summary 与实际候选、DIAGNOSIS、freeze-provenance、plan、evaluation-status；完整指纹与来源见 evidence.json。")
    return s.finish()


def index_html():
    cards = []
    for name, label in zip(NAMES, LABELS):
        cards.append(f'<section><h2>{escape(label)}</h2><div class="links">'
                     f'<a href="{name}.svg">可编辑 SVG</a>'
                     f'<a href="{name}.png">1920 × 1080 PNG</a></div>'
                     f'<img src="{name}.svg" alt="{escape(label)}" width="1920" '
                     'height="1080"></section>')
    return '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>电商 Harness RSI · 四张实验证据图</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#ebece5;color:#183134;
font-family:"PingFang SC","Hiragino Sans GB",sans-serif}main{max-width:1500px;margin:auto;
padding:48px 32px}h1{font-family:"Songti SC",serif;font-size:40px;margin:8px 0 16px}
p{line-height:1.8;max-width:1000px}a{color:#087d70;text-underline-offset:4px}
nav,.links{display:flex;gap:24px;flex-wrap:wrap}section{margin:48px 0 64px}
h2{display:inline-block;margin:0 24px 16px 0;font-size:23px}.links{display:inline-flex}
img{display:block;width:100%;height:auto;border:1px solid #cdd4ca;background:#f7f6f0}
@media(max-width:700px){main{padding:24px 12px}h1{font-size:30px}.links{display:flex;
margin-bottom:16px}}@media print{main{max-width:none;padding:0}header{display:none}
section{break-after:page;margin:0}h2,.links{display:none}img{border:0}@page{size:landscape;margin:0}}
</style><main><header><p>HARNESS RSI / MERCHANTBENCH · 演讲素材</p>
<h1>从实验过程，到可追溯的结果</h1>
<p>面向大学生的 45 分钟分享。四图均为 16:9，SVG 保留文字与矢量图形，PNG 可直接插入幻灯片。
只使用已有实验，未调用模型或改动历史运行。浅色背景适合投影，讲述顺序可采用 01 → 02 → 04 → 03。</p>
<nav><a href="README.md">口径与讲者提示</a><a href="evidence.json">脱敏证据与统计数据</a></nav>
</header>''' + "\n".join(cards) + "</main></html>\n"


def png_chunks(content):
    require(content[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG")
    offset = 8
    while offset < len(content):
        length = struct.unpack(">I", content[offset:offset + 4])[0]
        end = offset + length + 12
        require(end <= len(content), "Truncated PNG")
        kind = content[offset + 4:offset + 8]
        body = content[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", content[offset + 8 + length:end])[0]
        require(zlib.crc32(kind + body) & 0xFFFFFFFF == crc, "PNG CRC mismatch")
        yield kind, body, content[offset:end]
        offset = end
    require(offset == len(content), "Trailing PNG bytes")


def png_source_stamp(stamp):
    kind = b"tEXt"
    body = b"Source-SVG-SHA256\0" + stamp.encode("ascii")
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def check_or_stamp_png(name, *, stamp=False):
    path = HERE / f"{name}.png"
    require(path.is_file(), f"Missing PNG export: {path.name}")
    chunks = list(png_chunks(path.read_bytes()))
    require(struct.unpack(">II", chunks[0][1][:8]) == (W, H), "Wrong PNG dimensions")
    expected = hashlib.sha256((HERE / f"{name}.svg").read_bytes()).hexdigest()
    if stamp:
        result = [b"\x89PNG\r\n\x1a\n", png_source_stamp(expected)]
        # IHDR must precede all other chunks.
        result = [result[0], chunks[0][2], result[1]]
        result.extend(raw for kind, body, raw in chunks[1:]
                      if not (kind == b"tEXt" and body.startswith(b"Source-SVG-SHA256\0")))
        path.write_bytes(b"".join(result))
    else:
        stamps = [body.split(b"\0", 1)[1].decode("ascii") for kind, body, _ in chunks
                  if kind == b"tEXt" and body.startswith(b"Source-SVG-SHA256\0")]
        require(stamps == [expected], "PNG not linked to current SVG; re-export and stamp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify sources and all outputs; no writes")
    parser.add_argument("--stamp-png", action="store_true", help="Link existing browser PNGs to SVG hashes")
    args = parser.parse_args()
    require(not (args.check and args.stamp_png), "Choose one mode")
    evidence = collect()
    text = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Export an allowlist, never plan identity/profile, gateway URLs or raw logs.
    require("baseURL" not in text and "API_KEY" not in text and "10.52." not in text,
            "Private configuration leaked into exported evidence")
    outputs = {f"{name}.svg": fn(evidence) for name, fn in zip(
        NAMES, [timeline, rounds_figure, profit_figure, flow_figure])}
    outputs.update({"evidence.json": text, "index.html": index_html()})
    for filename, content in outputs.items():
        if filename.endswith(".svg"):
            ET.fromstring(content)
        path = HERE / filename
        if args.check or args.stamp_png:
            require(path.is_file() and path.read_text() == content,
                    f"Stale/missing generated file: {filename}")
        else:
            path.write_text(content)
    if args.check or args.stamp_png:
        for name in NAMES:
            check_or_stamp_png(name, stamp=args.stamp_png)
    print("PASS: native measurements, frozen trees, 10 rounds, paired report and 4 SVGs verified")
    if args.check:
        print("PASS: 4 PNG dimensions, CRCs and source-SVG hashes verified; zero files written")
    elif args.stamp_png:
        print("Stamped 4 derived PNGs; no historical experiment files written")
    else:
        print("Wrote 4 SVGs + evidence.json + index.html; PNGs require a browser export")


if __name__ == "__main__":
    main()
