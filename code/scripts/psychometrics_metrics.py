#!/usr/bin/env python3
"""Structured psychometric language metrics (NLP Psychometrics inspired).

Objective, verifiable alternatives to a pure LLM-judge: emotional profiles,
LIWC-style psycholinguistic counts, and forma-mentis network topology extracted
from GT and predicted posts, plus distribution-level alignment scores.

These features do not depend on the generator model, so they provide evidence
against the "LLM generates + LLM judges" circularity concern.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx


# ---- built-in lexicons (compact NRC/LIWC-style seeds, expanded for coverage) --
EN_EMOTION: dict[str, set[str]] = {
    "joy": {"happy", "happiness", "glad", "joy", "joyful", "delight", "delighted", "great", "love", "loved",
            "good", "win", "won", "hope", "excited", "excitement", "proud", "pride", "amazing", "wonderful",
            "cheer", "celebrate", "celebrating", "fun", "pleased", "satisfied", "thrilled", "enjoy", "enjoyed"},
    "anger": {"angry", "anger", "outrage", "outraged", "hate", "hatred", "furious", "rage", "mad", "annoyed",
              "irritated", "shame", "absurd", "ridiculous", "fight", "attack", "attacked", "condemn",
              "condemned", "denounce", "protest", "unfair", "injustice", "scandal", "corrupt", "liar", "lie"},
    "fear": {"afraid", "fear", "fearful", "risk", "risky", "danger", "dangerous", "threat", "threatened",
             "worry", "worried", "anxiety", "anxious", "scared", "panic", "alarm", "alarming", "collapse",
             "crash", "crisis", "warn", "warning", "uncertain", "insecurity", "vulnerable"},
    "sadness": {"sad", "sadness", "sorry", "loss", "lose", "lost", "grief", "grieving", "depressed",
                "depression", "suffer", "suffering", "regret", "regretful", "pain", "painful", "miss",
                "mourn", "mourning", "tragic", "tragedy", "heartbroken", "disappointed", "disappointment",
                "worse", "bad", "terrible", "hurt"},
    "trust": {"trust", "trusted", "believe", "belief", "support", "supported", "reliable", "faith",
              "confidence", "confident", "ally", "agree", "agreement", "honest", "honesty", "credible",
              "respect", "respected", "fair", "fairness", "genuine", "truth", "true", "solid", "stable"},
    "disgust": {"disgusting", "disgust", "gross", "hate", "hated", "nasty", "repulsive", "shameful",
                "awful", "revolting", "vile", "filthy", "toxic", "sickening", "abhorrent", "contempt",
                "despise", "loathe", "disgrace", "shame"},
    "anticipation": {"expect", "expected", "plan", "planned", "will", "soon", "future", "prepare", "prepared",
                     "upcoming", "next", "anticipate", "await", "hope", "hoping", "promise", "outlook",
                     "forecast", "predict", "projection", "roadmap", "schedule", "upcoming", "approaching"},
    "surprise": {"surprising", "surprisingly", "shock", "shocked", "unexpected", "unexpectedly", "amazed",
                 "amazing", "sudden", "suddenly", "wow", "astonishing", "astonished", "stunning", "surprise",
                 "shocker", "bombshell", "unprecedented", "staggering", "remarkable"},
    "positive": {"good", "great", "best", "better", "love", "win", "won", "hope", "happy", "support",
                 "success", "successful", "proud", "progress", "improve", "improved", "gain", "growth",
                 "peace", "stable", "strong", "excellent", "positive", "benefit", "opportunity", "breakthrough",
                 "record", "victory", "thrive", "prosperity"},
    "negative": {"bad", "worst", "worse", "fail", "failed", "failure", "loss", "hate", "risk", "danger",
                 "dangerous", "wrong", "suffer", "attack", "crisis", "threat", "problem", "decline",
                 "collapse", "corruption", "scandal", "disaster", "catastrophe", "damage", "harm",
                 "destruction", "fear", "panic", "weak", "negative"},
}

ZH_EMOTION: dict[str, set[str]] = {
    "joy": {"高兴", "开心", "快乐", "喜悦", "愉快", "欣喜", "欢乐", "很好", "不错", "棒", "支持",
            "希望", "成功", "自豪", "骄傲", "喜欢", "喜爱", "热爱", "胜利", "兴奋", "激动", "振奋",
            "鼓舞", "欣慰", "满意", "满意", "庆祝", "祝贺", "点赞", "给力", "精彩", "优秀"},
    "anger": {"愤怒", "气愤", "生气", "恼火", "谴责", "强烈谴责", "荒唐", "荒谬", "可笑", "可耻",
              "丢脸", "抗议", "反对", "怒斥", "怒批", "无耻", "不要脸", "反击", "抨击", "炮轰",
              "指责", "怒怼", "愤怒", "不满", "痛斥", "抨击", "荒唐至极", "岂有此理", "欺人太甚",
              "恶劣", "丑陋", "黑幕", "腐败", "舞弊", "撒谎", "造谣", "双标"},
    "fear": {"担心", "担忧", "忧虑", "风险", "威胁", "危险", "焦虑", "不安", "害怕", "恐惧", "恐慌",
             "惧怕", "危机", "警惕", "警醒", "危机感", "不确定性", "塌方", "崩溃", "暴跌", "崩盘",
             "预警", "警示", "岌岌可危", "危在旦夕", "凶险", "隐患", "陷阱", "深渊"},
    "sadness": {"悲伤", "悲痛", "哀伤", "悲哀", "遗憾", "惋惜", "损失", "痛苦", "难过", "伤心",
                "痛心", "心痛", "痛惜", "悼念", "哀悼", "默哀", "悲恸", "凄惨", "惨痛", "惨剧",
                "悲剧", "失落", "失望", "沮丧", "心碎", "泪目", "哭泣", "哽咽", "沉重", "哀叹"},
    "trust": {"相信", "信任", "信赖", "可靠", "可信", "信心", "坚信", "坚定", "支持", "力挺", "同意",
              "赞同", "认可", "认同", "盟友", "伙伴", "真诚", "诚实", "靠谱", "公正", "公平",
              "尊重", "敬重", "可信赖", "有担当", "说到做到", "言行一致", "实事求是"},
    "disgust": {"恶心", "厌恶", "反感", "憎恶", "可耻", "龌龊", "肮脏", "丢人", "羞耻", "不堪",
                "令人作呕", "令人不齿", "鄙夷", "唾弃", "嫌弃", "倒胃口", "辣眼睛", "毁三观",
                "道德败坏", "无耻之尤", "斯文扫地"},
    "anticipation": {"预计", "预期", "展望", "将", "将会", "计划", "准备", "筹备", "未来", "即将",
                     "马上", "下一步", "前景", "预料", "有望", "待", "安排", "部署", "规划",
                     "时间表", "路线图", "预告", "倒计时", "临近", "到来", "迎来"},
    "surprise": {"意外", "惊讶", "震惊", "震撼", "突然", "竟然", "居然", "没想到", "吃惊", "诧异",
                 "惊呆", "目瞪口呆", "出乎意料", "始料未及", "大跌眼镜", "惊人", "罕见", "前所未有",
                 "史无前例", "爆炸性", "重磅", "突发", "峰回路转", "不可思议"},
    "positive": {"好", "优秀", "出色", "支持", "成功", "胜利", "希望", "伟大", "进步", "和平", "发展",
                 "稳定", "繁荣", "强", "强大", "正能量", "突破", "领先", "提升", "改善", "增长",
                 "共赢", "利好", "机遇", "前景", "光明", "坚实", "辉煌", "造福", "惠民"},
    "negative": {"坏", "差", "失败", "损失", "威胁", "风险", "错误", "危机", "恶化", "攻击", "倒退",
                 "落后", "糟糕", "严重", "灾难", "事故", "问题", "腐败", "丑闻", "黑幕", "崩盘",
                 "暴跌", "衰退", "萧条", "破坏", "伤害", "危害", "损害", "恶劣", "黑暗", "倒退"},
}

EN_LIWC: dict[str, set[str]] = {
    "pronoun_i": {"i", "me", "my", "mine", "myself", "im", "ive", "id"},
    "pronoun_we": {"we", "us", "our", "ours", "ourselves"},
    "pronoun_you": {"you", "your", "yours", "yourself"},
    "pronoun_they": {"they", "them", "their", "theirs", "he", "him", "his", "she", "her", "it", "its"},
    "negation": {"not", "no", "never", "neither", "nor", "without", "cannot", "can't", "don't", "won't",
                 "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "nothing", "none", "hardly",
                 "barely", "refuse", "deny", "denied"},
    "certainty": {"always", "never", "must", "definitely", "certainly", "surely", "absolutely", "of course",
                  "undoubtedly", "indeed", "clearly", "obviously", "guaranteed", "inevitable", "unquestionably",
                  "certain", "sure", "true", "fact", "proven", "established"},
    "tentative": {"maybe", "perhaps", "might", "could", "possibly", "likely", "seems", "probably", "maybe",
                  "suggest", "appears", "possibly", "uncertain", "unclear", "roughly", "approximately",
                  "estimate", "estimated", "arguably", "supposedly", "reportedly", "allegedly", "may", "would"},
    "quantifier": {"all", "every", "everyone", "most", "some", "many", "few", "none", "any", "anyone",
                   "each", "several", "numerous", "millions", "billions", "huge", "massive", "entire",
                   "whole", "total", "half", "double", "triple"},
    "comparative": {"more", "less", "better", "worse", "greater", "than", "higher", "lower", "bigger",
                    "smaller", "faster", "slower", "stronger", "weaker", "most", "least", "best", "worst",
                    "increase", "decrease", "rise", "fall", "surge", "plunge", "up", "down"},
    "question": {"what", "why", "how", "when", "where", "who", "which", "?", "whether", "really?"},
    "causation": {"because", "therefore", "thus", "hence", "so", "since", "lead", "led", "cause", "caused",
                  "effect", "result", "resulted", "due", "owing", "consequently", "consequence",
                  "implies", "means", "explains", "driven", "triggered"},
}

ZH_LIWC: dict[str, set[str]] = {
    "pronoun_i": {"我", "我的", "本人", "自己", "咱", "俺", "个人", "自个"},
    "pronoun_we": {"我们", "我方", "大家", "咱们", "我等", "全体", "全体成员"},
    "pronoun_you": {"你", "您", "你们", "诸位", "各位", "贵方"},
    "pronoun_they": {"他", "她", "它", "他们", "她们", "它们", "他人", "别人", "对方", "彼"},
    "negation": {"不", "没", "没有", "无", "非", "别", "未", "否", "否认", "拒绝", "并非",
                 "毫不", "绝不", "从未", "无法", "不能", "不要", "不可", "毫无"},
    "certainty": {"一定", "必然", "肯定", "绝对", "毫无疑问", "必须", "总是", "从不", "显然",
                  "无疑", "确实", "的确", "事实", "真相", "铁证", "实锤", "板上钉钉", "毋庸置疑",
                  "百分百", "毫无疑问", "确凿", "铁定"},
    "tentative": {"可能", "也许", "或许", "大概", "似乎", "估计", "应当", "或", "或者", "听说",
                  "据说", "传闻", "疑似", "大概", "大约", "差不多", "好像", "似乎", "未必",
                  "不好说", "有待", "尚不", "尚未", "暂未"},
    "quantifier": {"所有", "全部", "每", "大多数", "一些", "少数", "任何", "全体", "每个", "各位",
                   "众多", "无数", "大量", "绝大部分", "一半", "几倍", "数倍", "翻倍", "上百",
                   "上千", "上万", "千万", "亿", "多个", "诸多", "若干"},
    "comparative": {"更", "更加", "较", "比", "最", "越", "更高", "更低", "不如", "超过", "超出",
                    "领先", "落后", "提升", "下降", "增长", "减少", "上涨", "下跌", "优于",
                    "差于", "好于", "强于", "弱于", "翻倍", "腰斩", "创新高", "创新低"},
    "question": {"什么", "为什么", "如何", "怎么", "怎样", "何时", "哪里", "谁", "哪个", "哪些",
                 "是否", "吗", "呢", "？", "?", "究竟", "到底"},
    "causation": {"因为", "所以", "因此", "由于", "导致", "从而", "由此", "可见", "于是", "结果",
                  "造成", "引发", "使得", "促使", "缘于", "归因", "缘故", "因而", "致使",
                  "进而", "以至于", "后果", "根源", "原因"},
}


def _tokens(text: str) -> list[str]:
    text = (text or "").lower()
    # keep Chinese chars as single tokens, latin words as whole tokens
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    en = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return en + zh


def emotion_profile(text: str) -> dict[str, float]:
    text = text or ""
    toks = _tokens(text)
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    lex = ZH_EMOTION if zh else EN_EMOTION
    prof = {}
    for cat, words in lex.items():
        if zh:
            hits = sum(text.count(w) for w in words)
        else:
            hits = sum(1 for t in toks if t in words)
        prof[cat] = round(hits / max(len(toks), 1), 4)
    return prof


def liwc_counts(text: str) -> dict[str, float]:
    text = text or ""
    toks = _tokens(text)
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    lex = ZH_LIWC if zh else EN_LIWC
    out = {}
    for cat, words in lex.items():
        if zh:
            hits = sum(text.count(w) for w in words)
        else:
            hits = sum(1 for t in toks if t in words)
        out[cat] = round(hits / max(len(toks), 1), 4)
    return out


def forma_mentis_features(text: str, window: int = 4) -> dict[str, float]:
    """Build a word co-occurrence network and extract topology features."""
    toks = _tokens(text)
    g = nx.Graph()
    for i, w in enumerate(toks):
        g.add_node(w)
    for i in range(len(toks)):
        for j in range(i + 1, min(i + window + 1, len(toks))):
            if toks[i] != toks[j]:
                g.add_edge(toks[i], toks[j])

    n_nodes = max(g.number_of_nodes(), 1)
    n_edges = g.number_of_edges()
    density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    try:
        clustering = nx.average_clustering(g)
    except Exception:
        clustering = 0.0
    try:
        comps = list(nx.connected_components(g))
        avg_path = 0.0
        for comp in comps:
            if len(comp) > 1:
                sub = g.subgraph(comp)
                avg_path += nx.average_shortest_path_length(sub)
        avg_path = avg_path / max(len(comps), 1)
    except Exception:
        avg_path = 0.0
    # modularity via greedy communities
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(g)
        modularity = nx.algorithms.community.modularity(g, communities)
    except Exception:
        modularity = 0.0
    # semantic diversity: unique tokens ratio + type-token
    unique = len(set(toks))
    ttr = unique / n_nodes
    # emotional coherence: adjacent-token emotion-category agreement
    prof = emotion_profile(text)
    dom = max(prof, key=prof.get) if prof else ""
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))
    lex = ZH_EMOTION if zh else EN_EMOTION
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))
    all_words = set()
    for ws in lex.values():
        all_words |= ws
    if zh:
        emotional = sum(1 for w in all_words if w in (text or ""))
        coherence = round(emotional / max(n_nodes, 1), 4)
    else:
        emotional = sum(1 for t in toks if t in all_words)
        coherence = round(emotional / max(n_nodes, 1), 4)
    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": round(density, 4),
        "avg_clustering": round(clustering, 4),
        "avg_shortest_path": round(avg_path, 4),
        "modularity": round(modularity, 4),
        "type_token_ratio": round(ttr, 4),
        "emotion_coherence": coherence,
        "dominant_emotion": dom,
    }


def text_features(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update({f"emo_{k}": v for k, v in emotion_profile(text).items()})
    out.update({f"liwc_{k}": v for k, v in liwc_counts(text).items()})
    fm = forma_mentis_features(text)
    dom = fm.pop("dominant_emotion", "")
    out.update({f"net_{k}": v for k, v in fm.items() if isinstance(v, (int, float))})
    out["net_dominant_emotion_cat"] = float(list(emotion_profile(text).keys()).index(dom)) if dom else -1.0
    return out


def alignment_score(gt_text: str, pred_text: str) -> dict[str, float]:
    """Objective per-post alignment between GT and predicted text (0..1).

    The co-occurrence network topology of a single short post is nearly always
    a complete graph (density/clustering/path ~ 1), so it is not discriminative
    per post. The per-post total therefore uses emotion + style only; network
    alignment is measured at corpus level via `corpus_network_align`.
    """
    g = text_features(gt_text)
    p = text_features(pred_text)
    keys = sorted(set(g) | set(p))
    gv = [g.get(k, 0.0) for k in keys]
    pv = [p.get(k, 0.0) for k in keys]
    if not gv:
        return {"psychometric_align": 0.0, "n_features": 0}
    emo_keys = [k for k in keys if k.startswith("emo_")]
    liwc_keys = [k for k in keys if k.startswith("liwc_")]
    net_keys = [k for k in keys if k.startswith("net_")]

    def subalign(ks: list[str]) -> float:
        if not ks:
            return 1.0
        a = [g.get(k, 0.0) for k in ks]
        b = [p.get(k, 0.0) for k in ks]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 and nb == 0:
            # Identical empty profiles (both sides express nothing on these
            # dimensions). This is an identity agreement, not a signal.
            return 1.0
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 4)

    emo = subalign(emo_keys)
    sty = subalign(liwc_keys)
    # Informational only (see docstring): per-post network topology is degenerate.
    net_degenerate = (g.get("net_n_edges", 0.0) < 2.0) or (p.get("net_n_edges", 0.0) < 2.0)
    net = subalign(net_keys) if not net_degenerate else None
    overall = round(0.5 * emo + 0.5 * sty, 4)

    def coverage(features: dict[str, float], prefix: str) -> float:
        cats = [k for k in features if k.startswith(prefix)]
        if not cats:
            return 0.0
        active = sum(1 for k in cats if g.get(k, 0.0) or p.get(k, 0.0))
        return round(active / len(cats), 4)

    return {
        "emotion_align": emo,
        "style_align": sty,
        "network_align": net if net is not None else None,
        "network_degenerate": net_degenerate,
        "emotion_coverage": coverage(g, "emo_"),
        "style_coverage": coverage(g, "liwc_"),
        "psychometric_align": overall,
        "n_features": len(keys),
    }


SCALE_FREE_NET_KEYS = [
    "net_density",
    "net_avg_clustering",
    "net_avg_shortest_path",
    "net_modularity",
    "net_type_token_ratio",
    "net_emotion_coherence",
]


def corpus_network_align(gt_texts: list[str], pred_texts: list[str], max_docs: int = 300) -> dict[str, float]:
    """Compare GT-corpus vs predicted-corpus co-occurrence network topology.

    Built over aggregated user texts (rather than single posts) so the graph
    topology is informative; compares length-independent shape features.
    """
    gt = _corpus_net_features(gt_texts[:max_docs])
    pred = _corpus_net_features(pred_texts[:max_docs])
    keys = SCALE_FREE_NET_KEYS
    a = [gt.get(k, 0.0) for k in keys]
    b = [pred.get(k, 0.0) for k in keys]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = round(dot / (na * nb), 4) if na * nb else 0.0
    l1 = round(1.0 - sum(abs(x - y) for x, y in zip(a, b)) / max(len(keys), 1), 4)
    return {
        "cosine": cos,
        "l1_agreement": l1,
        "gt_features": gt,
        "pred_features": pred,
    }


def _corpus_net_features(texts: list[str], window: int = 4) -> dict[str, float]:
    toks = [t for text in texts for t in _tokens(text)]
    if len(toks) > 6000:
        toks = toks[:6000]
    g = nx.Graph()
    for w in toks:
        g.add_node(w)
    for i in range(len(toks)):
        for j in range(i + 1, min(i + window + 1, len(toks))):
            if toks[i] != toks[j]:
                g.add_edge(toks[i], toks[j])
    n_nodes = max(g.number_of_nodes(), 1)
    n_edges = g.number_of_edges()
    density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    try:
        clustering = nx.average_clustering(g)
    except Exception:
        clustering = 0.0
    try:
        comps = list(nx.connected_components(g))
        avg_path = 0.0
        for comp in comps:
            if len(comp) > 1:
                sub = g.subgraph(comp)
                if sub.number_of_nodes() <= 1500:
                    avg_path += nx.average_shortest_path_length(sub)
                else:
                    nodes = list(sub.nodes())
                    import random as _random

                    _random.Random(42).shuffle(nodes)
                    srcs = nodes[:100]
                    partial = 0.0
                    for s in srcs:
                        lengths = nx.single_source_shortest_path_length(sub, s)
                        reachable = [d for d in lengths.values() if d > 0]
                        if reachable:
                            partial += sum(reachable) / len(reachable)
                    avg_path += partial / len(srcs)
        avg_path = avg_path / max(len(comps), 1)
    except Exception:
        avg_path = 0.0
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(g)
        modularity = nx.algorithms.community.modularity(g, communities)
    except Exception:
        modularity = 0.0
    unique = len(set(toks))
    ttr = unique / n_nodes
    return {
        "net_density": round(density, 4),
        "net_avg_clustering": round(clustering, 4),
        "net_avg_shortest_path": round(avg_path, 4),
        "net_modularity": round(modularity, 4),
        "net_type_token_ratio": round(ttr, 4),
        "net_emotion_coherence": round(
            sum(1 for t in toks if any(t in ws for ws in ZH_EMOTION.values()) or any(t in ws for ws in EN_EMOTION.values()))
            / max(n_nodes, 1),
            4,
        ),
    }


def evaluate_pairs(pairs: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        gt = pair.get("gt_text") or pair.get("ground_truth") or ""
        pred = pair.get("pred_text") or pair.get("prediction") or ""
        if not gt or not pred:
            continue
        scores = alignment_score(gt, pred)
        scores["post_id"] = pair.get("post_id") or ""
        rows.append(scores)
    if not rows:
        return {"n": 0, "mean": {}, "rows": []}
    keys = ["emotion_align", "style_align", "network_align", "psychometric_align"]
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    return {"n": len(rows), "mean": mean, "rows": rows}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("psychometric_report.json"))
    args = ap.parse_args()
    pairs = json.loads(args.pairs_json.read_text(encoding="utf-8"))
    report = evaluate_pairs(pairs)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "mean": report["mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
