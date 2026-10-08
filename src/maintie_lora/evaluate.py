"""Scoring utilities for MaintIE entity/relation extraction.

This module was extracted from notebooks 05, 07 and 08, which each used to carry
their own copy of these functions. Keeping one copy means a fix (or a metric
change) only has to be made in one place instead of three.
"""
import json
import re


def parse_json(raw):
    """Parse a model answer that should be a JSON object.

    Tolerates markdown code fences or stray text around the object.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def locate(tokens, words, used):
    """First occurrence of the word sequence in tokens that is not already taken."""
    n = len(words)
    for start in range(len(tokens) - n + 1):
        if tokens[start:start + n] == words and (start, start + n) not in used:
            return (start, start + n)
    return None


def level(t, k):
    """Truncate a slash-separated detailed type to its first k levels."""
    return "/".join(t.split("/")[:k])


def prf(tp, fp, fn):
    """Precision, recall, F1 from true/false positive/negative counts."""
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, 2 * tp / max(1, 2 * tp + fp + fn)


def score_text(rec, pred):
    """Per-text TP/FP/FN counts for one gold record against one parsed prediction.

    Predicted entity text is matched back to a token span with `locate`, since
    the model is asked for the entity text rather than token indices. Returns
    counts at several granularities (exact type, span only, coarse class,
    first two type levels) plus relations (by entity span, and strictly by
    both entity span and type).
    """
    tokens = rec["tokens"]
    g_ents = [((e["start"], e["end"]), e["type"]) for e in rec["entities"]]
    g_rels = {(g_ents[r["head"]][0], g_ents[r["tail"]][0], r["type"]) for r in rec["relations"]}
    g_rels_strict = {(g_ents[r["head"]], g_ents[r["tail"]], r["type"]) for r in rec["relations"]}

    p_ents, used = [], set()
    for e in (pred or {}).get("entities", []) if isinstance(pred, dict) else []:
        words = str(e.get("text", "")).split()
        span = locate(tokens, words, used) if words else None
        if span:
            used.add(span)
        p_ents.append((span, str(e.get("type", ""))))
    p_rels, p_rels_strict = set(), set()
    for r in (pred or {}).get("relations", []) if isinstance(pred, dict) else []:
        h, t = r.get("head"), r.get("tail")
        if isinstance(h, int) and isinstance(t, int) and 0 <= h < len(p_ents) and 0 <= t < len(p_ents):
            if p_ents[h][0] and p_ents[t][0]:
                p_rels.add((p_ents[h][0], p_ents[t][0], r.get("type")))
                p_rels_strict.add((p_ents[h], p_ents[t], r.get("type")))

    counts = {}
    for name, k in [("entity_exact", None), ("entity_span", 0), ("entity_coarse", 1), ("entity_two_level", 2)]:
        to = (lambda x: (x[0], x[1])) if k is None else ((lambda x: x[0]) if k == 0 else (lambda x, k=k: (x[0], level(x[1], k))))
        G = {to(x) for x in g_ents}
        P = {to(x) for x in p_ents if x[0] is not None}
        unplaced = sum(1 for x in p_ents if x[0] is None)  # predicted text not found in the work order
        counts[name] = (len(G & P), len(P - G) + unplaced, len(G - P))
    counts["relation"] = (len(g_rels & p_rels), len(p_rels - g_rels), len(g_rels - p_rels))
    counts["relation_strict"] = (len(g_rels_strict & p_rels_strict), len(p_rels_strict - g_rels_strict),
                                  len(g_rels_strict - p_rels_strict))
    return counts


def entity_sets(rec, pred):
    """Gold and predicted (span, type) sets for one text (entities only)."""
    tokens, used, p_ents = rec["tokens"], set(), []
    for e in (pred or {}).get("entities", []) if isinstance(pred, dict) else []:
        span = locate(tokens, str(e.get("text", "")).split(), used)
        if span:
            used.add(span)
            p_ents.append((span, str(e.get("type", ""))))
    gold_ents = {((e["start"], e["end"]), e["type"]) for e in rec["entities"]}
    return gold_ents, set(p_ents)


def per_text_type_counts(gold, preds, indices):
    """Per-text, per-type entity TP/FP/FN counts, for macro-averaging over types."""
    out = []
    for i in indices:
        g, p = entity_sets(gold[i], parse_json(preds[i]))
        counts = {}
        for t in {x[1] for x in g | p}:
            gt, pt = {x for x in g if x[1] == t}, {x for x in p if x[1] == t}
            counts[t] = (len(gt & pt), len(pt - gt), len(gt - pt))
        out.append(counts)
    return out


def macro_from(per_text, selection):
    """Macro-averaged entity F1 across types, summed over a subset of texts.

    `per_text` is indexed the same way as `selection` (see `per_text_type_counts`).
    """
    totals = {}
    for i in selection:
        for t, c in per_text[i].items():
            a = totals.setdefault(t, [0, 0, 0])
            a[0] += c[0]
            a[1] += c[1]
            a[2] += c[2]
    return sum(prf(*c)[2] for c in totals.values()) / len(totals)


def load_preds(path):
    """Load a predictions jsonl file (one {"gold_idx", "raw"} per line) as a dict."""
    with open(path) as f:
        return {d["gold_idx"]: d["raw"] for d in (json.loads(line) for line in f if line.strip())}
