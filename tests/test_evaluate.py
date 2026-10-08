from maintie_lora.evaluate import (
    entity_sets,
    level,
    locate,
    macro_from,
    parse_json,
    per_text_type_counts,
    prf,
    score_text,
)


def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fence():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_strips_surrounding_text():
    assert parse_json('Sure, here is the answer:\n{"a": 1}\nHope that helps!') == {"a": 1}


def test_parse_json_invalid_returns_none():
    assert parse_json("not json at all") is None
    assert parse_json("") is None
    assert parse_json(None) is None


def test_locate_finds_first_unused_span():
    tokens = ["brg", "repl", "o/s", "pump", "p12", "noisy"]
    assert locate(tokens, ["pump", "p12"], set()) == (3, 5)


def test_locate_skips_used_span():
    tokens = ["pump", "pump", "noisy"]
    first = locate(tokens, ["pump"], set())
    assert first == (0, 1)
    assert locate(tokens, ["pump"], {first}) == (1, 2)


def test_locate_no_match():
    assert locate(["a", "b", "c"], ["z"], set()) is None


def test_level_truncates_type_path():
    assert level("PhysicalObject/SensingObject/TemperatureSensingObject", 2) == "PhysicalObject/SensingObject"


def test_prf_perfect():
    assert prf(10, 0, 0) == (1.0, 1.0, 1.0)


def test_prf_handles_zero_division():
    p, r, f1 = prf(0, 0, 0)
    assert (p, r, f1) == (0.0, 0.0, 0.0)


GOLD_REC = {
    "tokens": ["air", "conditioner", "thermostat", "not", "working"],
    "entities": [
        {"start": 0, "end": 2, "type": "PhysicalObject/EmittingObject/ElectricCoolingObject"},
        {"start": 2, "end": 3, "type": "PhysicalObject/SensingObject/TemperatureSensingObject"},
    ],
    "relations": [{"head": 0, "tail": 1, "type": "hasPart"}],
}


def test_score_text_exact_match():
    pred = {
        "entities": [
            {"text": "air conditioner", "type": "PhysicalObject/EmittingObject/ElectricCoolingObject"},
            {"text": "thermostat", "type": "PhysicalObject/SensingObject/TemperatureSensingObject"},
        ],
        "relations": [{"head": 0, "tail": 1, "type": "hasPart"}],
    }
    counts = score_text(GOLD_REC, pred)
    for name in ["entity_exact", "entity_span", "entity_coarse", "entity_two_level", "relation", "relation_strict"]:
        tp, fp, fn = counts[name]
        assert (fp, fn) == (0, 0), f"{name} should have no errors, got {counts[name]}"
        assert tp > 0


def test_score_text_unmatched_prediction_counts_as_false_positive():
    pred = {"entities": [{"text": "nonexistent span", "type": "PhysicalObject"}], "relations": []}
    counts = score_text(GOLD_REC, pred)
    tp, fp, fn = counts["entity_exact"]
    assert tp == 0
    assert fp == 1  # unplaced prediction
    assert fn == 2  # both gold entities missed


def test_score_text_handles_none_prediction():
    counts = score_text(GOLD_REC, None)
    tp, fp, fn = counts["entity_exact"]
    assert (tp, fp) == (0, 0)
    assert fn == 2


def test_entity_sets_and_per_text_type_counts():
    gold = [GOLD_REC]
    preds = {0: '{"entities": [{"text": "thermostat", "type": "PhysicalObject/SensingObject/TemperatureSensingObject"}]}'}
    g, p = entity_sets(gold[0], parse_json(preds[0]))
    assert len(g) == 2
    assert len(p) == 1

    per_text = per_text_type_counts(gold, preds, [0])
    assert len(per_text) == 1
    f1 = macro_from(per_text, [0])
    assert 0.0 <= f1 <= 1.0
