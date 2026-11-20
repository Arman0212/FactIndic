"""Tests for the multi-annotator merge (src/eval/merge_annotations.py)."""

import math

from src.eval.merge_annotations import binary, cohen_kappa, fleiss_kappa, majority_label

LABELS = ["faithful", "minor_error", "major_error"]


def test_majority_simple():
    assert majority_label(["faithful", "faithful", "major_error"]) == ("faithful", False)


def test_majority_tie_goes_to_more_severe():
    votes = ["faithful", "faithful", "minor_error", "minor_error", "major_error"]
    assert majority_label(votes) == ("minor_error", True)


def test_fleiss_perfect_agreement():
    items = [["faithful"] * 5, ["major_error"] * 5, ["minor_error"] * 5]
    assert math.isclose(fleiss_kappa(items, LABELS), 1.0)


def test_fleiss_hand_computed():
    # 2 items, 3 raters. Item 1: all faithful (P=1). Item 2: 2 faithful, 1 major (P=1/3).
    # p_faithful = 5/6, p_major = 1/6 -> P_e = 26/36; P_bar = 2/3
    items = [["faithful"] * 3, ["faithful", "faithful", "major_error"]]
    expected = (2 / 3 - 26 / 36) / (1 - 26 / 36)  # = -0.2
    assert math.isclose(fleiss_kappa(items, LABELS), expected)


def test_cohen_kappa():
    a = ["faithful", "faithful", "major_error", "major_error"]
    b = ["faithful", "major_error", "major_error", "major_error"]
    # p_o = 3/4; p_e = (2/4)(1/4) + (2/4)(3/4) = 1/2 -> kappa = 0.5
    assert math.isclose(cohen_kappa(a, b), 0.5)
    assert math.isclose(cohen_kappa(a, a), 1.0)


def test_binary():
    assert binary(["faithful", "minor_error", "major_error"]) == ["faithful", "error", "error"]
