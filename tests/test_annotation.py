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


# ---------------------------------------------------------------- A1 blind protocol
from src.eval.label_server import check_decision, compare_with_draft  # noqa: E402


def test_check_decision_rules():
    assert check_decision("faithful", "", "") == ""
    assert check_decision("faithful", "meaning", "") != ""          # faithful has no type
    assert check_decision("minor_error", "", "x") != ""             # error needs a type
    assert check_decision("major_error", "meaning", "  ") != ""     # ... and a note
    assert check_decision("major_error", "meaning", "won vs lost") == ""
    assert check_decision("", "", "") != ""


def test_compare_with_draft_is_always_blind():
    assert compare_with_draft("faithful", "", "faithful", "") == "same_as_draft_blind"
    assert compare_with_draft("minor_error", "entity", "minor_error", "meaning") == "changed_from_draft_blind"


def test_krippendorff_alpha_textbook_example():
    """Krippendorff (2011), nominal example with missing ratings: alpha = 0.743."""
    from src.eval.merge_annotations import krippendorff_alpha
    data = [[1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
            [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, None, 3, None]]
    items = [[x for x in row if x is not None] for row in data]
    assert round(krippendorff_alpha(items), 3) == 0.743
    assert krippendorff_alpha([["a", "a"], ["b", "b"]]) == 1.0
