"""Unit tests for the Phase 2 text utilities and quality checks."""

from src.data.check_hinglish import check_pair, compute_cmi, extract_numbers
from src.text_utils import normalize


def test_normalize_dashes_quotes_spaces():
    assert normalize("a\u2013b\u2014c\u2212d") == "a-b-c-d"
    assert normalize("\u201chi\u201d \u2018x\u2019") == "\"hi\" 'x'"
    assert normalize("  a \n\t b\u00a0 c ") == "a b c"


def test_extract_numbers_devanagari_and_commas():
    nums = extract_numbers("२०१४ में 1,200 लोग, सुबह 9.15 बजे, 2014.")
    assert nums == {"2014": 2, "1200": 1, "9.15": 1}


def test_check_pair_flags_added_number():
    hi = "क़रीब एक लाख बार शेयर किया गया है."
    assert check_pair(hi, "kareeb ek lakh bar share kiya gaya hai.") == []
    assert "added numbers: 1" in check_pair(hi, "kareeb 1 lakh bar share kiya gaya hai.")


def test_cmi():
    assert compute_cmi("ye viman uda tha") == 0.0                    # all Hindi
    assert compute_cmi("police ne suspect ko arrest kiya") == 50.0   # 3 en / 3 hi


def test_romanize_hindi():
    from src.data.add_roman import romanize
    # final silent 'a' dropped, anusvara -> n, nukta letters, danda -> "."
    assert romanize("विमान में ख़बर सज़ा।") == "viman men khabar saza."
    assert romanize("ऑफ़िस 9.15 बजे") == "ofis 9.15 baje"   # digits untouched
