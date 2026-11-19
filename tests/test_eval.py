"""Unit tests for the Phase 4 number and entity metrics."""

import math

from src.eval.entities import THRESHOLD, is_present, match_score, roman_candidates, skeleton, word_skeleton
from src.eval.numbers import extract_numbers, number_scores


# ---------------------------------------------------------------- numbers
def test_digits_commas_decimals_devanagari():
    assert extract_numbers("1,200 log, 9.15 baje, सन २०१४ में") == {1200, 9.15, 2014}
    assert extract_numbers("1,20,000") == {120000}


def test_number_words_hindi_and_roman():
    assert extract_numbers("तीन हज़ार आदिवासी") == {3000}
    assert extract_numbers("teen hazaar adivasi") == {3000}
    assert extract_numbers("डेढ़ करोड़") == {15_000_000}
    assert extract_numbers("1.5 lakh") == {150_000}
    assert extract_numbers("nau hazaar char sau meter") == {9400}


def test_lone_ek_is_ignored_but_ek_lakh_counts():
    assert extract_numbers("ek mahila ne kaha") == set()
    assert extract_numbers("एक महिला") == set()
    assert extract_numbers("ek lakh bar share") == {100_000}


def test_number_prf():
    s = number_scores("46 airlines ke liye 642 jet, 1995 mein", "642 jet planes 1996 mein")
    assert s["num_p"] == 0.5                 # 642 ok, 1996 hallucinated
    assert math.isclose(s["num_r"], 1 / 3)   # only 642 of {46, 642, 1995}
    assert s["num_unsupported"] == "1996"
    assert math.isnan(number_scores("koi sankhya nahi", "kuch nahi")["num_p"])


# --------------------------------------------------------------- entities
def test_word_skeleton_rules():
    assert word_skeleton("manchester") == "mnkstr"
    assert word_skeleton("police") == "pls"          # soft c -> s
    assert word_skeleton("washington") == "vsngtn"   # w -> v, sh -> s


def test_same_entity_matches_across_scripts():
    pairs = [("स्टैनस्टेड", "Stansted"), ("मैनचेस्टर", "Manchester"),
             ("एसेक्स पुलिस", "Essex police"), ("निप्पॉन एयरवेज़", "Nippon Airways"),
             ("सचिन तेंदुलकर", "Sachin Tendulkar"), ("ग्वालियर", "Gwalior")]
    for hi, en in pairs:
        assert skeleton(hi) == skeleton(en) or match_score(hi, en) >= THRESHOLD, (hi, en)


def test_romanized_hindi_matches():
    # hi_roman text is made with the same transliteration, so it must match
    assert is_present("मैनचेस्टर", "ye viman lahaur se mainachestar ja raha tha")


def test_different_entities_do_not_match():
    assert not is_present("लाहौर", "Flight London se aayi")
    assert not is_present("बिहार", "Bharat ki team")
    assert not is_present("राम", "Rome mein meeting")


def test_roman_candidates():
    c = roman_candidates("Police ne PK709 Lahore flight roki. Company ne kaha ki Boeing jeetega.")
    assert c == ["PK709 Lahore", "Boeing"]   # sentence-initial common words skipped


def test_merged_span_supported_word_by_word():
    from src.eval.entities import is_supported
    source = "डेनियल क्रेग इस फ़िल्म में मुख्य भूमिका निभाएँगे. जेम्स बॉन्ड के निर्माताओं ने कहा."
    assert is_supported("डेनियल क्रेग जेम्स बॉन्ड", source)
    assert not is_supported("जेम्स एंडरसन", "एंडरसन ने 81 रन बनाए.")   # added first name


def test_t20_matches_twenty_twenty():
    assert extract_numbers("टी20 विश्व कप") == extract_numbers("ट्वेंटी ट्वेंटी विश्व कप") == {20}


def test_sentence_windows():
    from src.eval.nli import sentence_windows
    assert sentence_windows("A b. C d. E f.") == ["A b. C d.", "C d. E f."]
    assert sentence_windows("Only one.") == ["Only one."]
