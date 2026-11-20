# FactIndic labelling guidelines

## The task

You will see 20 news articles (Hindi in Devanagari, Hindi in Roman letters, or Hinglish), each with a short summary written by an AI model. Your job is to decide one thing: **is every claim in the summary supported by the article?** You are not judging style, grammar, spelling, language mixing or how complete the summary is.

## The one rule

**Judge only by the article, not by world knowledge.** If the summary says something that is true in real life but is not in the article, it is still an error (added_info). If the article says something you think is wrong, the article wins.

## The three labels

- **1 · faithful**: everything in the summary is stated in, or clearly follows from, the article. Leaving details out is fine. Paraphrasing is fine.
- **2 · minor error**: one small detail is wrong or not in the article, but the main message of the summary is still correct. A reader would not be seriously misled.
- **3 · major error**: the main point is wrong. A reader would come away believing something false about the main event (wrong winner, wrong person blamed, opposite meaning, invented main fact).

If there is more than one error, label the **most serious** one and mention the others in Notes.

## The six error types (only for minor or major errors)

- **entity**: a person, place or organisation is wrong, or does not appear in the article.
- **number**: a wrong number, date, amount, age, percentage or unit.
- **added_info**: a claim that is not in the article at all (even if it is true in real life).
- **meaning**: the names and numbers are right, but the relation is wrong: who did what to whom, won vs lost, supported vs opposed, cause vs effect, or a negation that was dropped or added.
- **omission**: something left out changes the meaning of what remains (for example "he was arrested" when the article says "he was arrested and released the same day"). Normal shortening is NOT an omission.
- **other**: any other factual problem (say what it is in Notes).

## Notes

For every error, quote the wrong part of the summary and say what the article says, for example: `"12 trains" - article says 21 trains`. For faithful summaries, Notes can stay empty. If you are unsure, pick your best label and write "unsure" plus why.

## Worked examples (invented, not from the set you will label)

Article: Mumbai mein Monday ko bhaari baarish ke kaaran 12 local trains cancel ho gayi. BMC ne bataya ki Andheri aur Dadar mein paani bhar gaya. Mausam vibhag ne agle do din ke liye orange alert jaari kiya hai. School band rahenge, lekin offices khule rahenge.

### Example 1: faithful

> Summary: Mumbai mein bhaari baarish se 12 local trains cancel hui aur agle do din ke liye orange alert jaari kiya gaya.

- Label: **faithful**. Every claim is in the article. Leaving out the school and office details is fine.

### Example 2: minor error

> Summary: Mumbai mein Monday ko baarish se 12 trains cancel hui. Andheri, Dadar aur Kurla mein paani bhar gaya.

- Label: **minor error**, type **added_info**. Notes: `"Kurla" - article names only Andheri and Dadar`. The main message (heavy rain, 12 trains cancelled) is still right, so it is minor.

### Example 3: major error

> Summary: Mumbai mein bhaari baarish ke kaaran schools aur offices dono band rahenge.

- Label: **major error**, type **meaning**. Notes: `"offices band rahenge" - article says offices will stay open`. The summary states the opposite of the article about a main point.

## Quick checks before you choose

- Read the whole summary, one claim at a time, and find each claim in the article.
- Check names and numbers carefully; the blue and amber highlights help you find them. A dashed red outline is only a computer hint and is often wrong.
- Watch for small words that flip meaning: not / nahi, won / lost, before / after, demanded / rejected.
- Hindi, Roman-script Hindi and Hinglish are all fine. Do not mark an error because of the language or script.
