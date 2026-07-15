"""Coverage for scripts/validate_source.py's check_paragraph_density —
the permanent version of the one-off check that caught Out of the Abyss's
get_text("text") pagination-collapse bug (design.md's ocr_ingest.py section).
No DB/network dependency, pure function over a list of lines.
"""
from scripts.validate_source import check_paragraph_density


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_flags_a_collapsed_extraction_like_out_of_the_abyss():
    # Given a "page" of prose with no internal blank lines, repeated many
    # times with only page-boundary blank lines — the exact shape of the
    # get_text("text") pagination-collapse bug (paragraph breaks only
    # between pages, never within one)
    page = "\n".join([f"Line {i} of a page with no internal blank lines." for i in range(40)])
    text = ("\n\n".join([page] * 20))  # 20 "pages", ~800 lines, only 20 paragraphs
    issues = check_paragraph_density(_lines(text), "fake.md")

    assert len(issues) == 1
    assert issues[0].check == "low_paragraph_density"


def test_does_not_flag_healthy_prose_with_real_internal_paragraph_breaks():
    # Given normal prose with a blank line roughly every few lines (the
    # healthy shape — Curse of Strahd/Lost Mine of Phandelver both sit
    # around a 0.2+ ratio when checked this way)
    paragraph = "A short paragraph of narration text spanning a couple lines\nof real content here."
    text = "\n\n".join([paragraph] * 200)  # 200 paragraphs, ~600 lines
    issues = check_paragraph_density(_lines(text), "fake.md")

    assert issues == []


def test_skips_files_too_short_for_the_ratio_to_be_meaningful():
    # Given a very short file (well under the 200-line floor)
    text = "Just a short file.\n\nWith one blank line."
    issues = check_paragraph_density(_lines(text), "fake.md")

    assert issues == []
