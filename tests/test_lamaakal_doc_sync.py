"""The paper-ready sentences in docs/analysis-lamaakal-style.md must be the generated ones, verbatim.

outputs/analysis-lamaakal-style-v1/paper_sentences.md is written by scripts/analyze_lamaakal_style.py
from the CSVs in the same directory. This test fails if the doc quotes a sentence that the
script did not generate, or drops one that it did.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "outputs/analysis-lamaakal-style-v1/paper_sentences.md"
DOC = ROOT / "docs/analysis-lamaakal-style.md"


def bullets(text):
    return [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]


@pytest.mark.skipif(not GENERATED.exists(), reason="analysis output not generated")
def test_every_generated_sentence_is_quoted_verbatim():
    doc = DOC.read_text(encoding="utf-8")
    missing = [line for line in bullets(GENERATED.read_text(encoding="utf-8")) if line not in doc]
    assert not missing, "Doc is missing generated sentences:\n" + "\n".join(missing)


@pytest.mark.skipif(not GENERATED.exists(), reason="analysis output not generated")
def test_doc_paper_blocks_contain_only_generated_sentences():
    generated = set(bullets(GENERATED.read_text(encoding="utf-8")))
    doc_lines = DOC.read_text(encoding="utf-8").splitlines()
    quoted, inside = [], False
    for line in doc_lines:
        if line.startswith("#### Paper-ready sentences"):
            inside = True
            continue
        if inside and line.startswith("#"):
            inside = False
        if inside and line.strip().startswith("- "):
            quoted.append(line.strip())
    assert quoted, "No paper-ready blocks found in the doc"
    extra = [line for line in quoted if line not in generated]
    assert not extra, "Doc quotes sentences the script did not generate:\n" + "\n".join(extra)
