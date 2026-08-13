"""Integrity check for the lecture decks: every embedded figure must exist."""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def test_lecture_figures_exist():
    decks = sorted(glob.glob(os.path.join(HERE, "lectures", "L*.md")))
    assert len(decks) == 4, decks
    for md in decks:
        text = open(md).read()
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", text):
            path = os.path.normpath(os.path.join(HERE, "lectures", rel))
            assert os.path.exists(path), f"{os.path.basename(md)} references missing {rel}"
