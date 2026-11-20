"""Every GPU queue script of this project restricts its jobs to the allowed cards (never GPUs 2 or 3)."""
import re
from pathlib import Path

REMOTE = Path(__file__).resolve().parents[1] / "scripts" / "remote"


def test_allowed_cards_exclude_2_and_3():
    cards = (REMOTE / "allowed_cards").read_text().split()
    assert cards and all(c.isdigit() for c in cards) and not {"2", "3"} & set(cards)


def test_every_queue_header_names_the_allowed_cards():
    scripts = sorted(REMOTE.glob("queue_*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text()
        headers = re.findall(r"^#Q .*$", text, flags=re.MULTILINE)
        assert headers and all("cards=$CARDS" in h for h in headers), script.name
        assert 'CARDS="$(cat "$ROOT/scripts/remote/allowed_cards")"' in text, script.name
