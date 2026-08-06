"""Create an offline comparison of up to ten recorded final boards."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from ..replay import load_recording


def export_comparison(recordings: list[str | Path], output_path: str | Path) -> Path:
    if not 1 <= len(recordings) <= 10:
        raise ValueError("comparison supports 1 to 10 games")
    sections = []
    for path in recordings:
        document = load_recording(path)
        result = document["result"]
        players = result["final_summary"]["players"]
        rows = []
        for player in players:
            purchased = "".join(
                f"<li>{html.escape(card['card_id'])}: {html.escape(card['bonus_color'])}, {card['points']} VP</li>"
                for card in player["purchased_cards"]
            )
            reserved = (
                ", ".join(
                    f"T{card['tier']} {card['origin']} {card['bonus_color']}"
                    for card in player["reserved_cards"]
                )
                or "None"
            )
            nobles = ", ".join(card["noble_id"] for card in player["nobles"]) or "None"
            bonuses = " ".join(
                f"{color}:{count}" for color, count in player["bonuses"].items()
            )
            rows.append(
                f"<article class='player'><h3>P{player['player_id']} · Rank {player['rank']} · {player['score']} VP</h3>"
                f"<p>{player['purchased_card_count']} cards · Bonuses {html.escape(bonuses)}</p>"
                f"<p>Reserved: {html.escape(reserved)} · Nobles: {html.escape(nobles)}</p>"
                f"<details><summary>Purchased cards</summary><ul>{purchased}</ul></details></article>"
            )
        sections.append(
            f"<details open><summary><b>{html.escape(Path(path).name)}</b> · seed {document['seed']} · "
            f"{result['turns']} turns · winners {result['winner_ids']}</summary><div class='players'>{''.join(rows)}</div></details>"
        )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Splendor final boards</title><style>body{{font:14px system-ui;background:#173b3a;margin:0;padding:20px;color:#202427}}h1{{color:white}}details{{background:#f7f7f3;border-radius:9px;padding:12px;margin:10px 0}}summary{{cursor:pointer}}.players{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}}.player{{background:white;border:1px solid #9aa4a5;border-radius:7px;padding:9px}}@media(max-width:700px){{.players{{grid-template-columns:1fr}}}}</style></head><body><h1>Splendor Final Board Comparison</h1>{"".join(sections)}</body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = export_comparison(args.recordings, args.output)
    print(f"exported {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
