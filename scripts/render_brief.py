"""Turn a round brief (Markdown with local figures) into one self-contained HTML file that opens on any device.

    python scripts/render_brief.py helm/rounds/R7.md runs/briefs/R7.html

The Markdown file stays the record in git; the HTML is a convenience copy with the figures embedded, so it is
not committed.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

import mistune

IMAGE = re.compile(r'<img src="([^"]+)" alt="([^"]*)"\s*/?>')
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
STYLE = """
:root { color-scheme: light; --ink: #1d1d1b; --muted: #55534e; --rule: #dedcd6; --surface: #fcfcfb; --tint: #f3f1ec; --accent: #c0392b; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --ink: #ecebe6; --muted: #b5b3ab; --rule: #3a3936;
  --surface: #1b1b1a; --tint: #262624; --accent: #e06a5c; } img { background: #fff; } }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 16px/1.75 "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", system-ui, sans-serif; }
main { max-width: 60rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
h1 { font-size: 1.7rem; line-height: 1.35; } h2 { margin-top: 2.6rem; padding-top: .8rem; border-top: 2px solid var(--rule); }
h3 { margin-top: 1.8rem; } blockquote { margin: 1rem 0; padding: .6rem 1rem; background: var(--tint); border-left: 4px solid var(--accent); }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: .45rem .6rem; vertical-align: top; text-align: left; }
th { background: var(--tint); } img { max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; }
code { font-size: .9em; background: var(--tint); padding: 0 .25em; border-radius: 3px; } a { color: inherit; }
"""


def embed_images(page: str, folder: Path) -> str:
    """Replace every local image source by a data URI; a missing figure is an error, not a broken picture."""
    def inline(match: re.Match) -> str:
        source, alt = match.group(1), match.group(2)
        if source.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (folder / source).resolve()
        media = MEDIA_TYPES.get(path.suffix.lower())
        if media is None or not path.is_file():
            raise FileNotFoundError(f"figure {source} not found next to the brief (looked for {path})")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{media};base64,{data}" alt="{alt}">'

    return IMAGE.sub(inline, page)


def render(markdown_file: Path) -> str:
    text = markdown_file.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), markdown_file.stem)
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(text)
    return (f'<!doctype html>\n<html lang="zh"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
            f"<body><main>{embed_images(body, markdown_file.parent)}</main></body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("brief", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.brief), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
