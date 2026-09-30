"""Convert REPORT.md to a styled HTML document (openable / printable as PDF)."""
import base64
import os
import re

import markdown

SRC = "REPORT.md"
OUT = "report.html"
ASSETS = ["figures", "."]

with open(SRC) as f:
    md_text = f.read()

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "attr_list"],
    extension_configs={"toc": {"title": "Contents"}},
)


def data_uri(path):
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "jpeg", "jpeg": "jpeg", "gif": "gif"}.get(ext, "application/octet-stream")
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


# inline every local image so the HTML is a single self-contained file
def repl(m):
    src = m.group(2)
    if src.startswith(("http://", "https://", "data:")):
        return m.group(0)
    for base in ASSETS:
        cand = os.path.join(base, src)
        if os.path.isfile(cand):
            return f'{m.group(1)}{data_uri(cand)}{m.group(3)}'
    # try basename in figures/
    cand = os.path.join("figures", os.path.basename(src))
    if os.path.isfile(cand):
        return f'{m.group(1)}{data_uri(cand)}{m.group(3)}'
    print("MISSING IMAGE:", src)
    return m.group(0)


html_body = re.sub(r'(<img\s+[^>]*src=")([^"]+)(")', repl, html_body)

CSS = """
:root { --bg:#ffffff; --fg:#1a1a2e; --muted:#5a5a72; --line:#e2e2ee;
        --accent:#4f46e5; --code:#f5f5fa; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 900px; margin: 0 auto; padding: 48px 28px 96px;
       line-height: 1.65; color: var(--fg); background: var(--bg);
       font-size: 15.5px; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 6px; letter-spacing: -.02em; }
h2 { font-size: 1.32rem; margin: 42px 0 12px; padding-bottom: 7px;
     border-bottom: 2px solid var(--line); letter-spacing: -.01em; }
h3 { font-size: 1.06rem; margin: 26px 0 8px; color: #333355; }
p { margin: 11px 0; }
a { color: var(--accent); }
code { background: var(--code); padding: 2px 6px; border-radius: 4px;
       font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .87em; }
pre { background: #16162a; color: #e6e6f0; padding: 15px 18px; border-radius: 8px;
      overflow-x: auto; line-height: 1.5; }
pre code { background: none; color: inherit; padding: 0; font-size: .85rem; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: .9rem; }
th, td { border: 1px solid var(--line); padding: 7px 11px; text-align: left; }
th { background: #f4f4fa; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafaff; }
img { max-width: 100%; display: block; margin: 20px auto; border: 1px solid var(--line);
      border-radius: 8px; }
blockquote { border-left: 3px solid var(--accent); margin: 16px 0;
             padding: 2px 0 2px 16px; color: var(--muted); }
hr { border: none; border-top: 1px solid var(--line); margin: 34px 0; }
strong { font-weight: 650; }
#toc { background: #fafaff; border: 1px solid var(--line); border-radius: 8px;
       padding: 14px 22px; margin: 26px 0 34px; font-size: .9rem; }
#toc ul { margin: 4px 0; padding-left: 20px; }
#toc > p:first-child { font-weight: 700; margin: 0 0 6px; }
@media print {
  body { padding: 0; max-width: none; font-size: 11pt; }
  h2 { page-break-after: avoid; }
  img, table, pre { page-break-inside: avoid; }
  #toc { page-break-after: always; }
}
"""

doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Real-Time Chess Outcome Prediction Using Machine Learning</title>
<style>{CSS}</style></head>
<body>
{html_body}
</body></html>"""

with open(OUT, "w") as f:
    f.write(doc)

n_img = len(re.findall(r'<img\s', doc))
size_kb = os.path.getsize(OUT) / 1024
print(f"wrote {OUT}  ({size_kb:.0f} KB, {n_img} inlined images)")
