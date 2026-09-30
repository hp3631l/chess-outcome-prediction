"""Measure rendered text and mark contrast in BOTH themes.

Backdrops are resolved by walking up to the first ancestor with a non-
transparent background, because a static list of selectors gets the backdrop
wrong whenever an element sits on a surface it does not itself paint (the
coord row, for instance, sits on the paper, not on the white board frame).
"""
import sys
from playwright.sync_api import sync_playwright

PROBE = """() => {
  const bgOf = el => { let n = el;
    while (n && n !== document.documentElement) {
      const c = getComputedStyle(n).backgroundColor;
      if (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') return c;
      n = n.parentElement; }
    return getComputedStyle(document.body).backgroundColor; };
  const txt = s => { const e = document.querySelector(s);
    return e ? [s, getComputedStyle(e).color, bgOf(e)] : null; };
  // The segment paints OVER the track, so the track is not its backdrop --
  // what surrounds the bar (the page) is.
  const seg = s => { const e = document.querySelector(s);
    return e ? [s, getComputedStyle(e).backgroundColor,
                     getComputedStyle(document.body).backgroundColor] : null; };
  const out = [txt('h1'), txt('#v-lead'), txt('#v-pct'), txt('#v-rest'),
               txt('.turnline'), txt('.meta'), txt('.sheetwrap h2'),
               txt('.sheet td'), txt('.hintline'), txt('.btn'), txt('.kbd'),
               txt('.about summary'), txt('.empty'), txt('.theme'),
               seg('.seg')];
  return out.filter(Boolean);
}"""

TEXT = ["h1", "#v-lead", "#v-pct", "#v-rest", ".turnline", ".meta",
        ".sheetwrap h2", ".sheet td", ".hintline", ".btn", ".kbd",
        ".about summary", ".empty", ".theme"]
MARKS = [".seg"]


def lum(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4
    return .2126 * f(r) + .7152 * f(g) + .0722 * f(b)


def ratio(a, b):
    x, y = lum(a), lum(b)
    hi, lo = max(x, y), min(x, y)
    return (hi + .05) / (lo + .05)


def hexof(css):
    nums = [float(x) for x in css[css.find("(") + 1:css.find(")")].replace(",", " ").split()[:3]]
    return "#%02X%02X%02X" % tuple(round(v) for v in nums)


with sync_playwright() as p:
    br = p.chromium.launch()
    res = {}
    for scheme in ("light", "dark"):
        ctx = br.new_context(color_scheme=scheme, viewport={"width": 1280, "height": 940})
        pg = ctx.new_page()
        pg.goto("http://127.0.0.1:5000/", wait_until="networkidle")
        pg.wait_for_timeout(1300)
        pg.evaluate("fetch('/new',{method:'POST'})")
        pg.wait_for_timeout(300)
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(1300)
        # Two snapshots: the empty state only exists before any move, and the
        # score sheet rows only exist after one. Merge both.
        snap = {r[0]: (r[1], r[2]) for r in pg.evaluate(PROBE)}
        for f, t in [("e2", "e4"), ("e7", "e5"), ("g1", "f3"), ("b8", "c6")]:
            pg.click(f'.sq[data-square="{f}"]'); pg.wait_for_timeout(170)
            pg.click(f'.sq[data-square="{t}"]'); pg.wait_for_timeout(250)
        snap.update({r[0]: (r[1], r[2]) for r in pg.evaluate(PROBE)})
        res[scheme] = snap
        ctx.close()

    allok = True
    for label, names, need in (("TEXT (>=4.5)", TEXT, 4.5),
                               ("BAR SEGMENTS (>=3.0)", MARKS, 3.0)):
        print(f"\n=== {label} ===")
        print(f"{'element':22} {'light':>12} {'dark':>12}  verdict")
        print("-" * 60)
        for n in names:
            if n not in res["light"] or n not in res["dark"]:
                print(f"{n:22} {'--':>12} {'--':>12}  MISSING FROM DOM")
                allok = False
                continue
            l, d = res["light"][n], res["dark"][n]
            rl = ratio(hexof(l[0]), hexof(l[1]))
            rd = ratio(hexof(d[0]), hexof(d[1]))
            ok = rl >= need and rd >= need
            allok &= ok
            print(f"{n:22} {rl:6.2f}:1 {rd:6.2f}:1  {'PASS' if ok else 'FAIL'}")

    print("\n" + ("ALL PASS IN BOTH THEMES" if allok else "*** SOME FAILURES ***"))
    br.close()
    sys.exit(0 if allok else 1)