"""Verify the piece FLIP animation, capture fade, and the board-flip rotation.

Each scenario resets the server game AND reloads the page first: state on
the server outlives a reload, so a scenario that forgets to reset silently
plays its moves on top of the previous one.
"""
from playwright.sync_api import sync_playwright

piece_at = """(sq) => { const e = document.querySelector('.sq[data-square="' + sq + '"] .pc');
                         return e ? e.dataset.pc : null; }"""
transform_at = """(sq) => { const e = document.querySelector('.sq[data-square="' + sq + '"] .pc');
                             return e ? getComputedStyle(e).transform : 'MISSING'; }"""
squares_top = """() => Array.from(document.querySelectorAll('.sq')).slice(0, 8)
                        .map(d => d.dataset.square).join(' ')"""
inflight = """() => Array.from(document.querySelectorAll('.sq .pc'))
               .map(e => getComputedStyle(e).transform)
               .filter(t => t !== 'none' && t !== 'matrix(1, 0, 0, 1, 0, 0)')"""
hints = """() => Array.from(document.querySelectorAll('.sq .hint'))
             .map(d => d.dataset.square).join(' ')"""
board_state = """() => { const cs = getComputedStyle(document.getElementById('board'));
                         return cs.transform + ' | ' + cs.filter; }"""
at_rest = """() => { const cs = getComputedStyle(document.getElementById('board'));
                      const t = cs.transform;
                      return (t === 'none' || t === 'matrix(1, 0, 0, 1, 0, 0)')
                             && cs.filter === 'none'; }"""
all_upright = """() => Array.from(document.querySelectorAll('.sq .pc')).every(e => {
                        const t = getComputedStyle(e).transform;
                        return t === 'none' || t === 'matrix(1, 0, 0, 1, 0, 0)'; })"""

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1120, "height": 900}, device_scale_factor=2)
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))

    def fresh():
        """Reset the game server-side, then reload so the page re-reads it."""
        pg.evaluate("fetch('/new',{method:'POST'})")
        pg.wait_for_timeout(350)
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(1500)

    def move(f, t, pause=380):
        pg.click(f'.sq[data-square="{f}"]')
        pg.wait_for_timeout(260)
        pg.click(f'.sq[data-square="{t}"]')
        pg.wait_for_timeout(pause)

    def sample(n=6, gap=45):
        out = []
        for _ in range(n):
            out.append(pg.evaluate(inflight))
            pg.wait_for_timeout(gap)
        return out

    pg.goto("http://127.0.0.1:5000/", wait_until="networkidle")
    pg.wait_for_timeout(1500)

    # ---------------------------------------------------------------- 1. FLIP
    print("=== 1. piece movement (FLIP) ===")
    fresh()
    pg.click('.sq[data-square="e2"]')
    pg.wait_for_timeout(320)
    pg.click('.sq[data-square="e4"]', no_wait_after=True)
    frames = sample()
    n_mid = sum(len(f) for f in frames)
    first = next((t for f in frames for t in f), "-")
    print(f"  in-flight transform samples : {n_mid}")
    print(f"  first mid-flight transform  : {first[:64]}")
    pg.wait_for_timeout(600)
    print(f"  e2 emptied                  : {pg.locator('.sq[data-square=\"e2\"] .pc').count() == 0}")
    print(f"  e4 holds                    : {pg.evaluate(piece_at, 'e4')!r} (want 'P')")
    print(f"  transform settled           : {pg.evaluate(transform_at, 'e4')}"
          f" (want none/identity)")

    # ----------------------------------------------------------- 2. castling
    print("\n=== 2. castling animates BOTH pieces ===")
    fresh()
    for f, t in [("e2", "e4"), ("e7", "e5"), ("g1", "f3"), ("b8", "c6"),
                 ("f1", "c4"), ("f8", "c5")]:
        move(f, t)
    pg.click('.sq[data-square="e1"]')
    pg.wait_for_timeout(350)
    print(f"  legal targets from e1       : {pg.evaluate(hints)!r}")
    pg.click('.sq[data-square="g1"]', no_wait_after=True)
    frames = sample(5)
    both = max((len(f) for f in frames), default=0)
    print(f"  max pieces animating at once: {both} (want 2: K and R)")
    pg.wait_for_timeout(700)
    for sq, want in [("g1", "K"), ("f1", "R"), ("e1", None), ("h1", None)]:
        got = pg.evaluate(piece_at, sq)
        mark = "ok" if got == want else "FAIL"
        print(f"  {sq} -> {got!r:6} want {want!r:6} [{mark}]")

    # -------------------------------------------------------- 3. normal capture
    print("\n=== 3. capture fades the taken piece out ===")
    fresh()
    for f, t in [("e2", "e4"), ("d7", "d5")]:
        move(f, t)
    pg.click('.sq[data-square="e4"]')
    pg.wait_for_timeout(300)
    pg.click('.sq[data-square="d5"]', no_wait_after=True)
    dying = []
    for _ in range(6):
        dying.append(pg.locator(".pc.dying").count())
        pg.wait_for_timeout(45)
    print(f"  .pc.dying observed          : {max(dying) > 0}  samples={dying}")
    pg.wait_for_timeout(700)
    print(f"  d5 (landing) holds          : {pg.evaluate(piece_at, 'd5')!r} (want 'P')")
    print(f"  e4 (origin) emptied         : {pg.locator('.sq[data-square=\"e4\"] .pc').count() == 0}")
    print(f"  no dying elements left over : {pg.locator('.pc.dying').count() == 0}")
    print(f"  material                    : {pg.locator('#evalcp').inner_text()}")

    # ----------------------------------------------------------- 4. en passant
    print("\n=== 4. en passant (pawn dies on a square that is neither end) ===")
    fresh()
    for f, t in [("e2", "e4"), ("a7", "a6"), ("e4", "e5"), ("f7", "f5")]:
        move(f, t)
    pg.click('.sq[data-square="e5"]')
    pg.wait_for_timeout(300)
    pg.click('.sq[data-square="f6"]', no_wait_after=True)
    dying = []
    for _ in range(6):
        dying.append(pg.locator(".pc.dying").count())
        pg.wait_for_timeout(45)
    print(f"  .pc.dying observed          : {max(dying) > 0}  samples={dying}")
    pg.wait_for_timeout(700)
    print(f"  f6 (landing) holds          : {pg.evaluate(piece_at, 'f6')!r} (want 'P')")
    print(f"  f5 (captured pawn) removed  : {pg.locator('.sq[data-square=\"f5\"] .pc').count() == 0}")
    print(f"  e5 (origin) emptied         : {pg.locator('.sq[data-square=\"e5\"] .pc').count() == 0}")

    # ------------------------------------------------------------ 5. flip turn
    print("\n=== 5. board flip: 2D rotation ===")
    fresh()
    before_top = pg.evaluate(squares_top)
    before_files = pg.locator("#cfiles").inner_text().replace("\n", " ")
    before_ranks = pg.locator("#cranks").inner_text().replace("\n", " ")
    pg.click("#btn-flip")
    frames = []
    for _ in range(14):
        frames.append(pg.evaluate(board_state))
        pg.wait_for_timeout(45)
    pg.wait_for_timeout(900)

    def degs(m):
        """Pull the rotation angle out of a computed 2D matrix, else None."""
        if not m.startswith("matrix("):
            return None
        a, bb = [float(x) for x in m[7:-1].split(",")[:2]]
        return round(__import__("math").degrees(__import__("math").atan2(bb, a)), 1)

    angles = [x for x in (degs(f.split(" | ")[0]) for f in frames) if x is not None]
    is3d = any(f.startswith("matrix3d") for f in frames)
    print(f"  frames showing rotation     : {len(angles)}/{len(frames)}")
    print(f"  angles sampled              : {angles[:6]}{' ...' if len(angles) > 6 else ''}")
    print(f"  max angle reached           : {max((abs(a) for a in angles), default=0)}deg")
    print(f"  stayed in 2D (no matrix3d)  : {not is3d}")
    after_top = pg.evaluate(squares_top)
    after_files = pg.locator("#cfiles").inner_text().replace("\n", " ")
    after_ranks = pg.locator("#cranks").inner_text().replace("\n", " ")
    print(f"  top row before              : {before_top}")
    print(f"  top row after               : {after_top}")
    print(f"  orientation changed         : {before_top != after_top}")
    print(f"  files {before_files}  ->  {after_files}")
    print(f"  ranks {before_ranks}  ->  {after_ranks}")
    print(f"  coords reversed correctly   : "
          f"{after_files == 'h g f e d c b a' and after_ranks == '1 2 3 4 5 6 7 8'}")
    print(f"  settled to rest             : {pg.evaluate(at_rest)}")
    print(f"  pieces upright (no residual): {pg.evaluate(all_upright)}")

    pg.screenshot(path="/tmp/opencode/anim_flip.png", full_page=True)

    # A move must still animate correctly after a flip (the geometry changed).
    print("\n=== 5b. FLIP animation still correct after flipping ===")
    pg.click('.sq[data-square="e2"]'); pg.wait_for_timeout(300)
    pg.click('.sq[data-square="e4"]', no_wait_after=True)
    frames = sample()
    print(f"  in-flight transform samples : {sum(len(f) for f in frames)}")
    pg.wait_for_timeout(700)
    print(f"  e4 holds                    : {pg.evaluate(piece_at, 'e4')!r} (want 'P')")

    pg.click("#btn-flip"); pg.wait_for_timeout(80)
    pg.click("#btn-flip"); pg.wait_for_timeout(80)
    pg.click("#btn-flip"); pg.wait_for_timeout(1600)
    print(f"  triple-click recovers       : {pg.evaluate(at_rest)}")

    # -------------------------------------------------------- 6. reduced motion
    print("\n=== 6. prefers-reduced-motion ===")
    pg2 = b.new_page(viewport={"width": 1120, "height": 900}, reduced_motion="reduce")
    pg2.goto("http://127.0.0.1:5000/", wait_until="networkidle")
    pg2.wait_for_timeout(1400)
    pg2.click('.sq[data-square="e2"]'); pg2.wait_for_timeout(300)
    pg2.click('.sq[data-square="e4"]'); pg2.wait_for_timeout(500)
    t = pg2.evaluate(transform_at, "e4")
    at_rest_pc = t in ("none", "matrix(1, 0, 0, 1, 0, 0)")
    print(f"  piece transform after move  : {t}")
    print(f"  piece landed with no motion : {t != 'MISSING' and at_rest_pc}")
    pg2.click("#btn-flip"); pg2.wait_for_timeout(500)
    after = pg2.evaluate(squares_top)
    cs = pg2.evaluate("() => getComputedStyle(document.getElementById('board')).transform")
    print(f"  flip applied instantly      : {after.startswith('h1 ')}")
    print(f"  no transform left on board  : {cs in ('none', 'matrix(1, 0, 0, 1, 0, 0)')}")

    print(f"\nERRORS: {errs if errs else 'none'}")
    b.close()
