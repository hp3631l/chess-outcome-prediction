"""Full interaction test for the redesigned UI + fresh report screenshots."""
from playwright.sync_api import sync_playwright


def click(pg, square):
    pg.click(f'.sq[data-square="{square}"]')
    pg.wait_for_timeout(360)


def play(pg, frm, to):
    click(pg, frm)
    click(pg, to)


def read(pg):
    return pg.evaluate(
        """() => ({
            blk: document.getElementById('v-say').textContent,
            drw: '',
            wht: '',
            lead: document.getElementById('v-lead').textContent,
            leadPct: document.getElementById('v-pct').textContent,
            rest: document.getElementById('v-rest').textContent,
            turn: document.getElementById('turntxt').textContent,
            turnCls: document.getElementById('turn').className,
            mat: document.getElementById('evalcp').textContent,
            rows: document.querySelectorAll('.sheet tbody tr').length,
            hist: document.getElementById('hist').innerText.replace(/\\s+/g,' ').trim(),
            sel: document.querySelectorAll('.sq.sel').length,
            hints: document.querySelectorAll('.hint').length,
            last: document.querySelectorAll('.sq.last').length,
            king: document.querySelectorAll('.sq.king').length,
            ann: document.getElementById('announce').textContent,
            undoDisabled: document.getElementById('btn-undo').disabled
        })"""
    )


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 880}, device_scale_factor=2)
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))

    pg.goto("http://127.0.0.1:5000/", wait_until="networkidle")
    pg.evaluate("fetch('/new',{method:'POST'})")
    pg.wait_for_timeout(400)
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(1600)

    s = read(pg)
    print(f"START   turn='{s['turn']}' mat={s['mat']} "
          f"blk {s['blk']}/drw {s['drw']}/wht {s['wht']} lead={s['lead']}")
    pg.screenshot(path="shot_1_start.png")

    # --- selection affordances ---
    click(pg, "e2")
    s = read(pg)
    print(f"SELECT  sel={s['sel']} hintDots={s['hints']} announce='{s['ann']}'")
    pg.screenshot(path="shot_2_selected.png")

    # --- a real move, check the animation ran ---
    click(pg, "e4")
    s = read(pg)
    print(f"MOVE    turn='{s['turn']}' last={s['last']} rows={s['rows']} "
          f"undoEnabled={not s['undoDisabled']}")

    # --- Scholar's mate: verify the full run ---
    pg.evaluate("fetch('/new',{method:'POST'})")
    pg.wait_for_timeout(350)
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(1400)
    for frm, to in [("e2", "e4"), ("e7", "e5"), ("f1", "c4"), ("b8", "c6"),
                    ("d1", "h5"), ("g8", "f6")]:
        play(pg, frm, to)
    s = read(pg)
    print(f"PRE-MATE {s['turn']} | blk {s['blk']}/drw {s['drw']}/wht {s['wht']} lead={s['lead']}")
    pg.screenshot(path="shot_3_midgame.png")

    play(pg, "h5", "f7")
    s = read(pg)
    print(f"MATE     turn='{s['turn']}' cls='{s['turnCls']}' "
          f"blk {s['blk']}/drw {s['drw']}/wht {s['wht']} lead={s['lead']}")
    pg.screenshot(path="shot_4_checkmate.png")

    # --- undo restores ---
    pg.click("#btn-undo")
    pg.wait_for_timeout(900)
    s = read(pg)
    print(f"UNDO     turn='{s['turn']}' rows={s['rows']} undoEnabled={not s['undoDisabled']}")

    # --- flip ---
    pg.click("#btn-flip")
    pg.wait_for_timeout(700)
    order = pg.evaluate(
        """() => Array.from(document.querySelectorAll('.sq')).slice(0,8)
                   .map(d => d.dataset.square).join(' ')"""
    )
    print(f"FLIP     top row now: {order}")
    pg.screenshot(path="shot_5_flipped.png")
    pg.click("#btn-flip")
    pg.wait_for_timeout(500)

    # --- keyboard: roving arrow focus, then select + move without a mouse ---
    pg.evaluate("fetch('/new',{method:'POST'})")
    pg.wait_for_timeout(350)
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(1400)
    pg.locator('.sq[data-square="e2"]').focus()
    pg.keyboard.press("Enter")          # select the e2 pawn
    pg.wait_for_timeout(500)
    focus_after_select = pg.evaluate("() => document.activeElement.dataset.square")
    # DOM row for rank 2 is 6, so ArrowUp (row-1) lands on rank 3 => e3
    pg.keyboard.press("ArrowUp")
    pg.wait_for_timeout(250)
    focus_target = pg.evaluate("() => document.activeElement.dataset.square")
    pg.keyboard.press("Enter")          # confirm the move
    pg.wait_for_timeout(800)
    s = read(pg)
    print(f"KEYBOARD focus after select={focus_after_select} -> arrow to {focus_target}"
          f" | rows={s['rows']} turn='{s['turn']}'")

    # --- focus ring visible ---
    pg.locator('.sq[data-square="d2"]').focus()
    pg.wait_for_timeout(200)
    ring = pg.evaluate(
        """() => { const s = getComputedStyle(document.activeElement, null);
                    return s.outlineWidth + ' ' + s.outlineStyle; }"""
    )
    print(f"FOCUS    outline on active square: {ring}")

    print(f"\nERRORS: {errs if errs else 'none'}")
    b.close()
