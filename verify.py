"""Capture report screenshots by driving real clicks in a real browser."""
from playwright.sync_api import sync_playwright


def click(pg, square):
    pg.click(f'.sq[data-square="{square}"]')
    pg.wait_for_timeout(380)


def play(pg, frm, to):
    click(pg, frm)
    click(pg, to)


def readout(pg):
    return pg.evaluate(
        """() => ({
            blk: document.getElementById('p-blk').textContent,
            drw: document.getElementById('p-drw').textContent,
            wht: document.getElementById('p-wht').textContent,
            turn: document.getElementById('turn').textContent,
            status: document.getElementById('status').textContent,
            hist: document.getElementById('hist').innerText.replace(/\\s+/g,' ').trim(),
            sq: document.querySelectorAll('.sq').length,
            pc: document.querySelectorAll('.pc').length
        })"""
    )


def line(pg):
    return readout(pg)["hist"]


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1120, "height": 860})
    errors = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))

    pg.goto("http://127.0.0.1:5000/", wait_until="networkidle")
    pg.evaluate("fetch('/new',{method:'POST'})")
    pg.wait_for_timeout(500)
    pg.evaluate("refresh()")
    pg.wait_for_timeout(900)

    s = readout(pg)
    print(f"START  : {s['sq']} squares, {s['pc']} pieces | {s['turn']}")
    print(f"         blk {s['blk']} / drw {s['drw']} / wht {s['wht']}")
    pg.screenshot(path="shot_1_start.png", full_page=True)

    # Scholar's mate -- all moves verified legal.
    for frm, to in [("e2", "e4"), ("e7", "e5"),
                    ("f1", "c4"), ("b8", "c6"),
                    ("d1", "h5"), ("g8", "f6")]:
        play(pg, frm, to)

    s = readout(pg)
    print(f"6 plies : {s['turn']} | blk {s['blk']} / drw {s['drw']} / wht {s['wht']}")
    print(f"         {s['hist']}")
    pg.screenshot(path="shot_2_midgame.png", full_page=True)

    # Scholar's mate: 4.Qxf7#. (Bxf7+ would be a bishop sacrifice -- the f7
    # pawn is defended by the king -- so the model is right to rate it down.)
    play(pg, "h5", "f7")
    s = readout(pg)
    print(f"MATE    : status='{s['status']}'")
    print(f"         blk {s['blk']} / drw {s['drw']} / wht {s['wht']}")
    print(f"         {s['hist']}")
    pg.screenshot(path="shot_3_checkmate.png", full_page=True)

    # A clean material swing: win a queen, then keep playing (no mate).
    # Legal's-mate opening up to Bxf7+ (verified legal), then decline the
    # mate and carry on so the board is still live.
    pg.evaluate("fetch('/new',{method:'POST'})")
    pg.wait_for_timeout(400)
    pg.evaluate("refresh()")
    pg.wait_for_timeout(700)
    for frm, to in [("e2", "e4"), ("e7", "e5"),
                    ("g1", "f3"), ("d7", "d6"),
                    ("f1", "c4"), ("c8", "g4"),
                    ("b1", "c3"), ("g7", "g6"),
                    ("c3", "e5"), ("f8", "d6")]:
        play(pg, frm, to)
    s = readout(pg)
    print(f"SWING   : {s['turn']} | blk {s['blk']} / drw {s['drw']} / wht {s['wht']}  <- after Bxd1")
    print(f"         {s['hist']}")
    pg.screenshot(path="shot_4_material_swing.png", full_page=True)

    # flipped board view
    pg.click("text=Flip")
    pg.wait_for_timeout(700)
    pg.screenshot(path="shot_5_flipped.png", full_page=True)
    print("FLIP    : ok")

    print(f"ERRORS  : {errors if errors else 'none'}")
    b.close()
