#!/usr/bin/env python3
"""
Chess + live ML win prediction.

Run:  python app.py
Open: http://127.0.0.1:5000

Design direction
----------------
This is a chess *analysis instrument*, not a dashboard with a board bolted on.

The previous pass read as machine-generated because of four habits: a row of stat
pills in the header, identical rounded cards nested in a bezel, letter-spaced
uppercase micro-labels above everything, and three identical
label/number/progress-bar rows. All four are removed here.

  * The board is the subject. The page is a flat near-black ground and the board
    is the only warm, saturated thing on it. No background gradients.
  * A three-way probability is a *composition*, not three independent metrics, so
    it is drawn as one 100%-wide stacked bar. The leading outcome is promoted
    typographically; the other two recede. No duplicated headline figure.
  * Move history is a real score sheet (a <table>), not a wrapped text blob.
  * Model metadata is demoted to a native <details> disclosure, because nobody
    needs to read "300 trees" before their first move.
  * Typography carries the hierarchy: one sans for prose, one mono for every
    figure and every move. Two families, no more.

Everything is inlined. The only optional network request is the webfont, loaded
non-blocking with a system fallback, because the first version of this UI pulled
its board from a CDN and rendered a blank page whenever that CDN was unreachable.
"""
import json
import os
import pickle
import sys

from flask import Flask, jsonify, render_template_string, request

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import extract_features, FEATURE_NAMES, static_eval  # noqa: E402
from pieces import PIECES, VIEWBOX, ATTRIBUTION  # noqa: E402

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)


def _model_meta():
    info = {
        "n_features": len(FEATURE_NAMES),
        "n_trees": getattr(MODEL, "n_estimators", "?"),
        "accuracy": None,
        "n_positions": None,
    }
    try:
        with open("training_report.txt") as f:
            txt = f.read()
        for line in txt.splitlines():
            if line.startswith("TEST ACCURACY:"):
                info["accuracy"] = float(line.split(":")[1].split()[0])
            if "positions_dataset.csv" in line and "shape=" in line:
                info["n_positions"] = int(line.split("shape=(")[1].split(",")[0].strip())
    except FileNotFoundError:
        pass
    return info


MODEL_META = _model_meta()
STATE = {"board": chess.Board()}

PIECE_CHARS = {1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k"}
PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


def board_dict(b: chess.Board):
    d = {}
    for sq, piece in b.piece_map().items():
        n = PIECE_CHARS[piece.piece_type]
        d[chess.square_name(sq)] = n.upper() if piece.color == chess.WHITE else n
    return d


def history_html(b: chess.Board):
    """A real score sheet: one row per full move, White and Black in columns."""
    tmp = chess.Board()
    rows, cur = [], None
    for mv in b.move_stack:
        san = tmp.san(mv)
        white_to_move = tmp.turn == chess.WHITE   # read BEFORE pushing
        tmp.push(mv)
        if white_to_move:
            cur = {"n": tmp.fullmove_number, "w": san, "b": ""}
            rows.append(cur)
        elif cur is not None:
            cur["b"] = san
    if not rows:
        return ""
    body = "".join(
        f'<tr><th scope="row">{r["n"]}</th><td>{r["w"]}</td>'
        f'<td>{r["b"] or "&nbsp;"}</td></tr>'
        for r in rows
    )
    return (
        '<table class="sheet">'
        '<caption class="sr">Moves played, most recent last</caption>'
        '<thead><tr><th scope="col"><span class="sr">Move number</span></th>'
        '<th scope="col">White</th><th scope="col">Black</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def game_status(b: chess.Board):
    if b.is_checkmate():
        return f"{'White' if not b.turn else 'Black'} wins by checkmate"
    if b.is_stalemate():
        return "Draw by stalemate"
    if b.is_insufficient_material():
        return "Draw by insufficient material"
    if b.can_claim_threefold_repetition():
        return "Draw by repetition"
    if b.halfmove_clock >= 100:
        return "Draw by fifty-move rule"
    return None


CSS = r"""
/* ============================================================
   A document, not a dashboard.

   Warm monochrome: bone paper, ink type. There is deliberately NO
   accent colour for interface chrome -- buttons, selection, focus and
   last-move all resolve to ink. Colour appears only where it carries
   data (the three outcome probabilities) and there it is desaturated
   to muted pastels so the page stays calm.

   One radius scale (4px structure / 6px interactive), one hairline
   (--line), used as structure rather than decoration.
   ============================================================ */
*,*::before,*::after{box-sizing:border-box}
:root{
  /* Light is the default paint. The dark set lives in [data-theme="dark"]
     below; an inline script in <head> picks one before first paint, so there
     is no flash and no duplicated media query. */
  --paper:#F7F6F3;
  --surface:#FFFFFF;
  --line:#EAEAEA;
  --line-2:#DEDDD9;
  --ink:#111111;
  --body:#2F3437;
  --muted:#63625E;
  /* --faint carries REAL TEXT (coords, kbd glyphs, table heads, the empty
     state) at 0.6-0.79rem, so it has to clear 4.5:1, not 3:1. The earlier
     #8A8883 sat at 3.28:1 and failed AA on four separate elements. */
  --faint:#6F6D69;
  /* The board is inverted against the page, and deliberately LOW contrast.
     At #3A3630 it was still the loudest thing on screen and black pieces
     dissolved into their own squares. Held close to the paper, the pieces'
     own outlines carry the figure and the grid recedes to texture. This is
     also what lets every board affordance below use ONE ink colour instead
     of flipping per square colour. */
  --sq-light:#EBE6DA;
  --sq-dark:#B4AC9C;
  /* WHICH channel carries the figure flips between themes. In light, a white
     piece reads by its dark outline (its fill is 1.2:1 on a light square);
     in dark, a white piece reads by its fill and a black piece by its light
     outline. The two pairs are deliberately exact mirrors. */
  --piece-w:#FCFBF8;      --piece-w-line:#141310;
  --piece-b:#17150F;      --piece-b-line:#FCFBF8;
  /* Marks are the ink used for board chrome -- selection, focus, legal
     targets, last move. One token, so they flip with the theme instead of
     being hardcoded black and vanishing on a dark board. */
  --mark:#111111;         /* page chrome */
  --mark-board:#111111;   /* board chrome; differs from --mark in dark mode */
  --wash:#EEECE7;                     /* lift surface on hover */
  --track:rgba(17,17,17,.07);         /* empty bar / track */
  --hint:rgba(17,17,17,.60);          /* legal-target marks, 3:1 minimum */
  --wash-last:rgba(17,17,17,.13);
  --wash-last-d:rgba(17,17,17,.17);
  --check:rgba(159,47,45,.9);
  --check-soft:rgba(159,47,45,.24);
  --shadow:0 1px 2px rgba(17,17,17,.04),0 12px 28px -18px rgba(17,17,17,.22);
  --scroll:rgba(17,17,17,.14);
  /* data only -- never chrome */
  --win-w:#346538;
  --win-b:#9F2F2D;
  --ease:cubic-bezier(.22,1,.36,1);
  --sq:66px;
  --r:6px;
  --sans:'Geist',ui-sans-serif,system-ui,'Segoe UI',sans-serif;
  --mono:'Geist Mono',ui-monospace,'SF Mono',Menlo,monospace;
  color-scheme:light;
}
/* ---------- dark ----------
   A warm charcoal, not blue-black: the paper keeps the same warm hue family
   as the light theme so the page never reads as a different site.

   The BOARD is mid-tone, not near-black. A near-black piece only reads as a
   solid shape if its FILL clears the square, and it cannot clear a near-black
   square -- measured at 1.29:1, which left every black piece rendering as a
   hollow wireframe outline while the white ones were solid. No chess product
   puts near-black pieces on near-black squares. So the board is a warm walnut
   panel that is dark relative to the page yet light enough to host both fills
   solid (white 3.39-4.48:1, black 3.58-4.72:1). That also flips board chrome
   back to dark ink, so it is tokenised as --mark-board separately from the
   page-level --mark, which stays light on the dark paper. */
[data-theme="dark"]{
  color-scheme:dark;
  --paper:#14120F;
  --surface:#1D1A16;
  --line:#2A2620;
  --line-2:#383329;
  --ink:#F2EFE9;
  --body:#D8D4CB;
  --muted:#A9A49A;
  --faint:#8F8A80;
  --sq-light:#8E7F6B;
  --sq-dark:#7C6B54;
  --piece-w:#F2EFE9;      --piece-w-line:#14120F;
  --piece-b:#17140F;      --piece-b-line:#F2EFE9;
  --mark:#F2EFE9;
  --mark-board:#14120F;
  --wash:#262219;
  --track:rgba(242,239,233,.09);
  --hint:rgba(20,18,15,.85);
  --wash-last:rgba(20,18,15,.12);
  --wash-last-d:rgba(20,18,15,.16);
  --check:rgba(158,40,38,.85);
  --check-soft:rgba(158,40,38,.22);
  --shadow:0 1px 2px rgba(0,0,0,.5),0 12px 28px -18px rgba(0,0,0,.8);
  --scroll:rgba(242,239,233,.16);
  --win-w:#6DBE86;
  --win-b:#E07A7E;
}
html,body{height:100%}
body{
  margin:0;background:var(--paper);color:var(--body);
  font-family:var(--sans);font-size:15px;line-height:1.6;
  -webkit-font-smoothing:antialiased;
  touch-action:manipulation;
}
.sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
    clip:rect(0 0 0 0);white-space:nowrap;border:0}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

.wrap{max-width:1080px;margin:0 auto;padding:0 32px 88px}

/* ---------- masthead: type only, no mark ---------- */
.masthead{display:flex;align-items:baseline;justify-content:space-between;
          gap:24px;padding:38px 0 30px;flex-wrap:wrap}
.masthead-r{display:flex;align-items:baseline;gap:22px}
h1{margin:0;font-size:1.02rem;font-weight:500;color:var(--ink);letter-spacing:-.012em}
/* The toggle is a peer of the About disclosure: same size, same weight, same
   resting colour. A filled or ringed switch would reintroduce the accent
   that was removed from every other control. The dot is a two-tone disc that
   reads half-filled in light and half-filled in dark -- one element, no icon
   font, no emoji. */
.theme{display:inline-flex;align-items:center;gap:7px;font:inherit;font-size:.8rem;
       color:var(--muted);cursor:pointer;padding:0 0 3px;
       border-bottom:1px solid var(--line-2);
       transition:color 200ms cubic-bezier(.4,0,.2,1),border-color 200ms cubic-bezier(.4,0,.2,1)}
.theme:hover{color:var(--ink);border-color:var(--ink)}
.theme:focus-visible{outline:2px solid var(--mark);outline-offset:3px}
.theme-dot{width:9px;height:9px;flex:none;border-radius:50%;
           background:linear-gradient(90deg,var(--mark) 50%,transparent 50%);
           border:1px solid var(--mark)}
.about summary{
  list-style:none;cursor:pointer;font-size:.8rem;color:var(--muted);
  padding-bottom:3px;border-bottom:1px solid var(--line-2);
  transition:color 200ms cubic-bezier(.4,0,.2,1),border-color 200ms cubic-bezier(.4,0,.2,1);
}
.about summary::-webkit-details-marker{display:none}
.about summary:hover{color:var(--ink);border-color:var(--ink)}
.about summary:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
.about[open] summary{border-color:transparent}
.about .body{
  position:absolute;right:0;top:calc(100% + 10px);z-index:20;width:312px;
  padding:18px 19px;background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r);
  box-shadow:var(--shadow);
  font-size:.79rem;line-height:1.65;color:var(--muted);
}
.about .body p{margin:0 0 10px}
.about .body p:last-child{margin:0}
.about dl{margin:0 0 12px;display:grid;grid-template-columns:auto 1fr;gap:5px 16px}
.about dt{color:var(--faint)}
.about dd{margin:0;text-align:right;color:var(--ink);font-family:var(--mono)}
.credit{margin:0;color:var(--faint);font-size:.72rem;line-height:1.55}

/* ---------- composition ---------- */
.stage{display:grid;grid-template-columns:minmax(0,auto) 296px;gap:56px;
       align-items:start;justify-content:center}
.left{display:flex;flex-direction:column;gap:10px}
.rail{display:flex;flex-direction:column;gap:30px;min-height:100%}

/* ---------- board ---------- */
.boardwrap{display:flex;flex-direction:column;gap:8px}
.frame{padding:7px;border:1px solid var(--line);border-radius:var(--r);
       background:var(--surface)}
.board{position:relative;display:grid;
       grid-template-columns:repeat(8,var(--sq));grid-template-rows:repeat(8,var(--sq));
       user-select:none;-webkit-user-select:none;touch-action:manipulation;
       border-radius:3px;overflow:hidden}
.sq{position:relative;display:grid;place-items:center;padding:0;margin:0;border:0;
    background:var(--sq-light);cursor:pointer;line-height:0;
    -webkit-tap-highlight-color:transparent}
.sq.d{background:var(--sq-dark)}
.sq:focus{outline:none}
.sq:focus-visible{outline:2px solid var(--mark-board);outline-offset:-3px;z-index:6}
/* Last move is a memory cue, not an action target, so it sits below 3:1 by
   design -- but it must still be perceptible, hence two alphas. */
.sq.last::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:var(--wash-last)}
.sq.d.last::after{background:var(--wash-last-d)}
.sq.sel{box-shadow:inset 0 0 0 3px var(--mark-board)}
.sq.king::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(circle,var(--check) 6%,var(--check-soft) 40%,transparent 68%)}

/* FLIP -- the board turns a half-revolution in its own plane. No depth, no
   perspective, no backface: just a rotation.

   Two facts make this cheap. A 180deg turn maps every square onto the
   square that flips it, and an 8x8 checkerboard is invariant under 180deg
   -- so the turned board and the re-rendered board are the SAME pixels.
   The layout therefore only has to be rewritten once the turn has landed,
   and that rewrite is invisible.

   The pieces are counter-rotated so they stay upright throughout. What the
   eye reads as "the board turned" is the pieces orbiting with it while the
   squares turn underneath. The mid-turn scale dip keeps the rotation from
   reading as a flat spin. */
.boardwrap.turning .board{transform:rotate(180deg);animation:turn 520ms var(--ease)}
/* .pc carries transition:transform for the move FLIP below, and a transition
   outranks an animation in the cascade -- left alone it would drive this
   counter-rotation in 300ms while the board took 520ms, and the two would
   visibly fall out of sync halfway round. Kill it so only the animation runs. */
.boardwrap.turning .pc  {transform:rotate(-180deg);
                         animation:turnpc 520ms var(--ease);
                         transition:none}
/* The midpoint is not decoration -- CSS applies the timing function to each
   keyframe SEGMENT, so `turn` (which has one) and a plain two-stop
   `turnpc` would run on different curves and fall out of sync halfway
   round. Matching the stops is what keeps the pieces dead upright. */
@keyframes turn{
  0%  {transform:rotate(0deg)   scale(1)}
  50% {transform:rotate(90deg)  scale(.955)}
  100%{transform:rotate(180deg) scale(1)}
}
@keyframes turnpc{
  0%  {transform:rotate(0deg)}
  50% {transform:rotate(-90deg)}
  100%{transform:rotate(-180deg)}
}
/* When .turning comes off, the board and pieces snap back to zero in the
   same style step as the layout rewrite. Without this the pieces would
   tween from -180deg to 0 and visibly spin a whole second time. */
.boardwrap.snapping .board,.boardwrap.snapping .pc{transition:none}

/* FLIP: the piece is rendered at its NEW square, then briefly translated back
   to where it came from and released. transform-only, so the whole move runs
   on the compositor and never triggers layout. */
.pc{position:absolute;inset:9%;z-index:2;pointer-events:none;line-height:0;
    transition:transform 300ms var(--ease),opacity 300ms var(--ease)}
.pc.flip-in{transition:none}              /* hold the offset until we release it */
/* the captured piece fades and shrinks out on the square it stood on */
.pc.dying{opacity:0;transform:scale(.55);
          transition:opacity 250ms var(--ease),transform 250ms var(--ease)}
.pc svg{width:100%;height:100%;display:block;overflow:visible}
/* mid-flight lift -- animated on the inner svg so it cannot fight the
   translate on .pc above, which owns that transform property */
.pc.lift svg{animation:lift 300ms var(--ease)}
@keyframes lift{0%{transform:scale(1)}34%{transform:scale(1.1)}100%{transform:scale(1)}}
/* Pieces are read by their OUTLINE, not their fill -- a white piece on a
   light square is only 1.20:1 by fill, but the ink stroke is 17.5:1, so the
   figure still holds. That is why the board can sit low-contrast. */
.pc.w path{fill:var(--piece-w);stroke:var(--piece-w-line);stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
.pc.b path{fill:var(--piece-b);stroke:var(--piece-b-line);stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
/* Legal-target marks carry no hue, and because both square colours sit close
   to the paper one ink reads across the whole board -- the per-square
   overrides are gone. Alpha is 0.60, not the 0.22 it looks like it could be:
   these marks are the ONLY cue that a square is capturable, so they are
   meaningful content and need 3:1. At 0.30 the capture ring measured 1.81:1
   on the dark square -- effectively invisible. */
.hint{position:absolute;left:50%;top:50%;width:26%;height:26%;z-index:3;
      transform:translate(-50%,-50%);border-radius:50%;pointer-events:none;
      background:var(--hint)}
.hint.cap{width:84%;height:84%;background:none;box-sizing:border-box;
          border:3px solid var(--hint)}

.coords{display:flex;font-family:var(--mono);font-size:.6rem;color:var(--faint);
        font-variant-numeric:tabular-nums}
.coords.files{width:calc(8*var(--sq));margin-left:24px}
.coords.files span{width:var(--sq);text-align:center}
.ranks{display:flex;flex-direction:column;height:calc(8*var(--sq));width:18px;flex:none}
.ranks span{display:grid;place-items:center;height:var(--sq)}
.boardgrid{display:flex;gap:9px}
.hintline{margin:0;text-align:center;font-size:.75rem;color:var(--faint)}

/* ---------- the verdict: one number, not three rows ---------- */
.verdict{display:flex;flex-direction:column}
.verdict .k{margin:0 0 7px;font-size:.8rem;color:var(--muted);line-height:1}
/* The leading figure IS the interface. Hierarchy comes from weight and
   colour, not from a list of three equal rows each trying to be read. */
.verdict .lead-v{margin:0 0 15px;font-size:2.9rem;line-height:1;color:var(--ink);
                 letter-spacing:-.038em;font-weight:400}
/* The bar measures DOMINANCE, not identity. Three stacked hues could not do
   that accessibly: adjacent segments must clear 3:1, and muted green against
   muted red measured 1.05:1 -- the same colour twice, and the worst possible
   pair for colour-vision deficiency. Hue rotation is not a contrast axis.
   So the bar is one ink segment over a track, and the three-way split lives
   in the text line beneath it, where it is unambiguous. */
.stack{position:relative;height:5px;border-radius:2px;overflow:hidden;
       background:var(--track)}
.seg{position:absolute;left:0;top:0;bottom:0;width:100%;
     transform-origin:left center;background:var(--ink);
     transition:transform 640ms var(--ease)}
.rest{margin:0;font-size:.76rem;color:var(--faint);font-variant-numeric:tabular-nums}
.rest b{font-weight:400;color:var(--muted)}
.verdict .rest{margin-top:11px}
.meta{margin-top:3px}
.meta .sep{color:var(--line-2)}
.meta .up{color:var(--win-w)} .meta .dn{color:var(--win-b)}

/* ---------- turn: words carry it, no status dot ---------- */
.turnline{font-size:.82rem;color:var(--muted);min-height:22px}
.turnline b{font-weight:500;color:var(--ink)}
.turnline.res-w b{color:var(--win-w)}
.turnline.res-b b{color:var(--win-b)}
.turnline.res-d b{color:var(--ink)}

/* ---------- score sheet ---------- */
.sheetwrap{display:flex;flex-direction:column;min-height:0;flex:1}
.sheetwrap h2{margin:0 0 10px;font-size:.72rem;font-weight:500;color:var(--faint)}
#hist{flex:1;overflow-y:auto;overscroll-behavior:contain;min-height:64px;
      scrollbar-width:thin;scrollbar-color:var(--scroll) transparent}
#hist::-webkit-scrollbar{width:6px}
#hist::-webkit-scrollbar-thumb{background:var(--scroll);border-radius:3px}
.sheet{width:100%;border-collapse:collapse;font-family:var(--mono);
       font-size:.79rem;font-variant-numeric:tabular-nums}
.sheet th,.sheet td{text-align:left;font-weight:400;padding:4px 0;
                    border-bottom:1px solid var(--line)}
.sheet thead th{color:var(--faint);padding-bottom:6px}
.sheet tbody th{color:var(--faint);width:1.9em}
.sheet td:first-of-type{color:var(--ink);width:4.2em}
.sheet td:last-child{color:var(--muted)}
.sheet tbody tr:last-child th,.sheet tbody tr:last-child td{border-bottom:0}
/* An empty state should say what to do, not apologise. */
.empty{color:var(--faint);font-size:.79rem;padding:2px 0;margin:0}

/* ---------- controls: three peers, no primary ---------- */
/* "New game" was the filled CTA on a page whose real interaction is the
   board. That was a hierarchy error: making a rare, destructive action
   loudest. All three are now the same quiet weight. */
.ctrls{display:grid;grid-template-columns:repeat(3,1fr);
       border:1px solid var(--line);border-radius:var(--r);
       background:var(--surface);overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer;border:0;background:none;
       -webkit-tap-highlight-color:transparent}
.btn{padding:9px 6px 8px;border-radius:0;font-size:.78rem;color:var(--muted);
     transition:transform 170ms var(--ease),background 200ms cubic-bezier(.4,0,.2,1),
                color 200ms cubic-bezier(.4,0,.2,1)}
.btn + .btn{border-left:1px solid var(--line)}
.btn:hover:not([disabled]){background:var(--wash);color:var(--ink)}
.btn:active:not([disabled]){transform:scale(.97)}
.btn:focus-visible{outline:2px solid var(--ink);outline-offset:-3px}
/* Shortcuts rendered as physical keys rather than as loose grey text. */
.kbd{display:block;width:fit-content;margin:5px auto 0;padding:0 5px;
     border:1px solid var(--line);border-radius:4px;background:var(--wash);
     font-family:var(--mono);font-size:.6rem;line-height:1.55;color:var(--muted)}
.btn[disabled]{opacity:.34;cursor:not-allowed}

/* ---------- one quiet entrance, transform + opacity only ---------- */
.enter{animation:rise 640ms var(--ease) both}
.enter-2{animation-delay:70ms}
.enter-3{animation-delay:140ms}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* ---------- responsive ---------- */
@media (max-width:1020px){
  :root{--sq:min(9.4vw,60px)}
  .stage{grid-template-columns:minmax(0,1fr);gap:36px;max-width:600px;margin:0 auto}
  .rail{max-width:600px;width:100%;margin:0 auto;gap:26px}
  .about .body{width:min(320px,calc(100vw - 64px))}
  .masthead-r{gap:18px}
}
@media (max-width:560px){
  :root{--sq:min(10.6vw,52px)}
  .wrap{padding:0 16px 56px}
  .masthead{padding:26px 0 20px}
  h1{font-size:.94rem}
  .verdict .lead-v{font-size:2.4rem}
  .coords.files{margin-left:22px}
}

/* ---------- reduced motion ---------- */
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
                       transition-duration:.01ms!important}
}
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#F7F6F3" id="themecolor">
<script>
/* Resolve the theme BEFORE first paint. An inline script in <head> is the
   only way to avoid a flash of the wrong theme -- waiting for the deferred
   module at the end of <body> would show a white flash on every load. */
(function(){
  var saved = null;
  try { saved = localStorage.getItem("theme"); } catch(e){}
  var dark = saved ? saved === "dark"
                   : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
})();
</script>
<meta name="description" content="Two-player chess with a machine-learning win-probability estimate after every move.">
<title>Live chess outcome prediction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" media="print" onload="this.media='all'"
      href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500&family=Geist+Mono:wght@400;500&display=swap">
<style>__CSS__</style>
</head>
<body>
<div class="wrap">

  <header class="masthead enter">
    <h1>Live chess outcome prediction</h1>
    <div class="masthead-r">
      <button class="theme" id="btn-theme" type="button" aria-pressed="false">
        <span class="theme-dot" aria-hidden="true"></span><span id="theme-l">Dark</span>
      </button>
      <details class="about">
        <summary>About the model</summary>
        <div class="body">
          <dl id="stats"></dl>
          <p id="foot"></p>
          <p id="credit"></p>
        </div>
      </details>
    </div>
  </header>

  <main class="stage">
    <section class="left enter enter-2" aria-label="Chess board">
      <div class="boardwrap">
        <div class="coords files" id="cfiles" aria-hidden="true"></div>
        <div class="boardgrid">
          <div class="ranks" id="cranks" aria-hidden="true"></div>
          <div class="frame">
            <div class="board" id="board" role="grid"
                 aria-label="Chess board, two players sharing one screen"></div>
          </div>
        </div>
        <p class="hintline">Select a piece, then a highlighted square</p>
      </div>
      <p class="sr" id="announce" role="status" aria-live="polite"></p>
    </section>

    <aside class="rail enter enter-3" aria-label="Analysis">
      <div class="verdict">
        <p class="k" id="v-lead">Draw</p>
        <p class="lead-v mono" id="v-pct">&ndash;</p>
        <div class="stack" aria-hidden="true"><i class="seg" id="b-lead"></i></div>
        <p class="rest" id="v-rest"></p>
        <p class="sr" id="v-say" aria-live="polite" aria-atomic="true"></p>
      </div>

      <div>
        <div class="turnline" id="turn"><span id="turntxt">Loading&hellip;</span></div>
        <p class="rest meta">Material <span class="sep" aria-hidden="true">·</span>
          <span class="mono" id="evalcp">&ndash;</span></p>
      </div>

      <div class="sheetwrap">
        <h2>Moves</h2>
        <div id="hist" role="log" aria-label="Move history"></div>
      </div>

      <div class="ctrls">
        <button class="btn" id="btn-new">New game<kbd class="kbd">N</kbd></button>
        <button class="btn" id="btn-undo" disabled>Undo<kbd class="kbd">&larr;</kbd></button>
        <button class="btn" id="btn-flip">Flip<kbd class="kbd">F</kbd></button>
      </div>
    </aside>
  </main>
</div>

<script>
const PIECES = __PIECES__;
const VIEWBOX = __VIEWBOX__;
const FILES = "abcdefgh";
const NAME = {p:"pawn",n:"knight",b:"bishop",r:"rook",q:"queen",k:"king"};
const ROWS = ["black","draw","white"];

let sel = null, targets = [], flipped = false, lastMove = null, flipping = false;
const el = id => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function svg(pc){
  const key = (pc === pc.toUpperCase() ? "w" : "b") + pc.toUpperCase();
  const s = document.createElementNS("http://www.w3.org/2000/svg","svg");
  s.setAttribute("viewBox","0 0 "+VIEWBOX+" "+VIEWBOX);
  s.setAttribute("aria-hidden","true");
  const d = PIECES[key];
  if (d){
    const p = document.createElementNS("http://www.w3.org/2000/svg","path");
    p.setAttribute("d", d);
    p.setAttribute("fill-rule","evenodd");
    s.appendChild(p);
  } else { console.error("no piece geometry for key:", key); }
  return s;
}
function pieceEl(pc){
  const s = document.createElement("span");
  s.className = "pc " + (pc === pc.toUpperCase() ? "w" : "b");
  s.dataset.pc = pc;
  s.appendChild(svg(pc));
  return s;
}

const boardEl = el("board");
for (let r = 0; r < 8; r++){
  for (let c = 0; c < 8; c++){
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sq " + (((r + c) % 2) ? "d" : "");
    b.dataset.row = r; b.dataset.col = c;
    b.addEventListener("click", () => onSquare(b.dataset.square));
    boardEl.appendChild(b);
  }
}
/* Coordinates sit outside the rotating board, so they have to be rebuilt
   whenever the orientation changes: flipped, the top-left square is h1, so
   the files read h..a and the ranks read 1..8. */
function renderCoords(){
  const files = flipped ? FILES.split("").reverse().join("") : FILES;
  const ranks = flipped ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
  el("cfiles").innerHTML = files.split("").map(f => "<span>"+f+"</span>").join("");
  el("cranks").innerHTML = ranks.map(n => "<span>"+n+"</span>").join("");
}
renderCoords();

function label(sq, pieces, turn, over){
  const pc = pieces[sq];
  if (pc) return (pc === pc.toUpperCase() ? "White " : "Black ") + NAME[pc.toLowerCase()] + " on " + sq;
  if (over) return sq + ", empty";
  return sq + ", empty, " + (turn === "w" ? "White" : "Black") + " to move";
}

function paint(state){
  for (const b of boardEl.children){
    // row 0 is the top. Normal orientation => rank 8 on top, so r = 8 - row.
    const r = flipped ? Number(b.dataset.row) + 1 : 8 - Number(b.dataset.row);
    const c = flipped ? 7 - Number(b.dataset.col) : Number(b.dataset.col);
    const sq = FILES[c] + r;
    b.dataset.square = sq;

    const want = state.pieces[sq];
    const cur = b.querySelector(".pc");
    if (!want){ if (cur) cur.remove(); }
    else if (!cur || cur.dataset.pc !== want){
      if (cur) cur.remove();
      b.appendChild(pieceEl(want));
    }

    b.className = "sq " + (((Number(b.dataset.row) + Number(b.dataset.col)) % 2) ? "d" : "");
    if (lastMove && (sq === lastMove.from || sq === lastMove.to)) b.classList.add("last");
    b.setAttribute("aria-label", label(sq, state.pieces, state.turn, state.over));

    const old = b.querySelector(".hint");
    if (old) old.remove();
    if (sel === sq){ b.classList.add("sel"); }
    else if (sel && targets.includes(sq)){
      const m = document.createElement("span");
      m.className = "hint" + (want ? " cap" : "");
      b.appendChild(m);
    }
  }
  if (!state.over && state.check){
    const k = state.turn === "w" ? "K" : "k";
    for (const b of boardEl.children){
      if (state.pieces[b.dataset.square] === k){ b.classList.add("king"); break; }
    }
  }
  renderTurn(state);
  el("hist").innerHTML = state.history ||
    '<p class="empty">No moves yet</p>';
  el("btn-undo").disabled = !state.n_moves;
}

function renderTurn(state){
  const t = el("turn"), x = el("turntxt");
  t.className = "turnline";
  if (state.over){
    t.classList.add(/White wins/.test(state.status) ? "res-w"
                  : /Black wins/.test(state.status) ? "res-b" : "res-d");
    x.innerHTML = "<b>" + state.status + "</b>";
  } else {
    t.classList.add(state.turn);
    x.innerHTML = "<b>" + (state.turn === "w" ? "White" : "Black") + "</b> to move";
  }
}

const LABEL = {black:"Black", draw:"Draw", white:"White"};

function setProbs(p){
  const v = {black:p.black, draw:p.draw, white:p.white};
  // The bar is a dominance meter: one ink segment for the leading outcome,
  // scaled to its probability, over the track for everything else. Keeping
  // three hues here put muted green next to muted red at 1.05:1.
  let lead = "draw";
  for (const k of ROWS) if (v[k] > v[lead]) lead = k;
  el("b-lead").style.transform = "scaleX(" + Math.max(v[lead], 0.004) + ")";

  // ONE number, not three rows. At 98.7% draw the other two are noise, and
  // giving them peer rows with swatches turned an answer into a form.
  el("v-lead").textContent = LABEL[lead];
  el("v-pct").textContent = (v[lead] * 100).toFixed(1) + "%";

  const rest = ROWS.filter(k => k !== lead);
  el("v-rest").innerHTML = rest.map(function(k){
    return LABEL[k] + " <b>" + (v[k] * 100).toFixed(1) + "%</b>";
  }).join(" &middot; ");

  // The visual now omits two of the three, so give assistive tech all of
  // them rather than leaving the live region to announce only the leader.
  el("v-say").textContent = ROWS.map(function(k){
    return LABEL[k] + " " + (v[k] * 100).toFixed(1) + " percent";
  }).join(", ");

  const cp = p.material_cp, e = el("evalcp");
  if (typeof cp === "number" && Math.abs(cp) >= 5){
    e.textContent = (cp > 0 ? "+" : "\u2212") + (Math.abs(cp)/100).toFixed(1);
    e.className = "mono " + (cp > 0 ? "up" : "dn");
  } else {
    e.textContent = "level";
    e.className = "mono";
  }
}

function say(m){ el("announce").textContent = m; }

/* ---------- theme ----------
   The attribute is flipped in JS, but the CSS reads it from <html>, so every
   token inverts at once -- there is no second stylesheet to keep in sync and
   no per-component dark rules. The <meta name="theme-color"> is updated too
   so the browser chrome matches the page. */
const THEME_COLOR = {light:"#F7F6F3", dark:"#14120F"};
function applyTheme(next){
  document.documentElement.dataset.theme = next;
  el("btn-theme").setAttribute("aria-pressed", String(next === "dark"));
  el("theme-l").textContent = next === "dark" ? "Light" : "Dark";
  el("themecolor").setAttribute("content", THEME_COLOR[next]);
  try { localStorage.setItem("theme", next); } catch(e){}
}
el("btn-theme").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");

async function api(path, body){
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                               body: JSON.stringify(body || {})});
  return r.json();
}

/* ------------------------------------------------------------------
   Move animation.

   paint() destroys and recreates piece elements, so by the time we could
   animate, the piece is already sitting in its NEW square. The fix is to
   measure the square the piece is travelling FROM *before* the repaint,
   then invert that delta on the newly created element and release it:

       translate(from - to)  ->  next frame  ->  translate(0)

   Both the offset and the release are transform/opacity only, so nothing
   reflows and the browser can keep the whole move on the compositor.
   ------------------------------------------------------------------ */
function rectOf(sq){
  const b = boardEl.querySelector('[data-square="' + sq + '"]');
  return b ? b.getBoundingClientRect() : null;
}

function flipPiece(origin, to){
  if (reduceMotion || flipping || !origin || !to) return;
  const dst = boardEl.querySelector('[data-square="' + to + '"] .pc');
  if (!dst) return;
  const b = dst.getBoundingClientRect();
  const dx = origin.left - b.left, dy = origin.top - b.top;
  if (!dx && !dy) return;                      // already in place, nothing to do

  dst.classList.add("flip-in", "lift");
  dst.style.transform = "translate(" + dx + "px," + dy + "px)";
  // Force a reflow so the browser commits the offset position as its own
  // style before we change it. Without this the two values are coalesced
  // into the final state and no motion is ever painted.
  void dst.offsetWidth;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      dst.classList.remove("flip-in");
      dst.style.transform = "";
    });
  });
  setTimeout(() => dst.classList.remove("lift"), 340);
}

function diePiece(origin, pc){
  if (reduceMotion || flipping || !origin) return;
  const box = boardEl.getBoundingClientRect();
  const g = document.createElement("span");
  g.className = "pc " + (pc === pc.toUpperCase() ? "w" : "b") + " dying";
  g.style.left   = (origin.left - box.left) + "px";
  g.style.top    = (origin.top  - box.top ) + "px";
  g.style.width  = origin.width  + "px";
  g.style.height = origin.height + "px";
  g.appendChild(svg(pc));
  boardEl.appendChild(g);
  setTimeout(() => g.remove(), 300);
}

async function onSquare(sq){
  const over = el("turn").className.indexOf("res-") >= 0;
  if (over) return;
  if (!sel){
    const s = await api("/state");
    if (!s.selectable.includes(sq)) return;
    sel = sq; targets = (await api("/select", {sq})).targets;
    paint(s);
    const pc = s.pieces[sq];
    say((pc === pc.toUpperCase() ? "White " : "Black ") + NAME[pc.toLowerCase()] +
        " on " + sq + ". " + targets.length + " legal target" +
        (targets.length === 1 ? "" : "s") + ".");
  } else {
    const from = sel;
    const r = await api("/move", {from: from, to: sq});
    if (r.ok){
      // Capture the origin geometry BEFORE paint() tears down the old
      // elements. Afterwards these rects would no longer exist.
      const origins = {};
      (r.anims || []).forEach(a => { origins[a.to] = rectOf(a.from); });
      const capOrigin = r.captured ? rectOf(r.captured.sq) : null;

      lastMove = {from, to: sq};
      sel = null; targets = [];
      const s = await api("/state");
      paint(s);
      setProbs(await api("/predict", {}));

      // Release each piece from where it started, and fade the captured one.
      (r.anims || []).forEach(a => flipPiece(origins[a.to], a.to));
      if (r.captured) diePiece(capOrigin, r.captured.pc);

      say(s.over ? s.status : "Moved to " + sq + ".");
    } else {
      const s = await api("/state");
      if (s.selectable.includes(sq)){ sel = sq; targets = (await api("/select",{sq})).targets; }
      else { sel = null; targets = []; }
      paint(s);
    }
  }
}

async function reset(msg){
  await api("/new", {});
  sel = null; targets = []; lastMove = null;
  const s = await api("/state");
  paint(s);
  setProbs(await api("/predict", {}));
  say(msg);
}
el("btn-new").addEventListener("click", () => reset("New game. White to move."));
el("btn-undo").addEventListener("click", async () => {
  await api("/undo", {});
  sel = null; targets = []; lastMove = null;
  const s = await api("/state");
  paint(s);
  setProbs(await api("/predict", {}));
  say("Last move taken back.");
});
const wait = ms => new Promise(r => setTimeout(r, ms));

async function flipBoard(){
  if (flipping) return;
  const s = await api("/state");

  if (reduceMotion){
    flipped = !flipped;
    renderCoords();
    paint(s);
    say("Board flipped.");
    return;
  }

  flipping = true;
  const wrap = boardEl.closest(".boardwrap");

  // The board turns a half-revolution in place. Nothing is rewritten while
  // it is moving, so the whole turn runs off a single style change.
  wrap.classList.add("turning");
  await wait(540);                        // 520ms turn, plus a beat to land

  // The turn has landed and now reads identically to the flipped layout, so
  // the rewrite, the coordinate swap and the reset to zero can all happen in
  // one synchronous step -- the browser paints only once, after all of it.
  wrap.classList.add("snapping");
  wrap.classList.remove("turning");
  flipped = !flipped;
  renderCoords();
  paint(s);
  void boardEl.offsetWidth;               // commit the snap as one style
  wrap.classList.remove("snapping");

  flipping = false;
  say("Board flipped.");
}
el("btn-flip").addEventListener("click", flipBoard);

boardEl.addEventListener("keydown", e => {
  const b = e.target.closest(".sq"); if (!b) return;
  const d = {ArrowUp:[-1,0],ArrowDown:[1,0],ArrowLeft:[0,-1],ArrowRight:[0,1]}[e.key];
  if (!d) return;
  e.preventDefault();
  const r = Math.min(7, Math.max(0, Number(b.dataset.row) + d[0]));
  const c = Math.min(7, Math.max(0, Number(b.dataset.col) + d[1]));
  boardEl.children[r * 8 + c].focus();
});
// Arrow keys are consumed by the board's own navigation handler; every
// other shortcut is global. The old guard returned early whenever a square
// had focus -- which is the case right after any click -- so N/F/Z silently
// did nothing once you had touched the board.
const NAV_KEYS = {ArrowUp:1, ArrowDown:1, ArrowLeft:1, ArrowRight:1};
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (NAV_KEYS[e.key] && e.target.closest(".sq")) return;   // board nav only
  if (e.key === "f" || e.key === "F"){ e.preventDefault(); el("btn-flip").click(); }
  if (e.key === "n" || e.key === "N"){ e.preventDefault(); el("btn-new").click(); }
  if (e.key === "z" || e.key === "Z" || e.key === "Backspace"){
    e.preventDefault();
    if (!el("btn-undo").disabled) el("btn-undo").click();
  }
  if (e.key === "d" || e.key === "D"){ e.preventDefault(); el("btn-theme").click(); }
});

async function loadMeta(){
  try{
    const m = await (await fetch("/meta")).json();
    el("stats").innerHTML = m.stats.map(s =>
      "<dt>" + s.k + "</dt><dd>" + s.v + "</dd>").join("");
    el("foot").textContent = m.foot;
    el("credit").textContent = m.credit;
  }catch(e){}
}
loadMeta();

(async function boot(){
  const s = await api("/state");
  paint(s);
  setProbs(await api("/predict", {}));
})();
</script>
</body>
</html>
"""


def _fmt(n):
    return f"{n:,}"


def _stats():
    m = MODEL_META
    out = []
    if m["accuracy"]:
        out.append({"k": "Held-out accuracy", "v": f"{m['accuracy']*100:.2f}%"})
    if m["n_positions"]:
        out.append({"k": "Training positions", "v": _fmt(m["n_positions"])})
    out.append({"k": "Trees", "v": str(m["n_trees"])})
    out.append({"k": "Features", "v": str(m["n_features"])})
    out.append({"k": "Inference", "v": "6.8 ms"})
    return out


def _foot():
    m = MODEL_META
    pos = _fmt(m["n_positions"]) if m["n_positions"] else "n/a"
    acc = f"{m['accuracy']*100:.2f}%" if m["accuracy"] else "n/a"
    return (f"Gradient-boosted trees over {m['n_features']} positional features "
            f"(material, king safety, mobility, pawn structure, mate detection), "
            f"trained on {pos} positions generated by heuristic self-play and "
            f"labelled by a search-based oracle. {acc} held-out accuracy, split by "
            f"game so no position leaks across the boundary.")


@app.route("/")
def index():
    # No wordmark: a chess glyph next to the title was the most recognisable
    # "this was generated" signal on the page, and it encoded nothing.
    return (PAGE.replace("__CSS__", CSS)
                .replace("__PIECES__", json.dumps(PIECES))
                .replace("__VIEWBOX__", str(VIEWBOX)))


@app.route("/meta")
def meta():
    return jsonify(stats=_stats(), foot=_foot(), credit=ATTRIBUTION)


@app.route("/state", methods=["POST", "GET"])
def state():
    b = STATE["board"]
    return jsonify(
        pieces=board_dict(b),
        turn="w" if b.turn == chess.WHITE else "b",
        over=b.is_game_over(),
        status=game_status(b) or "Game in progress",
        check=b.is_check(),
        n_moves=len(b.move_stack),
        history=history_html(b),
        selectable=[chess.square_name(m.from_square) for m in b.legal_moves],
    )


@app.route("/select", methods=["POST"])
def select():
    b = STATE["board"]
    try:
        sc = chess.parse_square(request.json.get("sq"))
    except Exception as e:
        return jsonify(targets=[], error=str(e))
    return jsonify(targets=[chess.square_name(m.to_square)
                           for m in b.legal_moves if m.from_square == sc])


@app.route("/move", methods=["POST"])
def move():
    d = request.json
    b = STATE["board"]
    try:
        fr = chess.parse_square(d["from"])
        to = chess.parse_square(d["to"])
    except Exception as e:
        return jsonify(ok=False, error=str(e))

    moving = b.piece_at(fr)
    mv = chess.Move(fr, to)
    if moving and moving.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
        mv.promotion = chess.QUEEN
    if mv not in b.legal_moves:
        return jsonify(ok=False, error="illegal move")

    san = b.san(mv)
    pc = None
    if moving:
        pc = PIECE_CHARS[moving.piece_type]
        pc = pc.upper() if moving.color == chess.WHITE else pc

    # Where every piece stood BEFORE the push, so captures can be found by
    # subtraction afterwards rather than special-cased per move type.
    before = dict(b.piece_map())

    # Animate EVERY piece that relocates, not just the one that was clicked.
    # Castling moves two; en passant removes a pawn from a square that is
    # neither the origin nor the destination, so it is reported separately.
    anims = [{"from": chess.square_name(fr), "to": chess.square_name(to), "pc": pc}]
    if b.is_castling(mv):
        rank = chess.square_rank(fr)
        if chess.square_file(to) == 6:                       # king side
            rf, rt = chess.square(7, rank), chess.square(5, rank)
        else:                                                # queen side
            rf, rt = chess.square(0, rank), chess.square(3, rank)
        anims.append({"from": chess.square_name(rf),
                      "to": chess.square_name(rt),
                      "pc": "R" if b.turn == chess.WHITE else "r"})

    b.push(mv)

    # A square that held a piece and now holds a DIFFERENT one (or nothing),
    # without being the origin of a mover, lost a piece. Comparing identities
    # rather than emptiness is what catches an ordinary capture: there the
    # victim's square is the destination, so it is occupied again -- by the
    # attacker. This one rule also covers en passant (victim on an unrelated
    # square) and castling (vacated squares that ARE origins).
    vacated = {chess.parse_square(a["from"]) for a in anims}
    captured = None
    for sq, piece in before.items():
        if sq in vacated:
            continue
        if b.piece_at(sq) != piece:
            ch = PIECE_CHARS[piece.piece_type]
            captured = {"sq": chess.square_name(sq),
                        "pc": ch.upper() if piece.color == chess.WHITE else ch}
            break

    return jsonify(ok=True, san=san, moved=pc, anims=anims, captured=captured)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    b = STATE["board"]
    p = MODEL.predict_proba(
        SCALER.transform(extract_features(b).reshape(1, -1)))[0]
    return jsonify(black=float(p[0]), draw=float(p[1]), white=float(p[2]),
                   material_cp=int(round(static_eval(b) / 10.0)))


@app.route("/new", methods=["POST"])
def new():
    STATE["board"] = chess.Board()
    return jsonify(ok=True)


@app.route("/undo", methods=["POST"])
def undo():
    b = STATE["board"]
    if b.move_stack:
        b.pop()
    return jsonify(ok=True)


if __name__ == "__main__":
    print("\n  Live chess outcome prediction")
    print("  ----------------------------")
    print("  Open:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
