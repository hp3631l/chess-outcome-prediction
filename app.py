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
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#0d0e11; --ink:#eceef2; --ink-2:#8b93a3; --ink-3:#565e6d;
  --rule:rgba(255,255,255,.085); --rule-2:rgba(255,255,255,.15);
  --accent:#d9a441;
  --blk:#e2687f; --drw:#8b95a8; --wht:#4ad295;
  --sq-light:#e6d3ae; --sq-dark:#a67c56;
  --ease:cubic-bezier(.22,1,.36,1);
  --sq:66px;
  --sans:'Geist',ui-sans-serif,system-ui,'Segoe UI',sans-serif;
  --mono:'Geist Mono',ui-monospace,'SF Mono',Menlo,monospace;
  color-scheme:dark;
}
html,body{height:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  touch-action:manipulation;
}
.sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
    clip:rect(0 0 0 0);white-space:nowrap;border:0}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

.wrap{max-width:1120px;margin:0 auto;padding:0 28px 72px}

/* ---------- masthead: quiet, no stat pills ---------- */
.masthead{display:flex;align-items:baseline;justify-content:space-between;
          gap:20px;padding:30px 0 20px;flex-wrap:wrap}
.title{display:flex;align-items:center;gap:12px;min-width:0}
.title svg{width:26px;height:26px;flex:none}
.title h1{margin:0;font-size:1.06rem;font-weight:500;letter-spacing:-.014em}
.about{position:relative}
.about summary{
  list-style:none;cursor:pointer;font-size:.8rem;color:var(--ink-2);
  padding:5px 2px;border-bottom:1px solid var(--rule);
  transition:color 200ms cubic-bezier(.4,0,.2,1),border-color 200ms cubic-bezier(.4,0,.2,1);
}
.about summary::-webkit-details-marker{display:none}
.about summary:hover{color:var(--ink);border-color:var(--rule-2)}
.about summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.about[open] summary{border-color:transparent}
.about .body{
  position:absolute;right:0;top:calc(100% + 10px);z-index:20;width:308px;
  padding:16px 17px;background:#14161b;border:1px solid var(--rule-2);
  border-radius:10px;box-shadow:0 20px 50px -20px rgba(0,0,0,.9);
  font-size:.79rem;line-height:1.65;color:var(--ink-2);
}
.about .body p{margin:0 0 10px}
.about .body p:last-child{margin:0}
.about dl{margin:0 0 11px;display:grid;grid-template-columns:auto 1fr;gap:4px 14px}
.about dt{color:var(--ink-3)}
.about dd{margin:0;text-align:right;color:var(--ink)}
.credit{margin:0;color:var(--ink-3);font-size:.72rem;line-height:1.55}

/* ---------- composition: the board alone on the left, all of its reading in
              one narrow analysis column on the right. The verdict must NOT
              span the board width -- label/value pairs stretched across
              790px read as unrelated. ---------- */
.stage{display:grid;grid-template-columns:minmax(0,auto) 302px;gap:40px;
       align-items:start;justify-content:center}
.left{display:flex;flex-direction:column;gap:9px}
.rail{display:flex;flex-direction:column;gap:20px;min-height:100%}

/* ---------- board ---------- */
.boardwrap{display:flex;flex-direction:column;gap:7px}
.frame{padding:7px;border:1px solid var(--rule);border-radius:4px;background:#0a0b0e}
.board{position:relative;display:grid;
       grid-template-columns:repeat(8,var(--sq));grid-template-rows:repeat(8,var(--sq));
       user-select:none;-webkit-user-select:none;touch-action:manipulation;
       border-radius:2px;overflow:hidden}
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
/* Coordinate labels live outside .board, so they never turn with it. Fade
   them across the swap rather than letting them reverse while still legible. */
.coords,.ranks{transition:opacity 300ms var(--ease)}
.boardwrap.turning .coords,.boardwrap.turning .ranks{opacity:0}
.sq{position:relative;display:grid;place-items:center;padding:0;margin:0;border:0;
    background:var(--sq-light);cursor:pointer;line-height:0;
    -webkit-tap-highlight-color:transparent}
.sq.d{background:var(--sq-dark)}
.sq:focus{outline:none}
.sq:focus-visible{outline:3px solid var(--accent);outline-offset:-3px;z-index:6}
.sq.last::after{content:'';position:absolute;inset:0;
    background:rgba(217,164,65,.22);pointer-events:none}
.sq.sel{box-shadow:inset 0 0 0 4px var(--accent)}
.sq.king::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(circle,rgba(226,104,127,.95) 7%,rgba(226,104,127,.4) 42%,transparent 72%)}
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
/* mid-flight lift — animated on the inner svg so it cannot fight the
   translate on .pc above, which owns that transform property */
.pc.lift svg{animation:lift 300ms var(--ease)}
@keyframes lift{0%{transform:scale(1)}34%{transform:scale(1.1)}100%{transform:scale(1)}}
.pc.w path{fill:#fbfaf7;stroke:#171a21;stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
.pc.b path{fill:#191d26;stroke:#fbfaf7;stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
.hint{position:absolute;left:50%;top:50%;width:26%;height:26%;z-index:3;
      transform:translate(-50%,-50%);border-radius:50%;pointer-events:none;
      background:rgba(16,18,22,.55)}
.hint.cap{width:84%;height:84%;background:none;box-sizing:border-box;
          border:5px solid rgba(16,18,22,.48)}

.coords{display:flex;font-family:var(--mono);font-size:.6rem;color:var(--ink-3);
        font-variant-numeric:tabular-nums}
.coords.files{width:calc(8*var(--sq));margin-left:26px}
.coords.files span{width:var(--sq);text-align:center}
.ranks{display:flex;flex-direction:column;height:calc(8*var(--sq));width:18px;flex:none}
.ranks span{display:grid;place-items:center;height:var(--sq)}
.boardgrid{display:flex;gap:8px}

/* ---------- verdict: a composition, not three metrics ---------- */
.verdict{border-top:1px solid var(--rule);padding-top:18px}
.stack{position:relative;height:7px;border-radius:99px;overflow:hidden;
       background:rgba(255,255,255,.05)}
.seg{position:absolute;left:0;top:0;bottom:0;width:100%;
     transform-origin:left center;transition:transform 640ms var(--ease)}
.seg-b{background:var(--blk)} .seg-d{background:var(--drw)} .seg-w{background:var(--wht)}

.vlist{margin-top:15px}
/* Rows are centre-aligned, not baseline: promoting the leading figure with a
   baseline-aligned label left a large dead band under it. */
.vrow{display:flex;align-items:center;gap:9px;padding:5px 0;min-height:30px;
      color:var(--ink-2);transition:color 400ms var(--ease)}
.vrow .swatch{width:8px;height:8px;flex:none;border-radius:2px;background:var(--drw);
              transition:background 400ms var(--ease),transform 400ms var(--ease)}
.vrow .who{font-size:.85rem}
.vrow .pct{margin-left:auto;font-size:.85rem;color:var(--ink-2);
           transition:color 400ms var(--ease),font-size 400ms var(--ease)}
/* the leading outcome is promoted: brighter, larger, own accent on the swatch */
.vrow.lead{color:var(--ink)}
.vrow.lead .who{font-weight:500;font-size:.92rem}
.vrow.lead .pct{font-size:1.22rem;color:var(--ink);letter-spacing:-.025em}
.vrow.lead .swatch{transform:scale(1.45)}

.matline{display:flex;align-items:baseline;justify-content:space-between;
         gap:14px;margin-top:14px;padding-top:14px;border-top:1px solid var(--rule)}
.matline .k{font-size:.86rem;color:var(--ink-2)}
.matline .v{font-size:1.02rem;color:var(--ink)}
.matline .v.up{color:var(--wht)} .matline .v.dn{color:var(--blk)}

/* ---------- turn strip ---------- */
.turnline{display:flex;align-items:center;gap:9px;font-size:.9rem;color:var(--ink-2);
         min-height:22px}
.turnline .dot{width:8px;height:8px;flex:none;border-radius:50%;background:var(--ink-3);
               box-shadow:0 0 0 3px rgba(255,255,255,.05);
               transition:background 400ms var(--ease)}
.turnline.w .dot{background:#fbfaf7} .turnline.b .dot{background:#191d26;
                 box-shadow:0 0 0 1px var(--rule-2),0 0 0 4px rgba(255,255,255,.05)}
.turnline.res-w{color:var(--wht)} .turnline.res-w .dot{background:var(--wht)}
.turnline.res-b{color:var(--blk)} .turnline.res-b .dot{background:var(--blk)}
.turnline.res-d{color:var(--drw)} .turnline.res-d .dot{background:var(--drw)}

/* ---------- score sheet ---------- */
.sheetwrap{display:flex;flex-direction:column;min-height:0;flex:1}
.sheetwrap h2{margin:0 0 9px;font-size:.86rem;font-weight:500;color:var(--ink-2)}
#hist{flex:1;overflow-y:auto;overscroll-behavior:contain;min-height:64px;
      scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.14) transparent}
#hist::-webkit-scrollbar{width:6px}
#hist::-webkit-scrollbar-thumb{background:rgba(255,255,255,.14);border-radius:99px}
.sheet{width:100%;border-collapse:collapse;font-family:var(--mono);
       font-size:.79rem;font-variant-numeric:tabular-nums}
.sheet th,.sheet td{text-align:left;font-weight:400;padding:4px 0;
                    border-bottom:1px solid var(--rule)}
.sheet thead th{color:var(--ink-3);padding-bottom:6px}
.sheet tbody th{color:var(--ink-3);width:1.9em}
.sheet td:first-of-type{color:var(--ink);width:4.2em}
.sheet td:last-child{color:var(--ink-2)}
.sheet tbody tr:last-child th,.sheet tbody tr:last-child td{border-bottom:0}
.empty{color:var(--ink-3);font-size:.79rem;font-style:italic;padding:4px 0;margin:0}
.hintline{margin:0;text-align:center;font-size:.75rem;color:var(--ink-3)}

/* ---------- controls: understated ---------- */
.ctrls{display:flex;flex-direction:column;gap:7px}
button{font:inherit;color:inherit;cursor:pointer;border:0;background:none;
       -webkit-tap-highlight-color:transparent}
.btn{padding:10px 15px;border-radius:7px;font-size:.85rem;color:var(--ink-2);
     background:rgba(255,255,255,.045);border:1px solid var(--rule);text-align:left;
     display:flex;align-items:center;justify-content:space-between;gap:10px;
     transition:transform 170ms var(--ease),background 220ms cubic-bezier(.4,0,.2,1),
                color 220ms cubic-bezier(.4,0,.2,1),border-color 220ms cubic-bezier(.4,0,.2,1)}
.btn:hover:not([disabled]){background:rgba(255,255,255,.08);color:var(--ink);
                          border-color:var(--rule-2)}
.btn:active:not([disabled]){transform:scale(.985)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.btn .k{font-family:var(--mono);font-size:.72rem;color:var(--ink-3)}
.btn[disabled]{opacity:.35;cursor:not-allowed}
.btn.primary{background:var(--accent);color:#151006;border-color:transparent;
             font-weight:550;justify-content:center}
.btn.primary:hover{background:#e6b358;color:#151006}
.btn.primary .k{color:rgba(21,16,6,.5)}

/* ---------- responsive ---------- */
@media (max-width:1040px){
  :root{--sq:min(9.2vw,60px)}
  .stage{grid-template-columns:minmax(0,1fr);gap:30px;max-width:600px;margin:0 auto}
  .rail{padding-top:0;max-width:600px;width:100%;margin:0 auto}
  .ctrls{flex-direction:row;flex-wrap:wrap}
  .btn{flex:1 1 auto;min-width:120px;justify-content:center}
  .btn .k{display:none}
  .about .body{width:min(320px,calc(100vw - 56px))}
}
@media (max-width:560px){
  :root{--sq:min(10.4vw,52px)}
  .wrap{padding:0 14px 48px}
  .masthead{padding:20px 0 14px}
  .title h1{font-size:.96rem}
  .vrow.lead .pct{font-size:1.5rem}
}

/* ---------- reduced motion / transparency ---------- */
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
<meta name="theme-color" content="#0d0e11">
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

  <header class="masthead">
    <div class="title">__KNIGHT__<h1>Live chess outcome prediction</h1></div>
    <details class="about">
      <summary>About the model</summary>
      <div class="body">
        <dl id="stats"></dl>
        <p id="foot"></p>
        <p id="credit"></p>
      </div>
    </details>
  </header>

  <main class="stage">
    <section class="left" aria-label="Chess board">
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

    <aside class="rail" aria-label="Analysis">
      <div>
        <div class="turnline" id="turn"><span class="dot" aria-hidden="true"></span>
          <span id="turntxt">Loading&hellip;</span></div>
        <div class="verdict">
          <div class="stack" aria-hidden="true">
            <i class="seg seg-b" id="b-black"></i>
            <i class="seg seg-d" id="b-draw"></i>
            <i class="seg seg-w" id="b-white"></i>
          </div>
          <div class="vlist" id="vlist" aria-live="polite" aria-atomic="true">
            <div class="vrow" data-k="black"><i class="swatch" aria-hidden="true"></i>
              <span class="who">Black</span><span class="pct mono" id="v-black">&ndash;</span></div>
            <div class="vrow" data-k="draw"><i class="swatch" aria-hidden="true"></i>
              <span class="who">Draw</span><span class="pct mono" id="v-draw">&ndash;</span></div>
            <div class="vrow" data-k="white"><i class="swatch" aria-hidden="true"></i>
              <span class="who">White</span><span class="pct mono" id="v-white">&ndash;</span></div>
          </div>
          <div class="matline"><span class="k">Material</span>
            <span class="v mono" id="evalcp">&ndash;</span></div>
        </div>
      </div>

      <div class="sheetwrap">
        <h2>Moves</h2>
        <div id="hist" role="log" aria-label="Move history"></div>
      </div>

      <div class="ctrls">
        <button class="btn primary" id="btn-new">New game</button>
        <button class="btn" id="btn-undo" disabled>Undo<span class="k">&larr;</span></button>
        <button class="btn" id="btn-flip">Flip board<span class="k">F</span></button>
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
    x.textContent = state.status;
  } else {
    t.classList.add(state.turn);
    x.textContent = (state.turn === "w" ? "White" : "Black") + " to move";
  }
}

function setProbs(p){
  const v = {black:p.black, draw:p.draw, white:p.white};
  // A three-way probability is a composition, so draw it as ONE 100%-wide bar.
  // transform-only animation keeps it on the compositor.
  const bb = v.black;
  const dd = bb + v.draw;
  el("b-black").style.transform = "scaleX(" + bb + ")";
  el("b-draw").style.transform  = "translateX(" + (bb*100) + "%) scaleX(" + v.draw + ")";
  el("b-white").style.transform = "translateX(" + (dd*100) + "%) scaleX(" + v.white + ")";

  // Promote the leading outcome instead of printing a second headline figure.
  let lead = "draw";
  for (const k of ROWS) if (v[k] > v[lead]) lead = k;
  for (const k of ROWS){
    const row = el("vlist").querySelector('[data-k="'+k+'"]');
    row.classList.toggle("lead", k === lead);
    row.querySelector(".swatch").style.background = "var(--"+k+")";
    el("v-" + k).textContent = (v[k]*100).toFixed(1) + "%";
  }

  const cp = p.material_cp, e = el("evalcp");
  if (typeof cp === "number" && Math.abs(cp) >= 5){
    e.textContent = (cp > 0 ? "+" : "\u2212") + (Math.abs(cp)/100).toFixed(1);
    e.className = "v mono " + (cp > 0 ? "up" : "dn");
  } else {
    e.textContent = "level";
    e.className = "v mono";
  }
}

function say(m){ el("announce").textContent = m; }

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
document.addEventListener("keydown", e => {
  if (e.target.closest(".sq")) return;          // don't hijack board navigation
  if (e.key === "f" || e.key === "F") el("btn-flip").click();
  if (e.key === "n" || e.key === "N") el("btn-new").click();
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
    v = str(VIEWBOX)
    knight = ('<svg viewBox="0 0 ' + v + " " + v + '" aria-hidden="true">'
              '<path d="' + PIECES["wN"] + '" fill="#d9a441" fill-rule="evenodd" '
              'stroke="#0d0e11" stroke-width="7" stroke-linejoin="round" '
              'paint-order="stroke fill"/></svg>')
    return (PAGE.replace("__CSS__", CSS)
                .replace("__PIECES__", json.dumps(PIECES))
                .replace("__VIEWBOX__", v)
                .replace("__KNIGHT__", knight))


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
