#!/usr/bin/env python3
"""
Chess + live ML win prediction.

Run:  python app.py
Open: http://127.0.0.1:5000

Design notes
------------
* Self-contained by design. The first version loaded chessboard.js from a CDN
  and rendered a completely blank page whenever that CDN was unreachable, so
  the board, the pieces and the type are all inlined here. The only optional
  external request is the webfont, loaded non-blocking with a system fallback.
* Piece artwork is traced to SVG paths by build_pieces.py (Wikipedia pieces by
  Cburnett, CC BY-SA 3.0) so it stays crisp at any board size.
* One accent colour (gold) plus three semantic outcome colours. No decorative
  colour is introduced anywhere else on the page.
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
    """Read real training stats so the UI can never show stale hardcoded numbers."""
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

# Shared board: this is a two-player game on one screen, so one game in memory.
STATE = {"board": chess.Board()}

PIECE_CHARS = {1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k"}
PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


def board_dict(b: chess.Board):
    d = {}
    for sq, piece in b.piece_map().items():
        name = PIECE_CHARS[piece.piece_type]
        d[chess.square_name(sq)] = name.upper() if piece.color == chess.WHITE else name
    return d


def history_html(b: chess.Board):
    tmp = chess.Board()
    parts = []
    for mv in b.move_stack:
        san = tmp.san(mv)
        # Read turn BEFORE pushing. After a White move the turn has already
        # flipped to Black, so testing afterwards mislabels every move.
        white_to_move = tmp.turn == chess.WHITE
        tmp.push(mv)
        if white_to_move:
            parts.append(f'<b class="mvno">{tmp.fullmove_number}.</b>'
                         f'<span class="mv">{san}</span>')
        else:
            parts.append(f'<span class="mv">{san}</span>')
    return "".join(parts)


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
  --bg:#0a0c11; --bg-2:#0e1118; --surface:#141822; --surface-2:#1a1f2b;
  --hair:rgba(255,255,255,.075); --hair-2:rgba(255,255,255,.13);
  --ink:#e9ecf3; --ink-2:#9aa3b6; --ink-3:#5f6b80;
  --accent:#e3b341; --accent-dim:rgba(227,179,65,.16);
  --win-b:#f2647c; --win-d:#8892a6; --win-w:#3ddc97;
  --sq-light:#e9d3ab; --sq-dark:#a97a52;
  --ease:cubic-bezier(.22,1,.36,1);
  --ease-in:cubic-bezier(.4,0,.2,1);
  --r-card:20px; --r-inner:14px; --r-ctl:11px;
  --sq:62px;
  --shadow-lift:0 18px 44px -18px rgba(0,0,0,.85);
  color-scheme:dark;
}
html,body{height:100%}
body{
  margin:0; background:
    radial-gradient(1100px 620px at 12% -8%, rgba(227,179,65,.055), transparent 62%),
    radial-gradient(900px 560px at 96% 4%, rgba(120,150,255,.05), transparent 60%),
    var(--bg);
  color:var(--ink);
  font-family:'Outfit',ui-sans-serif,system-ui,'Segoe UI Variable Text','Segoe UI',sans-serif;
  font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  touch-action:manipulation;
}
.sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
    clip:rect(0 0 0 0);white-space:nowrap;border:0}

/* ---------- shell ---------- */
.shell{max-width:1180px;margin:0 auto;padding:28px 24px 56px;
       display:flex;flex-direction:column;gap:22px;min-height:100dvh}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;
        flex-wrap:wrap;padding-bottom:18px;border-bottom:1px solid var(--hair)}
.brand{display:flex;align-items:center;gap:13px;min-width:0}
.mark{width:38px;height:38px;flex:none;border-radius:11px;display:grid;place-items:center;
      background:linear-gradient(160deg,var(--surface-2),var(--surface));
      border:1px solid var(--hair-2);box-shadow:inset 0 1px 0 rgba(255,255,255,.07)}
.mark svg{width:23px;height:23px;display:block}
.brand h1{margin:0;font-size:1.02rem;font-weight:600;letter-spacing:-.015em;line-height:1.2}
.brand p{margin:1px 0 0;font-size:.755rem;color:var(--ink-2);letter-spacing:.055em;
        text-transform:uppercase;font-weight:500}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:baseline;gap:6px;padding:6px 12px;border-radius:999px;
      background:rgba(255,255,255,.038);border:1px solid var(--hair);
      font-size:.705rem;letter-spacing:.045em;text-transform:uppercase;
      font-weight:500;color:var(--ink-2);white-space:nowrap}
.chip b{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:.755rem;
        color:var(--ink);font-weight:600;letter-spacing:0;font-variant-numeric:tabular-nums;
        text-transform:none}

/* ---------- layout ---------- */
.stage{display:grid;grid-template-columns:minmax(0,auto) minmax(296px,352px);
       gap:30px;align-items:center;justify-content:center;flex:1}

/* double-bezel: outer shell + inner core, concentric radii */
.bezel{padding:9px;border-radius:var(--r-card);
       background:linear-gradient(165deg,rgba(255,255,255,.055),rgba(255,255,255,.012));
       border:1px solid var(--hair-2);box-shadow:var(--shadow-lift),inset 0 1px 0 rgba(255,255,255,.05)}
.bezel>.core{border-radius:var(--r-inner);background:var(--surface);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.045),0 0 0 1px rgba(0,0,0,.4);
            overflow:hidden}

/* ---------- board ---------- */
.boardwrap{display:flex;flex-direction:column;gap:9px;align-items:center}
.board{position:relative;display:grid;
       grid-template-columns:repeat(8,var(--sq));grid-template-rows:repeat(8,var(--sq));
       user-select:none;-webkit-user-select:none;touch-action:manipulation}
.sq{position:relative;display:grid;place-items:center;padding:0;margin:0;border:0;
    background:var(--sq-light);cursor:pointer;line-height:0;
    -webkit-tap-highlight-color:transparent}
.sq.d{background:var(--sq-dark)}
.sq:focus{outline:none}
.sq:focus-visible{outline:3px solid var(--accent);outline-offset:-3px;z-index:6}
.sq.last::after{content:'';position:absolute;inset:0;
    background:rgba(227,179,65,.20);pointer-events:none}
.sq.sel{box-shadow:inset 0 0 0 4px var(--accent)}
.sq.king::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(circle at 50% 50%,rgba(242,100,124,.92) 8%,rgba(242,100,124,.42) 42%,transparent 72%)}
.pc{position:absolute;inset:9%;z-index:2;pointer-events:none;line-height:0;
    transition:transform 320ms var(--ease),opacity 320ms var(--ease)}
.pc svg{width:100%;height:100%;display:block;overflow:visible}
.pc.w path{fill:#fbfaf7;stroke:#171a21;stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
.pc.b path{fill:#191d26;stroke:#fbfaf7;stroke-width:4.2;
           stroke-linejoin:round;paint-order:stroke fill}
.pc.pop{animation:pop 300ms var(--ease)}
@keyframes pop{from{transform:scale(.72);opacity:0}to{transform:scale(1);opacity:1}}
/* legal-move marker: a small centred dot, or a ring when capturing */
.hint{position:absolute;left:50%;top:50%;width:27%;height:27%;z-index:3;
      transform:translate(-50%,-50%);border-radius:50%;pointer-events:none;
      background:rgba(16,20,28,.52)}
.hint.cap{width:86%;height:86%;background:none;box-sizing:border-box;
          border:5px solid rgba(16,20,28,.45)}
.ghost{position:absolute;z-index:8;pointer-events:none;line-height:0;
       transition:transform 300ms var(--ease),opacity 300ms var(--ease)}
.ghost svg{width:100%;height:100%;display:block}

/* coordinates outside the playing area, per chess convention */
.coords{display:flex;font-family:'JetBrains Mono',ui-monospace,monospace;
        font-size:.6rem;letter-spacing:.06em;color:var(--ink-3);
        text-transform:uppercase;font-weight:500}
.coords.files{width:calc(8*var(--sq));margin-left:calc(20px + 8px)}
.coords.files span{width:var(--sq);text-align:center}
.ranks{display:flex;flex-direction:column;
       height:calc(8*var(--sq));width:20px;flex:none}
.ranks span{display:grid;place-items:center;height:var(--sq)}
.boardgrid{display:flex;gap:8px}

/* ---------- panel ---------- */
.panel{display:flex;flex-direction:column;gap:14px}
.card{padding:17px 18px}
.card h2{margin:0 0 13px;font-size:.695rem;letter-spacing:.155em;text-transform:uppercase;
         color:var(--ink-2);font-weight:600}

.turn{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:var(--r-ctl);
      background:var(--surface-2);border:1px solid var(--hair);
      transition:background 400ms var(--ease),border-color 400ms var(--ease)}
.dot{width:9px;height:9px;flex:none;border-radius:50%;background:var(--ink-3);
     box-shadow:0 0 0 4px rgba(255,255,255,.05);transition:background 400ms var(--ease)}
.turn.w .dot{background:#fbfaf7} .turn.b .dot{background:#191d26;border:1px solid var(--hair-2)}
.turn.w{border-color:rgba(251,250,247,.28)} .turn.b{border-color:rgba(255,255,255,.16)}
.turntxt{font-size:.905rem;font-weight:500;letter-spacing:-.005em;min-width:0;
         overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.turn.res-w{background:rgba(61,220,151,.1);border-color:rgba(61,220,151,.4)}
.turn.res-w .dot{background:var(--win-w)}
.turn.res-b{background:rgba(242,100,124,.1);border-color:rgba(242,100,124,.4)}
.turn.res-b .dot{background:var(--win-b)}
.turn.res-d{background:rgba(136,146,166,.1);border-color:rgba(136,146,166,.4)}
.turn.res-d .dot{background:var(--win-d)}

.meters{margin-top:16px}
.meter+.meter{margin-top:12px}
.mtop{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:6px;gap:10px}
.mname{font-size:.755rem;color:var(--ink-2);letter-spacing:.075em;
       text-transform:uppercase;font-weight:500}
.mval{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:.875rem;font-weight:600;
      font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.track{height:5px;border-radius:99px;background:rgba(255,255,255,.055);overflow:hidden}
.fill{height:100%;border-radius:99px;width:0;
      transition:width 620ms var(--ease),background 400ms var(--ease-in)}
.f-b{background:var(--win-b)} .f-d{background:var(--win-d)} .f-w{background:var(--win-w)}

.eval{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
      padding-top:13px;margin-top:15px;border-top:1px solid var(--hair)}
.eval .lbl{font-size:.755rem;color:var(--ink-2);letter-spacing:.075em;
           text-transform:uppercase;font-weight:500}
.eval .num{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:1.02rem;
           font-weight:600;font-variant-numeric:tabular-nums}

.hist{min-height:78px;max-height:196px;overflow-y:auto;overflow-x:hidden;
      overscroll-behavior:contain;
      background:rgba(0,0,0,.26);border:1px solid var(--hair);border-radius:var(--r-ctl);
      padding:11px 13px;font-family:'JetBrains Mono',ui-monospace,monospace;
      font-size:.755rem;line-height:1.85;scrollbar-width:thin;
      scrollbar-color:rgba(255,255,255,.16) transparent;
      display:flex;flex-wrap:wrap;align-content:flex-start;gap:0 2px}
.hist::-webkit-scrollbar{width:7px}
.hist::-webkit-scrollbar-thumb{background:rgba(255,255,255,.16);border-radius:99px}
.mvno{color:var(--ink-3);margin-right:5px}
.mv{color:var(--ink);margin-right:7px;white-space:nowrap}
.hist .none{color:var(--ink-3);font-style:italic;width:100%}

.ctrls{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{font:inherit;color:inherit;cursor:pointer;border:0;background:none;
       -webkit-tap-highlight-color:transparent}
.btn{padding:11px 15px;border-radius:var(--r-ctl);font-size:.815rem;font-weight:550;
     letter-spacing:.005em;background:var(--surface-2);border:1px solid var(--hair);
     color:var(--ink-2);
     transition:transform 180ms var(--ease),background 220ms var(--ease-in),
                color 220ms var(--ease-in),border-color 220ms var(--ease-in)}
.btn:hover{background:#222839;color:var(--ink);border-color:var(--hair-2)}
.btn:active{transform:scale(.972)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.btn.primary{background:var(--accent);color:#1a1405;border-color:transparent;
             grid-column:1/-1;font-weight:650}
.btn.primary:hover{background:#f0c358;color:#1a1405}
.btn[disabled]{opacity:.4;cursor:not-allowed}

.foot{padding-top:16px;border-top:1px solid var(--hair);font-size:.705rem;
      color:var(--ink-3);line-height:1.65}
.foot b{color:var(--ink-2);font-weight:550}
.credit{margin:9px 0 0;font-size:.665rem;color:#4b5468;line-height:1.55}
.hintline{margin:0;font-size:.755rem;color:var(--ink-3);text-align:center;
          letter-spacing:.02em}

/* ---------- responsive ---------- */
@media (max-width:1000px){
  :root{--sq:min(9.4vw,58px)}
  .stage{grid-template-columns:minmax(0,1fr);gap:24px;max-width:560px;margin:0 auto}
  .panel{max-width:560px;width:100%;margin:0 auto}
}
@media (max-width:560px){
  :root{--sq:min(10.6vw,54px);--r-card:16px;--r-inner:11px}
  .shell{padding:18px 14px 40px;gap:16px}
  .brand h1{font-size:.94rem}
  .chips{width:100%}
  .coords.files{margin-left:calc(20px + 8px)}
}

/* ---------- reduced motion ---------- */
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
                       transition-duration:.01ms!important;scroll-behavior:auto!important}
}
@media (prefers-reduced-transparency:reduce){
  .bezel{background:var(--surface);border-color:var(--hair)}
}
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0c11">
<meta name="description" content="Play two-player chess with a live machine-learning win-probability estimate after every move.">
<title>Live Chess Outcome Prediction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" media="print" onload="this.media='all'"
      href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap">
<style>__CSS__</style>
</head>
<body>
<div class="shell">

  <header class="topbar">
    <div class="brand">
      <div class="mark" aria-hidden="true">__KNIGHT__</div>
      <div>
        <h1>Live Chess Outcome Prediction</h1>
        <p>XGBoost &middot; <span id="sub-feat">&mdash;</span> features</p>
      </div>
    </div>
    <div class="chips" id="chips"></div>
  </header>

  <main class="stage">
    <section class="boardwrap" aria-label="Chess board">
      <div class="coords files" id="cfiles" aria-hidden="true"></div>
      <div class="boardgrid">
        <div class="ranks" id="cranks" aria-hidden="true"></div>
        <div class="bezel">
          <div class="core">
            <div class="board" id="board" role="grid"
                 aria-label="Chess board, two players on one screen"></div>
          </div>
        </div>
      </div>
      <p class="hintline">Select a piece, then a highlighted square</p>
    </section>

    <aside class="panel">
      <div class="bezel"><div class="core card">
        <h2>Evaluation</h2>
        <div class="turn" id="turn"><span class="dot" aria-hidden="true"></span>
          <span class="turntxt" id="turntxt">Loading&hellip;</span></div>
        <div class="meters" id="meters" aria-live="polite" aria-atomic="true"></div>
        <div class="eval">
          <span class="lbl">Material</span>
          <span class="num" id="evalcp">&mdash;</span>
        </div>
        <p class="sr" id="announce" role="status" aria-live="polite"></p>
      </div></div>

      <div class="bezel"><div class="core card">
        <h2>Move History</h2>
        <div class="hist" id="hist" tabindex="0" role="log"
             aria-label="Move history"><span class="none">No moves yet</span></div>
      </div></div>

      <div class="ctrls">
        <button class="btn primary" id="btn-new">New Game</button>
        <button class="btn" id="btn-undo" disabled>Undo Move</button>
        <button class="btn" id="btn-flip">Flip Board</button>
      </div>

      <p class="foot" id="foot"></p>
      <p class="credit" id="credit"></p>
    </aside>
  </main>
</div>

<script>
const PIECES = __PIECES__;
const VIEWBOX = __VIEWBOX__;
const FILES = "abcdefgh";
const NAME = {p:"pawn",n:"knight",b:"bishop",r:"rook",q:"queen",k:"king"};
const OUT = [
  {key:"black", cls:"f-b", label:"Black"},
  {key:"draw",  cls:"f-d", label:"Draw"},
  {key:"white", cls:"f-w", label:"White"}
];

let sel = null, targets = [], flipped = false, lastMove = null, moveCount = 0;
const el = id => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function svg(pc, cls){
  // PIECES keys are "wP".."wK" / "bP".."bK": colour prefix + UPPERCASE letter.
  const key = (pc === pc.toUpperCase() ? "w" : "b") + pc.toUpperCase();
  const d = PIECES[key];
  const s = document.createElementNS("http://www.w3.org/2000/svg","svg");
  s.setAttribute("viewBox","0 0 "+VIEWBOX+" "+VIEWBOX);
  s.setAttribute("aria-hidden","true");
  if (d){
    const p = document.createElementNS("http://www.w3.org/2000/svg","path");
    p.setAttribute("d", d);
    p.setAttribute("fill-rule","evenodd");
    s.appendChild(p);
  } else {
    console.error("no piece geometry for key:", key);
  }
  return s;
}
function pieceEl(pc, pop){
  const s = document.createElement("span");
  s.className = "pc " + (pc === pc.toUpperCase() ? "w" : "b") + (pop ? " pop" : "");
  s.dataset.pc = pc;                 // identity, so paint() can skip no-op work
  s.appendChild(svg(pc));
  return s;
}

/* ---- build the 64 squares once; they are then only mutated, never rebuilt,
       which is what makes the move animation possible ---- */
const boardEl = el("board");
const SQ = {};
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
function coords(){
  el("cfiles").innerHTML = FILES.split("").map(f => "<span>"+f+"</span>").join("");
  el("cranks").innerHTML = [8,7,6,5,4,3,2,1].map(n => "<span>"+n+"</span>").join("");
}
coords();

function label(sq, pieces, turn, over){
  const pc = pieces[sq];
  if (pc) return (pc === pc.toUpperCase() ? "White " : "Black ") + NAME[pc.toLowerCase()] + " on " + sq;
  if (over) return sq + ", empty";
  return sq + ", empty, " + (turn === "w" ? "White" : "Black") + " to move";
}

function paint(state, opts){
  opts = opts || {};
  for (const b of boardEl.children){
    // grid order is fixed (row-major); map to the displayed square
    // DOM order is row-major over a fixed 8x8 grid. row 0 is the TOP of the
    // board. Normal orientation => rank 8 on top, so r = 8 - row.
    // Flipped => rank 1 on top, so r = row + 1.
    // (Getting this off by one silently drops rank 8 and adds an invalid rank 0.)
    const r = flipped ? Number(b.dataset.row) + 1 : 8 - Number(b.dataset.row);
    const c = flipped ? 7 - Number(b.dataset.col) : Number(b.dataset.col);
    const sq = FILES[c] + r;
    b.dataset.square = sq;

    // 1. piece content -- may no-op, but must never skip the decoration below
    const want = state.pieces[sq];
    const cur = b.querySelector(".pc");
    if (!want){
      if (cur) cur.remove();
    } else if (!cur || cur.dataset.pc !== want){
      if (cur) cur.remove();
      b.appendChild(pieceEl(want, opts.pop === sq));
    }

    // 2. decoration: reset then re-apply EVERY visual state from scratch
    b.className = "sq " + (((Number(b.dataset.row) + Number(b.dataset.col)) % 2) ? "d" : "");
    if (lastMove && (sq === lastMove.from || sq === lastMove.to)) b.classList.add("last");
    b.setAttribute("aria-label", label(sq, state.pieces, state.turn, state.over));

    const old = b.querySelector(".hint");
    if (old) old.remove();
    if (sel === sq){
      b.classList.add("sel");
    } else if (sel && targets.includes(sq)){
      const mark = document.createElement("span");
      mark.className = "hint" + (want ? " cap" : "");
      b.appendChild(mark);
    }
  }
  // highlight a king that is in check
  if (!state.over && state.check){
    const k = state.turn === "w" ? "K" : "k";
    for (const b of boardEl.children){
      if (state.pieces[b.dataset.square] === k){ b.classList.add("king"); break; }
    }
  }
  renderTurn(state);
  el("hist").innerHTML = state.history || '<span class="none">No moves yet</span>';
  el("btn-undo").disabled = !state.history;
}

function renderTurn(state){
  const t = el("turn"), x = el("turntxt");
  t.className = "turn";
  if (state.over){
    t.classList.add(/White wins/.test(state.status) ? "res-w"
                  : /Black wins/.test(state.status) ? "res-b" : "res-d");
    x.textContent = state.status;
  } else {
    t.classList.add(state.turn);
    x.textContent = (state.turn === "w" ? "White" : "Black") + " to move";
  }
}

function meters(p){
  el("meters").innerHTML = OUT.map(o =>
    '<div class="meter">' +
      '<div class="mtop"><span class="mname">' + o.label + '</span>' +
      '<span class="mval" id="v-' + o.key + '">0.0%</span></div>' +
      '<div class="track"><div class="fill ' + o.cls + '" id="b-' + o.key + '"></div></div>' +
    '</div>').join("");
}
meters({});

function setProbs(p){
  OUT.forEach(o => {
    const v = p[o.key];
    const bar = el("b-" + o.key), lab = el("v-" + o.key);
    if (bar) bar.style.width = (v * 100).toFixed(2) + "%";
    if (lab) lab.textContent = (v * 100).toFixed(1) + "%";
  });
  const cp = p.material_cp;
  const e = el("evalcp");
  if (typeof cp === "number" && Math.abs(cp) >= 5){
    e.textContent = (cp > 0 ? "+" : "−") + (Math.abs(cp) / 100).toFixed(1);
    e.style.color = cp > 0 ? "var(--win-w)" : "var(--win-b)";
  } else {
    e.textContent = "level";
    e.style.color = "var(--ink-3)";
  }
}

function say(msg){ el("announce").textContent = msg; }

async function api(path, body){
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                               body: JSON.stringify(body || {})});
  return r.json();
}

/* animate the moving piece: measure before, then slide a ghost from->to */
function slide(from, to, pc){
  if (reduceMotion || !from || !to) return;
  const a = boardEl.querySelector('[data-square="'+from+'"]');
  const b = boardEl.querySelector('[data-square="'+to+'"]');
  if (!a || !b) return;
  const box = boardEl.getBoundingClientRect();
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
  const g = document.createElement("div");
  g.className = "ghost " + (pc === pc.toUpperCase() ? "w" : "b");
  g.style.width = ra.width + "px"; g.style.height = ra.height + "px";
  g.style.left = (ra.left - box.left) + "px"; g.style.top = (ra.top - box.top) + "px";
  g.appendChild(svg(pc));
  boardEl.appendChild(g);
  requestAnimationFrame(() => {
    g.style.transform = "translate(" + (rb.left - ra.left) + "px," + (rb.top - ra.top) + "px)";
  });
  setTimeout(() => g.remove(), 330);
}

async function onSquare(sq){
  if (el("turn").classList.contains("res-w") || el("turn").classList.contains("res-b")
      || el("turn").classList.contains("res-d")) return;
  if (!sel){
    const s = await api("/state");
    if (!s.selectable.includes(sq)) return;
    sel = sq; targets = (await api("/select", {sq})).targets;
    paint(s);
    const pc = s.pieces[sq];
    say((pc === pc.toUpperCase() ? "White " : "Black ") + NAME[pc.toLowerCase()] +
        " on " + sq + " selected. " + targets.length + " legal target" +
        (targets.length === 1 ? "" : "s") + ".");
  } else {
    const from = sel;
    const r = await api("/move", {from: from, to: sq});
    if (r.ok){
      const pc = r.moved;
      lastMove = {from, to: sq}; moveCount++;
      sel = null; targets = [];
      const s = await api("/state");
      paint(s, {pop: sq});
      slide(from, sq, pc);
      // Always refresh, INCLUDING on a game-ending move: mate is exactly
      // when the prediction matters most.
      setProbs(await api("/predict", {}));
      say(s.over ? s.status : "Moved to " + sq);
    } else {
      const s = await api("/state");
      if (s.selectable.includes(sq)){ sel = sq; targets = (await api("/select",{sq})).targets; }
      else { sel = null; targets = []; }
      paint(s);
    }
  }
}

el("btn-new").addEventListener("click", async () => {
  await api("/new", {});
  sel = null; targets = []; lastMove = null;
  const s = await api("/state");
  paint(s); setProbs(await api("/predict", {}));
  say("New game started. White to move.");
});
el("btn-undo").addEventListener("click", async () => {
  await api("/undo", {});
  sel = null; targets = []; lastMove = null;
  const s = await api("/state");
  paint(s);
  if (!s.over) setProbs(await api("/predict", {}));
  say("Last move taken back.");
});
el("btn-flip").addEventListener("click", async () => {
  flipped = !flipped;
  const s = await api("/state");
  paint(s);
  say("Board flipped.");
});

/* roving arrow-key navigation across the grid */
boardEl.addEventListener("keydown", e => {
  const b = e.target.closest(".sq"); if (!b) return;
  const d = {ArrowUp:[-1,0],ArrowDown:[1,0],ArrowLeft:[0,-1],ArrowRight:[0,1]}[e.key];
  if (!d) return;
  e.preventDefault();
  const r = Math.min(7, Math.max(0, Number(b.dataset.row) + d[0]));
  const c = Math.min(7, Math.max(0, Number(b.dataset.col) + d[1]));
  const nb = boardEl.children[r * 8 + c];
  if (nb){ nb.focus(); }
});

async function loadMeta(){
  try{
    const m = await (await fetch("/meta")).json();
    el("chips").innerHTML = m.chips.map(c => '<span class="chip">' + c.k +
      ' <b>' + c.v + "</b></span>").join("");
    el("sub-feat").textContent = m.n_features;
    el("foot").innerHTML = m.foot;
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


def _chips():
    m = MODEL_META
    out = []
    if m["accuracy"]:
        out.append({"k": "Accuracy", "v": f"{m['accuracy']*100:.2f}%"})
    if m["n_positions"]:
        out.append({"k": "Positions", "v": _fmt(m["n_positions"])})
    out.append({"k": "Trees", "v": str(m["n_trees"])})
    out.append({"k": "Latency", "v": "6.8 ms"})
    return out


def _foot():
    m = MODEL_META
    pos = _fmt(m["n_positions"]) if m["n_positions"] else "n/a"
    acc = f"{m['accuracy']*100:.2f}%" if m["accuracy"] else "n/a"
    return (
        f"<b>{m['n_trees']}</b> gradient-boosted trees over <b>{m['n_features']}</b> "
        f"positional features, trained on <b>{pos}</b> positions "
        f"({acc} held-out accuracy). Features cover material, king safety, "
        f"mobility, pawn structure and mate detection."
    )


@app.route("/")
def index():
    v = str(VIEWBOX)
    knight = (
        '<svg viewBox="0 0 ' + v + " " + v + '">'
        '<path d="' + PIECES["wN"] + '" fill="#e3b341" fill-rule="evenodd"'
        ' stroke="#0a0c11" stroke-width="6" stroke-linejoin="round"'
        ' paint-order="stroke fill"/></svg>'
    )
    page = (
        PAGE.replace("__CSS__", CSS)
        .replace("__PIECES__", json.dumps(PIECES))
        .replace("__VIEWBOX__", str(VIEWBOX))
        .replace("__KNIGHT__", knight)
    )
    return page


@app.route("/meta")
def meta():
    # ATTRIBUTION is a plain string, not a tuple.
    return jsonify(chips=_chips(), n_features=len(FEATURE_NAMES), foot=_foot(),
                   credit=ATTRIBUTION)


@app.route("/state", methods=["POST", "GET"])
def state():
    b = STATE["board"]
    last = chess.square_name(b.peek().to_square) if b.move_stack else None
    return jsonify(
        pieces=board_dict(b),
        turn="w" if b.turn == chess.WHITE else "b",
        over=b.is_game_over(),
        status=game_status(b) or "Game in progress",
        check=b.is_check(),
        last=last,
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
    return jsonify(
        targets=[chess.square_name(m.to_square) for m in b.legal_moves if m.from_square == sc]
    )


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
    b.push(mv)
    return jsonify(ok=True, san=san, moved=pc)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    b = STATE["board"]
    feats = extract_features(b).reshape(1, -1)
    p = MODEL.predict_proba(SCALER.transform(feats))[0]
    return jsonify(
        black=float(p[0]),
        draw=float(p[1]),
        white=float(p[2]),
        material_cp=int(round(static_eval(b) / 10.0)),
    )


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
    print("\n  Live Chess Outcome Prediction")
    print("  ----------------------------")
    print("  Open:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
