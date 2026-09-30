#!/usr/bin/env python3
"""
Chess game with ML win prediction. Fully self-contained - no CDN, no external JS.

Run:  python app.py
Open: http://127.0.0.1:5000
"""
from flask import Flask, request, jsonify
import chess
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import extract_features, FEATURE_NAMES

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
                # shape=(25865, 36) -> rows only
                info["n_positions"] = int(
                    line.split("shape=(")[1].split(",")[0].strip()
                )
    except FileNotFoundError:
        pass
    acc = f"{info['accuracy']*100:.2f}%" if info["accuracy"] else "n/a"
    npos = f"{info['n_positions']:,}" if info["n_positions"] else "n/a"
    return {
        "subtitle": (
            f"XGBoost classifier &middot; {info['n_features']} positional features"
            f" &middot; updates after every move"
        ),
        "meta": (
            f"Model: XGBoost {info['n_trees']} trees &middot; {npos} training positions<br>"
            f"Test accuracy: {acc} &middot; features: material, king safety,<br>"
            f"mobility, pawn structure, mate detection"
        ),
    }


MODEL_META = _model_meta()

# One shared board (single game at a time)
STATE = {"board": chess.Board(), "moves": []}

FILES = "abcdefgh"

PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Chess - ML Win Prediction</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #1a1a2e; color: #eee; min-height: 100vh;
  }
  h1 { text-align: center; font-size: 1.5rem; margin: 0 0 4px; font-weight: 600; }
  .sub { text-align: center; color: #888; font-size: .8rem; margin-bottom: 18px; }
  .wrap { display: flex; gap: 28px; justify-content: center; align-items: flex-start; flex-wrap: wrap; }
  .board {
    display: grid; grid-template-columns: repeat(8, 62px);
    grid-template-rows: repeat(8, 62px);
    border: 3px solid #444; border-radius: 4px; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,.5); user-select: none;
  }
  .sq { position: relative; display: flex; align-items: center; justify-content: center;
        font-size: 44px; line-height: 1; cursor: pointer; }
  .sq.l { background: #f0d9b5; } .sq.d { background: #b58863; }
  .sq.sel { outline: 4px solid #ff9800; outline-offset: -4px; z-index: 2; }
  .sq.tgt::after {
    content:''; position:absolute; width:26px; height:26px; border-radius:50%;
    background: rgba(0,0,0,.28); z-index:1;
  }
  .sq.last { box-shadow: inset 0 0 0 4px #4caf50; }
  .sq .pc { position: relative; z-index: 2; text-shadow: 0 1px 2px rgba(0,0,0,.35); }
  .sq.w .pc { color: #fff; text-shadow: 0 1px 3px #000, 0 0 1px #000; }
  .sq.b .pc { color: #111; text-shadow: 0 1px 2px rgba(255,255,255,.25); }
  .coord { position:absolute; font-size:10px; font-weight:700; opacity:.75; }
  .coord.fr { right:3px; bottom:1px; } .coord.rk { left:3px; top:1px; }
  .sq.l .coord { color:#b58863; } .sq.d .coord { color:#f0d9b5; }

  .panel { width: 300px; background:#16213e; border:1px solid #2a2a4a;
           border-radius:10px; padding:18px; }
  .panel h2 { margin:0 0 12px; font-size:1rem; letter-spacing:.5px; color:#9fb3ff;
              text-transform:uppercase; }
  .turn { font-size:1.25rem; font-weight:700; text-align:center; padding:10px;
          background:#0f3460; border-radius:8px; margin-bottom:6px; }
  .turn.w { color:#7CFC9A; } .turn.b { color:#ff8fa3; } .turn.d { color:#ffd166; }
  .status { text-align:center; font-size:.85rem; color:#ffd166; min-height:20px; margin-bottom:16px; }
  .bar-row { margin-bottom:12px; }
  .bar-lab { display:flex; justify-content:space-between; font-size:.8rem; margin-bottom:4px; }
  .track { height:20px; background:#0a0a1a; border-radius:10px; overflow:hidden; }
  .fill { height:100%; border-radius:10px; transition: width .25s ease; }
  .f-blk { background:linear-gradient(90deg,#7f1d1d,#ef4444); }
  .f-drw { background:linear-gradient(90deg,#78350f,#f59e0b); }
  .f-wht { background:linear-gradient(90deg,#14532d,#22c55e); }
  .sec { margin-top:20px; padding-top:16px; border-top:1px solid #2a2a4a; }
  .sec h3 { margin:0 0 8px; font-size:.8rem; text-transform:uppercase; color:#9fb3ff; letter-spacing:.5px; }
  #hist { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.8rem;
          height:170px; overflow-y:auto; background:#0a0a1a; padding:10px;
          border-radius:6px; line-height:1.6; }
  .btns { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }
  button { flex:1; min-width:88px; padding:10px 8px; font-size:.8rem; font-weight:600;
           border:none; border-radius:6px; cursor:pointer; background:#2a2a4a; color:#eee;
           transition:background .15s; }
  button:hover { background:#3d3d6a; }
  button.prim { background:#4f46e5; } button.prim:hover { background:#6366f1; }
  .meta { margin-top:14px; font-size:.68rem; color:#666; text-align:center; line-height:1.5; }
  .legend { display:flex; gap:14px; justify-content:center; font-size:.7rem; color:#888; margin-top:10px; }
</style>
</head>
<body>
<h1>♟️ Chess with ML Win Prediction</h1>
<div class="sub" id="subtitle">loading&hellip;</div>

<div class="wrap">
  <div class="board" id="board"></div>

  <div class="panel">
    <h2>Live Prediction</h2>
    <div class="turn" id="turn">Loading&hellip;</div>
    <div class="status" id="status"></div>

    <div class="bar-row">
      <div class="bar-lab"><span>Black</span><span id="p-blk">–</span></div>
      <div class="track"><div class="fill f-blk" id="b-blk" style="width:0%"></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-lab"><span>Draw</span><span id="p-drw">–</span></div>
      <div class="track"><div class="fill f-drw" id="b-drw" style="width:0%"></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-lab"><span>White</span><span id="p-wht">–</span></div>
      <div class="track"><div class="fill f-wht" id="b-wht" style="width:0%"></div></div>
    </div>

    <div class="sec">
      <h3>Move History</h3>
      <div id="hist"></div>
    </div>

    <div class="btns">
      <button class="prim" onclick="newGame()">New Game</button>
      <button onclick="undo()">Undo</button>
      <button onclick="flip()">Flip</button>
    </div>

    <div class="meta" id="meta"></div>
  </div>
</div>

<div class="legend"><span>Click a piece, then click destination</span></div>

<script>
const FILES = "abcdefgh";
const UNI = {
  K:'♔', Q:'♕', R:'♖', B:'♗', N:'♘', P:'♙',
  k:'♚', q:'♛', r:'♜', b:'♝', n:'♞', p:'♟'
};
let sel = null, legal = [], flipped = false;

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body || {})
  });
  return r.json();
}

function draw(state) {
  const b = document.getElementById('board');
  b.innerHTML = '';
  // Guard: if the server payload is malformed, do not wipe the board.
  if (!state || !state.pieces) { console.error('bad state payload', state); return; }
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      // row 0 is the TOP of the grid. Normal orientation => rank 8 on top,
      // so rk = 8 - row. Flipped => rank 1 on top, so rk = row + 1.
      // (Getting this backwards produces ranks 0..7 and silently breaks moves.)
      const fr = flipped ? 7 - col : col;
      const rk = flipped ? row + 1 : 8 - row;
      const sq = FILES[fr] + rk;
      const pc = state.pieces[sq];
      const div = document.createElement('div');
      const light = (fr + Number(rk)) % 2 === 0;
      div.className = 'sq ' + (light ? 'l' : 'd');
      div.dataset.square = sq;
      div.dataset.piece = pc || '';
      if (sel === sq) div.classList.add('sel');
      if (legal.includes(sq)) div.classList.add('tgt');
      if (state.last === sq) div.classList.add('last');
      if (pc) {
        div.classList.add(pc === pc.toUpperCase() ? 'w' : 'b');
        const s = document.createElement('span');
        s.className = 'pc'; s.textContent = UNI[pc];
        div.appendChild(s);
      }
      const cf = document.createElement('span');
      cf.className = 'coord fr'; cf.textContent = FILES[fr];
      div.appendChild(cf);
      const cr = document.createElement('span');
      cr.className = 'coord rk'; cr.textContent = rk;
      div.appendChild(cr);
      div.onclick = () => click(sq);
      b.appendChild(div);
    }
  }
  const t = document.getElementById('turn');
  if (state.over) {
    // The result belongs in the turn box; do not repeat it in the status line.
    t.textContent = state.status;
    // colour by the WINNER, not by whoever is notionally "to move"
    const whiteWon = /White wins/.test(state.status);
    t.className = 'turn ' + (whiteWon ? 'w' : (/Black wins/.test(state.status) ? 'b' : 'd'));
    document.getElementById('status').textContent = '';
  } else {
    t.textContent = state.turn === 'w' ? 'White to move' : 'Black to move';
    t.className = 'turn ' + state.turn;
    document.getElementById('status').textContent = '';
  }
  document.getElementById('hist').innerHTML = state.history || '';
  const h = document.getElementById('hist'); h.scrollTop = h.scrollHeight;
}

function setProbs(p) {
  document.getElementById('b-blk').style.width = (p.black*100)+'%';
  document.getElementById('b-drw').style.width = (p.draw*100)+'%';
  document.getElementById('b-wht').style.width = (p.white*100)+'%';
  document.getElementById('p-blk').textContent = (p.black*100).toFixed(1)+'%';
  document.getElementById('p-drw').textContent = (p.draw*100).toFixed(1)+'%';
  document.getElementById('p-wht').textContent = (p.white*100).toFixed(1)+'%';
}

async function refresh() {
  const s = await api('/state');
  legal = s.legal_targets; sel = null;
  draw(s);
  if (!s.over) { const p = await api('/predict', {}); setProbs(p); }
}

async function click(sq) {
  if (!sel) {
    const s = await api('/state');
    if (!s.selectable.includes(sq)) return;
    const r = await api('/select', {sq});
    sel = sq; legal = r.targets; draw(s);
  } else {
    const r = await api('/move', {from: sel, to: sq});
    if (r.ok) {
      sel = null; legal = []; draw(r.state);
      // Always refresh, INCLUDING on a game-ending move: mate is exactly
      // when the prediction matters most.
      const p = await api('/predict', {}); setProbs(p);
    } else {
      const s = await api('/state'); sel = null; legal = s.legal_targets; draw(s);
    }
  }
}

async function newGame() {
  const r = await api('/new', {});
  sel = null; legal = []; draw(r.state);
  const p = await api('/predict', {}); setProbs(p);
}
async function undo() {
  const r = await api('/undo', {});
  sel = null; legal = []; draw(r.state);
  if (!r.state.over) { const p = await api('/predict', {}); setProbs(p); }
}
function flip() { flipped = !flipped; refresh(); }

async function loadMeta() {
  try {
    const m = await (await fetch('/meta')).json();
    document.getElementById('subtitle').innerHTML = m.subtitle;
    document.getElementById('meta').innerHTML = m.meta;
  } catch (e) { /* leave placeholders */ }
}
loadMeta();
refresh();
</script>
</body>
</html>
"""

PIECE_CHARS = {1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k"}


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
        # flipped to Black, so testing afterwards mislabels every move and
        # the numbering comes out shifted by one full move.
        white_to_move = tmp.turn == chess.WHITE
        tmp.push(mv)
        if white_to_move:
            parts.append(f"<b>{tmp.fullmove_number}.</b> {san}")
        else:
            parts.append(san)
    return " ".join(parts)


def game_status(b: chess.Board):
    if b.is_checkmate():
        return f"{'White' if not b.turn else 'Black'} wins by checkmate!"
    if b.is_stalemate():
        return "Draw by stalemate"
    if b.is_insufficient_material():
        return "Draw by insufficient material"
    if b.can_claim_threefold_repetition():
        return "Draw by repetition"
    if b.halfmove_clock >= 100:
        return "Draw by fifty-move rule"
    return "Game in progress"


@app.route("/")
def index():
    return PAGE.replace("__SUBTITLE__", MODEL_META["subtitle"]).replace(
        "__META__", MODEL_META["meta"]
    )


@app.route("/meta", methods=["GET"])
def meta():
    return jsonify(MODEL_META)


@app.route("/state", methods=["POST", "GET"])
def state():
    b = STATE["board"]
    last = chess.square_name(b.peek().to_square) if b.move_stack else None
    return jsonify(
        pieces=board_dict(b),
        turn="w" if b.turn == chess.WHITE else "b",
        over=b.is_game_over(),
        status=game_status(b),
        last=last,
        history=history_html(b),
        legal_targets=[],
        selectable=[
            chess.square_name(m.from_square)
            for m in b.legal_moves
        ],
    )


@app.route("/select", methods=["POST"])
def select():
    sq = request.json.get("sq")
    try:
        b = STATE["board"]
        sc = chess.parse_square(sq)
        targets = [
            chess.square_name(m.to_square) for m in b.legal_moves if m.from_square == sc
        ]
        return jsonify(targets=targets)
    except Exception as e:
        return jsonify(targets=[], error=str(e))


@app.route("/move", methods=["POST"])
def move():
    d = request.json
    b = STATE["board"]
    try:
        fr = chess.parse_square(d["from"])
        to = chess.parse_square(d["to"])
    except Exception as e:
        return jsonify(ok=False, error=str(e))

    mv = chess.Move(fr, to)
    if b.piece_type_at(fr) == chess.PAWN and chess.square_rank(to) in (0, 7):
        mv.promotion = chess.QUEEN

    if mv not in b.legal_moves:
        return jsonify(ok=False, error="illegal move")

    san = b.san(mv)
    b.push(mv)
    over = b.is_game_over()
    return jsonify(
        ok=True,
        san=san,
        state=dict(
            pieces=board_dict(b),
            turn="w" if b.turn == chess.WHITE else "b",
            over=over,
            status=game_status(b),
            last=chess.square_name(mv.to_square),
            history=history_html(b),
        ),
    )


@app.route("/predict", methods=["POST", "GET"])
def predict():
    b = STATE["board"]
    feats = extract_features(b).reshape(1, -1)
    p = MODEL.predict_proba(SCALER.transform(feats))[0]
    return jsonify(black=float(p[0]), draw=float(p[1]), white=float(p[2]))


@app.route("/new", methods=["POST"])
def new():
    STATE["board"] = chess.Board()
    b = STATE["board"]
    return jsonify(
        state=dict(
            pieces=board_dict(b),
            turn="w",
            over=False,
            status="Game in progress",
            last=None,
            history="",
        )
    )


@app.route("/undo", methods=["POST"])
def undo():
    b = STATE["board"]
    if b.move_stack:
        b.pop()
    over = b.is_game_over()
    last = None
    if b.move_stack:
        last = chess.square_name(b.peek().to_square)
    return jsonify(
        state=dict(
            pieces=board_dict(b),
            turn="w" if b.turn == chess.WHITE else "b",
            over=over,
            status=game_status(b),
            last=last,
            history=history_html(b),
        )
    )


if __name__ == "__main__":
    print("\n  Chess + ML Win Prediction")
    print("  -----------------------")
    print("  Open in browser:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
