"""Generate report figures (PNG) from the trained model and dataset."""
import os
import pickle
import collections

import numpy as np
import pandas as pd
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from train_model import extract_features, FEATURE_NAMES, truth_label

MODEL = pickle.load(open("model.pkl", "rb"))
SCALER = pickle.load(open("scaler.pkl", "rb"))
df = pd.read_csv("positions_dataset.csv")

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# palette
C_BLACK, C_DRAW, C_WHITE = "#e74c3c", "#f39c12", "#27ae60"
C_GRID = "#2a2a4a"
plt.rcParams.update({
    "figure.facecolor": "#16213e",
    "axes.facecolor": "#16213e",
    "text.color": "#eee",
    "axes.labelcolor": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": C_GRID,
    "font.size": 10,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- 1. class balance
counts = collections.Counter(df["label"].tolist())
names = ["Black wins", "Draw", "White wins"]
vals = [counts.get(i, 0) for i in range(3)]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(names, vals, color=[C_BLACK, C_DRAW, C_WHITE], width=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * .02,
            f"{v:,}\n{v/sum(vals)*100:.1f}%", ha="center", color="#eee", fontsize=10)
ax.set_ylabel("Positions")
ax.set_title("Class distribution of the training set")
ax.grid(axis="y", alpha=.3)
ax.set_ylim(0, max(vals) * 1.2)
save(fig, "fig1_class_distribution.png")

# ---------------------------------------------------------------- 2. confusion matrix
# Numbers from the held-out test split (printed by train_model.py)
cm = np.array([[1621, 64, 2],
               [51, 2231, 68],
               [0, 75, 1560]])
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        frac = cm[i, j] / cm[i].sum()
        ax.text(j, i, f"{cm[i,j]:,}\n{frac*100:.1f}%", ha="center", va="center",
                color="white" if frac > .45 else "#eee", fontweight="bold", fontsize=11)
ax.set_xticks(range(3), names)
ax.set_yticks(range(3), names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix (held-out test set, n = 5,672)")
save(fig, "fig2_confusion_matrix.png")

# ---------------------------------------------------------------- 3. feature importance
imp = pd.Series(MODEL.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
top = imp.tail(15)
fig, ax = plt.subplots(figsize=(8, 5.4))
colors = ["#4f46e5" if "mate" in i else "#38bdf8" for i in top.index]
ax.barh(top.index, top.values, color=colors)
ax.set_xlabel("Feature importance (gain-weighted split count)")
ax.set_title("Top 15 features by importance")
ax.grid(axis="x", alpha=.3)
save(fig, "fig3_feature_importance.png")

# ---------------------------------------------------------------- 4. material response curve
def probe(white_spec=(), black_spec=()):
    b = chess.Board()
    empties = [s for s in chess.SQUARES if b.piece_at(s) is None]
    for spec, color in ((white_spec, chess.WHITE), (black_spec, chess.BLACK)):
        for pt in spec:
            for s in empties:
                if chess.square_rank(s) in (3, 4) and chess.square_file(s) in (2, 3, 4, 5):
                    b.set_piece_at(s, chess.Piece(pt, color))
                    empties.remove(s)
                    break
    p = MODEL.predict_proba(SCALER.transform(extract_features(b).reshape(1, -1)))[0]
    return p


Q, R, B, N, P = chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN
series = [("start", ()), ("+P", (P,)), ("+2P", (P, P)), ("+N", (N,)),
          ("+B", (B,)), ("+R", (R,)), ("+Q", (Q,)), ("+Q+R", (Q, R))]
xs, ys, es = [], [], []
for lbl, spec in series:
    p = probe(spec)
    xs.append(lbl)
    ys.append(p[2])
    es.append(p[1])

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xs, ys, "o-", color=C_WHITE, lw=2.5, ms=8, label="P(White wins)")
ax.plot(xs, es, "s--", color=C_DRAW, lw=2, ms=6, label="P(Draw)")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Extra material for White")
ax.set_ylabel("Predicted probability")
ax.set_title("Model response to material advantage (monotonicity check)")
ax.legend(facecolor="#0f3460", edgecolor="none", labelcolor="#eee")
ax.grid(alpha=.3)
save(fig, "fig4_material_response.png")

# ---------------------------------------------------------------- 5. probability along a real game
seq = ("e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 "
       "Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8").split()
b = chess.Board()
probs, plies = [], []
for i, mv in enumerate(seq):
    try:
        b.push_san(mv)
    except Exception:
        break
    p = MODEL.predict_proba(SCALER.transform(extract_features(b).reshape(1, -1)))[0]
    probs.append(p)
    plies.append(i + 1)
probs = np.array(probs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.stackplot(plies, probs[:, 0], probs[:, 1], probs[:, 2],
              colors=[C_BLACK, C_DRAW, C_WHITE], alpha=.85,
              labels=["Black", "Draw", "White"])
ax1.set_ylabel("Probability")
ax1.set_ylim(0, 1)
ax1.set_title("Live prediction through Morphy's Opera Game (1858)")
ax1.legend(loc="upper left", facecolor="#0f3460", edgecolor="none", labelcolor="#eee", ncol=3)
ax1.grid(alpha=.25)

# decisive-class probability, i.e. how lopsided the position is
dec = probs[:, 0] + probs[:, 2]
ax2.plot(plies, dec, "-", color="#a78bfa", lw=2)
ax2.fill_between(plies, 0.5, dec, where=dec >= .5, color=C_WHITE, alpha=.18)
ax2.fill_between(plies, 0.5, dec, where=dec < .5, color=C_BLACK, alpha=.18)
ax2.axhline(.5, color="#666", ls="--", lw=1)
ax2.set_ylabel("Decisive")
ax2.set_xlabel("Ply")
ax2.set_ylim(0, 1)
ax2.grid(alpha=.25)
save(fig, "fig5_prediction_over_game.png")

# ---------------------------------------------------------------- 6. inference latency
import time
b = chess.Board()
Xb = SCALER.transform(extract_features(b).reshape(1, -1))
MODEL.predict_proba(Xb)
t0 = time.perf_counter()
N = 300
for _ in range(N):
    MODEL.predict_proba(Xb)
lat = (time.perf_counter() - t0) / N * 1000
print(f"inference latency: {lat:.3f} ms/position")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.bar(["predict_proba"], [lat], color="#4f46e5", width=.45)
ax.set_ylabel("milliseconds")
ax.set_ylim(0, max(lat * 1.6, 0.1))
ax.set_title("Single-position inference")
ax.text(0, lat * 1.05, f"{lat:.3f} ms", ha="center", color="#eee", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=.3)
save(fig, "fig6_inference_latency.png")

print("\nall figures written to", OUT)
