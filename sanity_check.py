"""Independent, correct sanity check of the trained model."""
import pickle
import chess
import numpy as np
from train_model import extract_features, PIECE_VALUES

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


def mat_diff(b):
    return (sum(PIECE_VALUES[p] * len(b.pieces(p, chess.WHITE)) for p in PIECE_VALUES)
            - sum(PIECE_VALUES[p] * len(b.pieces(p, chess.BLACK)) for p in PIECE_VALUES)) / 100.0


def predict(b):
    p = model.predict_proba(scaler.transform(extract_features(b).reshape(1, -1)))[0]
    return p


def with_extra(white_pieces=(), black_pieces=()):
    """Start position + extra pieces placed on empty squares. Kings untouched."""
    b = chess.Board()
    empties = [s for s in chess.SQUARES if b.piece_at(s) is None]

    def place(spec, color):
        nonlocal empties
        for pt in spec:
            # find an empty square that keeps the king safe-ish (mid-board)
            for s in empties:
                if chess.square_rank(s) in (3, 4) and chess.square_file(s) in (2, 3, 4, 5):
                    b.set_piece_at(s, chess.Piece(pt, color))
                    empties.remove(s)
                    break

    place(white_pieces, chess.WHITE)
    place(black_pieces, chess.BLACK)
    return b


print("=" * 74)
print("SANITY CHECK 1: material advantage vs predicted win probability")
print("=" * 74)
print(f"{'scenario':30s} {'WhiteMat':>9s} {'P(black)':>9s} {'P(draw)':>9s} {'P(white)':>9s}")

Q, R, B, N, P = chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN

series_w = []
cases = [
    ("start position",            ()),
    ("White + Pawn",              (P,)),
    ("White + Pawn + Pawn",       (P, P)),
    ("White + Knight",            (N,)),
    ("White + Bishop",            (B,)),
    ("White + Rook",              (R,)),
    ("White + Queen",             (Q,)),
    ("White + Queen + Rook",      (Q, R)),
    ("White + Q + R + B + N",     (Q, R, B, N)),
]
for name, spec in cases:
    b = with_extra(white_pieces=spec)
    p = predict(b)
    print(f"{name:30s} {mat_diff(b):+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
    series_w.append(p[2])

ok = all(series_w[i] <= series_w[i+1] + 2e-3 for i in range(len(series_w)-1))
print(f"\n{'PASS' if ok else 'FAIL'}: P(White) is monotonically non-decreasing in White's material advantage")

print()
print("=" * 74)
print("SANITY CHECK 2: symmetric positions must give symmetric probabilities")
print("=" * 74)
# board with white queen on d4 and black queen on d5 (equal material, symmetric-ish)
sym = chess.Board("8/8/8/3q4/3Q4/8/8/3k1K2 w - - 0 1")
p = predict(sym)
print(f"{'Qd4 vs Qd5, equal material':30s} {mat_diff(sym):+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
print("  -> P(black) and P(white) should both be low, P(draw) high (roughly equal material)")

# colour-swapped: same position, colours flipped -> probs should mirror
sw = chess.Board("8/8/8/3Q4/3q4/8/8/3K1k2 w - - 0 1")
p2 = predict(sw)
print(f"{'Qd5(white) vs Qd4(black)':30s} {mat_diff(sw):+9.1f} {p2[0]:9.3f} {p2[1]:9.3f} {p2[2]:9.3f}")

print()
print("=" * 74)
print("SANITY CHECK 3: terminal positions")
print("=" * 74)
def play(moves_san):
    b = chess.Board()
    for m in moves_san.split():
        b.push_san(m)
    return b


fools = play("f3 e5 g4 Qh4")
print("Fool's mate (1.f3 e5 2.g4 Qh4#) -> checkmate =", fools.is_checkmate())
p = predict(fools)
print(f"  {'  ->':30s} {mat_diff(fools):+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
print("  -> P(black) should be high (Black won by mate)")

scholar = play("e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7")
print("Scholar's mate (4.Qxf7#)       -> checkmate =", scholar.is_checkmate())
p = predict(scholar)
print(f"  {'  ->':30s} {mat_diff(scholar):+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
print("  -> P(white) should be high (White won by mate)")

# White wins a queen cleanly
wq = chess.Board("4k3/8/8/3q4/8/8/4Q3/4K3 w - - 0 1")
p = predict(wq)
print(f"  {'White Qe2 vs Black Qd5':30s} {mat_diff(wq):+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")

print()
print("=" * 74)
print("SANITY CHECK 4: does the model track a REAL game?")
print("=" * 74)
# Opera Game (Morphy 1858) - verified move by move
seq = "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#"
moves = seq.split()
b = chess.Board()
p = predict(b)
print(f"  ply  0  mat={mat_diff(b):+5.1f}  black={p[0]:.3f} draw={p[1]:.3f} white={p[2]:.3f}")
for i, mv in enumerate(moves):
    b.push_san(mv)
    p = predict(b)
    if i % 5 == 4 or i == len(moves) - 1:
        star = "  <- checkmate" if b.is_checkmate() else ""
        print(f"  ply {i+1:2d}  mat={mat_diff(b):+5.1f}  black={p[0]:.3f} draw={p[1]:.3f} white={p[2]:.3f}{star}")
print("  (Morphy's Opera Game - White wins. Model should end strongly in White's favour.)")
