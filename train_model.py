#!/usr/bin/env python3
"""
Train an ML chess evaluation model.

Task: learn to classify a position as Black-winning / Draw / White-winning.

Dataset : positions from heuristic self-play (realistic position distribution)
Target  : ground truth from a capture-search (quiescence) evaluator + material margin,
          NOT the noisy "eventual game result" of the position's parent game.
Features: 29 shallow, human-interpretable positional features (piece counts, material,
          king safety, mobility, pawn structure, control, castling, phase...)

Why not label with the parent's final result?
    Because an early, balanced position in a game that White later wins is NOT
    a "White winning" position. Labelling it as one injects pure noise and the
    model ends up anti-correlated with material. Measured: material-advantage
    sanity checks failed (White +R+R+Q predicted Black winning).
    Searching from the position itself gives a target that is actually a
    function of the position, so the learned mapping is meaningful.

Run:  python train_model.py
Out:  model.pkl, scaler.pkl, positions_dataset.csv, training_report.txt
"""
import os
import sys
import time
import random
import collections

import numpy as np
import pandas as pd
import chess
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------------------------------------------------------
# Piece values
# ----------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
VALS = list(PIECE_VALUES.values())


# ----------------------------------------------------------------------------
# Feature extraction  (29 features, all from-white-minus-black perspective)
# ----------------------------------------------------------------------------
FEATURE_NAMES = [
    "piece_diff_pawn", "piece_diff_knight", "piece_diff_bishop",
    "piece_diff_rook", "piece_diff_queen", "piece_diff_king",
    "material_diff",
    "king_pawn_shield_white", "king_pawn_shield_black",
    "mobility_white", "mobility_black",
    "center_pawns_white", "center_pawns_black",
    "doubled_pawns", "isolated_pawns", "passed_pawns",
    "open_file_rooks", "rook_on_7th",
    "bishop_pair_white", "bishop_pair_black",
    "king_zone_attack_white", "king_zone_attack_black",
    "game_phase", "in_check", "has_moved_knight",
    "castling_white", "castling_black",
    "hanging_pieces", "passed_pawn_advance",
    "is_checkmate", "is_stalemate", "side_to_move", "legal_move_count",
    "white_can_mate_now", "black_can_mate_now",
]


def extract_features(board: chess.Board) -> np.ndarray:
    """29 shallow positional features. Never mutates `board`."""
    f = []

    # 1-6: piece count differences
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        f.append(len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))

    # 7: total material difference
    mat = 0
    for pt, v in PIECE_VALUES.items():
        mat += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    f.append(mat / 100.0)

    # 8-9: king pawn shield
    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        shield = 0
        if ksq is not None:
            kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
            step = 1 if color else -1
            for df in (-1, 0, 1):
                af = kf + df
                ar = kr + step
                if not (0 <= af <= 7 and 0 <= ar <= 7):
                    continue
                p = board.piece_at(chess.square(af, ar))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
        f.append(shield)

    # 10-11: mobility (copy so we never mutate the real board)
    tmp = board.copy(stack=False)
    tmp.turn = chess.WHITE
    mob_w = len(list(tmp.legal_moves))
    tmp.turn = chess.BLACK
    mob_b = len(list(tmp.legal_moves))
    f.extend([mob_w, mob_b])

    # 12-13: centre pawn control
    centre = [chess.D4, chess.E4, chess.D5, chess.E5]
    for color in (chess.WHITE, chess.BLACK):
        f.append(sum(
            1 for sq in centre
            if board.piece_at(sq)
            and board.piece_at(sq).piece_type == chess.PAWN
            and board.piece_at(sq).color == color
        ))

    # 14-15: doubled + isolated pawns
    doubled = isolated = 0
    for color in (chess.WHITE, chess.BLACK):
        pf = [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]
        for file_i in range(8):
            c = pf.count(file_i)
            if c > 1:
                doubled += c - 1
            if c > 0:
                left = file_i > 0 and pf.count(file_i - 1) > 0
                right = file_i < 7 and pf.count(file_i + 1) > 0
                if not left and not right:
                    isolated += 1
    f.extend([doubled, isolated])

    # 16: passed pawns
    passed = 0
    for color in (chess.WHITE, chess.BLACK):
        opp = chess.BLACK if color else chess.WHITE
        for sq in board.pieces(chess.PAWN, color):
            fi, ri = chess.square_file(sq), chess.square_rank(sq)
            ahead = range(ri + 1, 8) if color else range(ri - 1, -1, -1)
            blocked = any(
                board.piece_at(chess.square(af, ar))
                and board.piece_at(chess.square(af, ar)).piece_type == chess.PAWN
                and board.piece_at(chess.square(af, ar)).color == opp
                for af in (fi - 1, fi, fi + 1) if 0 <= af <= 7
                for ar in ahead
            )
            if not blocked:
                passed += 1
    f.append(passed)

    # 17-18: rooks on open files, rook on 7th
    open_rooks = seventh = 0
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.ROOK, color):
            fi = chess.square_file(sq)
            if not any(
                board.piece_at(chess.square(fi, r))
                and board.piece_at(chess.square(fi, r)).piece_type == chess.PAWN
                for r in range(8)
            ):
                open_rooks += 1 if color else -1
            on7 = chess.square_rank(sq) == (6 if color else 1)
            seventh += 1 if on7 else -1
    f.extend([open_rooks, seventh])

    # 19-20: bishop pair
    f.append(1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0)
    f.append(1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0)

    # 21-22: enemy attacks on own king zone
    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        if ksq is None:
            f.append(0)
            continue
        att = 0
        for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]) | chess.SquareSet({ksq}):
            att += 1 if board.is_attacked_by(not color, sq) else 0
        f.append(att)

    # 23: game phase (1 = full board, 0 = bare kings)
    total = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, c))
        for pt in PIECE_VALUES for c in (chess.WHITE, chess.BLACK)
    )
    f.append(round(total / 3900.0, 3))

    # 24: side to move is in check
    f.append(1 if board.is_check() else 0)

    # 25: any knight developed
    f.append(1 if any(
        board.piece_at(s) and board.piece_at(s).piece_type == chess.KNIGHT
        and board.piece_at(s).color == chess.WHITE
        and chess.square_rank(s) >= 2
        for s in chess.SQUARES
    ) else 0)

    # 26-27: castling availability
    w_cast = int(board.has_kingside_castling_rights(chess.WHITE)) + int(board.has_queenside_castling_rights(chess.WHITE))
    b_cast = int(board.has_kingside_castling_rights(chess.BLACK)) + int(board.has_queenside_castling_rights(chess.BLACK))
    f.extend([w_cast, b_cast])

    # 28: hanging pieces (undefended material under attack)
    hanging = 0
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            for sq in board.pieces(pt, color):
                if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                    hanging += PIECE_VALUES[pt] / 100.0 * (1 if color == chess.BLACK else -1)
    f.append(hanging)

    # 29: advanced passed pawns (closer to promotion = more decisive)
    adv = 0
    for color in (chess.WHITE, chess.BLACK):
        opp = chess.BLACK if color else chess.WHITE
        for sq in board.pieces(chess.PAWN, color):
            fi, ri = chess.square_file(sq), chess.square_rank(sq)
            ahead = range(ri + 1, 8) if color else range(ri - 1, -1, -1)
            blocked = any(
                board.piece_at(chess.square(af, ar))
                and board.piece_at(chess.square(af, ar)).piece_type == chess.PAWN
                and board.piece_at(chess.square(af, ar)).color == opp
                for af in (fi - 1, fi, fi + 1) if 0 <= af <= 7
                for ar in ahead
            )
            if not blocked:
                adv += (ri if color else 7 - ri) * (1 if color else -1)
    f.append(adv)

    # 30-33: terminal / side-to-move context
    # Without these the model cannot see mate at all -- a shallow material
    # feature set reports "draw" for a finished checkmate.
    f.append(1 if board.is_checkmate() else 0)
    f.append(1 if board.is_stalemate() else 0)
    f.append(1 if board.turn == chess.WHITE else 0)
    f.append(board.legal_moves.count())

    # 34-35: can either side deliver mate on the very next move?
    # Material alone is blind to mating attacks, which is why a piece-up
    # side is regularly mated in real play. Each side is checked on its own
    # turn so the position stays legal.
    turn = board.turn
    board.turn = chess.WHITE
    w_now = 1 if _has_mating_move(board, chess.WHITE) else 0
    board.turn = chess.BLACK
    b_now = 1 if _has_mating_move(board, chess.BLACK) else 0
    board.turn = turn
    f.extend([w_now, b_now])

    assert len(f) == len(FEATURE_NAMES), f"{len(f)} != {len(FEATURE_NAMES)}"
    return np.array(f, dtype=np.float32)


# ----------------------------------------------------------------------------
# Heuristic agent for generating a realistic position distribution
# ----------------------------------------------------------------------------
def _pst(board, sq, pt, color):
    fi, ri = chess.square_file(sq), chess.square_rank(sq)
    if not color:
        ri = 7 - ri
    center = 3.5 - (abs(3.5 - fi) + abs(3.5 - ri)) / 2.0
    if pt == chess.PAWN:
        return center * 1.5 + ri * 1.2
    if pt == chess.KNIGHT:
        return center * 2.0
    if pt == chess.BISHOP:
        return center * 1.2
    if pt == chess.ROOK:
        return ri * 0.3 + (3.0 if fi in (0, 7) else 0.0)
    if pt == chess.QUEEN:
        return center * 0.8
    if pt == chess.KING:
        return (4.0 - abs(3.5 - fi) * 0.8) if ri <= 1 else center * 0.5
    return 0.0


def _move_score(board, mv, color):
    s = 0.0
    pt = board.piece_type_at(mv.from_square)
    victim = board.piece_type_at(mv.to_square)
    if victim:
        s += 12.0 * PIECE_VALUES[victim] - PIECE_VALUES[pt]
    s += _pst(board, mv.to_square, pt, color)
    if mv.promotion == chess.QUEEN:
        s += 900.0
    if board.is_castling(mv):
        s += 250.0
    board.push(mv)
    if board.is_check():
        s += 120.0
    if board.is_stalemate() or board.is_insufficient_material():
        s -= 5000.0
    board.pop()
    return s


def _pick(board, color, temperature):
    legal = list(board.legal_moves)
    if not legal:
        return None
    scores = np.array([_move_score(board, m, color) for m in legal], dtype=np.float64)
    if temperature <= 0:
        return legal[int(np.argmax(scores))]
    sc = scores / max(temperature, 1e-6)
    sc -= sc.max()
    p = np.exp(sc)
    p /= p.sum()
    return legal[int(np.random.choice(len(legal), p=p))]


def random_game(temperature=6.0, max_plies=140):
    b = chess.Board()
    for _ in range(max_plies):
        if b.is_game_over(claim_draw=False):
            break
        mv = _pick(b, b.turn, temperature)
        if mv is None:
            break
        b.push(mv)
    return b


# ----------------------------------------------------------------------------
# Ground truth oracle: a symmetric piece-square evaluation + capture search.
#
# This MUST be exactly colour-symmetric, otherwise it silently biases every
# label. The previous hand-rolled version was not: it scored a mirrored extra
# pawn at +121 for White but -151 for Black, and its capture search returned
# +366 on an equal-material opening position -- which is why the model claimed
# 97.7% White in a balanced middlegame.
# ----------------------------------------------------------------------------
# White-oriented piece-square tables, indexed by chess.square (a1 = 0).
# Black pieces read the SAME table through chess.square_mirror(), which
# guarantees eval(board) == -eval(colour_swapped_board).
_PST = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
         50,  50,  50,  50,  50,  50,  50,  50,
         10,  10,  20,  30,  30,  20,  10,  10,
          5,   5,  10,  27,  27,  10,   5,   5,
          0,   0,   0,  25,  25,   0,   0,   0,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          5,  10,  10, -25, -25,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
       -50, -40, -30, -30, -30, -30, -40, -50,
       -40, -20,   0,   0,   0,   0, -20, -40,
       -30,   0,  10,  15,  15,  10,   0, -30,
       -30,   5,  15,  20,  20,  15,   5, -30,
       -30,   0,  15,  20,  20,  15,   0, -30,
       -30,   5,  10,  15,  15,  10,   5, -30,
       -40, -20,   0,   5,   5,   0, -20, -40,
       -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
       -20, -10, -10, -10, -10, -10, -10, -20,
       -10,   0,   0,   0,   0,   0,   0, -10,
       -10,   0,   5,  10,  10,   5,   0, -10,
       -10,   5,   5,  10,  10,   5,   5, -10,
       -10,   0,  10,  10,  10,  10,   0, -10,
       -10,  10,  10,  10,  10,  10,  10, -10,
       -10,   5,   0,   0,   0,   0,   5, -10,
       -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0,
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20,
    ],
}


def static_eval(board) -> int:
    """Exactly colour-symmetric evaluation, in centipawns, from White's view."""
    score = 0
    for pt, table in _PST.items():
        base = PIECE_VALUES[pt]
        for sq in board.pieces(pt, chess.WHITE):
            score += base + table[sq]
        for sq in board.pieces(pt, chess.BLACK):
            # mirror so the table is read from Black's own perspective
            score -= base + table[chess.square_mirror(sq)]
    return int(score)


def quiesce(board, depth, alpha, beta):
    """
    Alpha-beta over capture sequences. Returns a score from the
    SIDE-TO-MOVE's perspective. Assumes board.turn is the real side to move
    (never force it -- doing so produces illegal positions).
    """
    stand = static_eval(board)
    if board.turn == chess.BLACK:
        stand = -stand

    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand
    if depth <= 0:
        return alpha

    for mv in board.legal_moves:
        if not board.is_capture(mv) and not board.gives_check(mv):
            continue
        board.push(mv)
        score = -quiesce(board, depth - 1, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def search_score_white(board, depth=5) -> int:
    """Capture search from the real side to move, converted to White's view."""
    if board.turn == chess.WHITE:
        return quiesce(board, depth, -100000, 100000)
    return -quiesce(board, depth, -100000, 100000)


def _has_mating_move(board, color) -> bool:
    """True if `color` can deliver checkmate right now."""
    for mv in list(board.legal_moves):
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def find_mate(board, color, max_depth=3):
    """
    Depth-limited search for a forced mate for `color` (True = White).
    Returns the distance to mate, or None if none is found within max_depth.

    Only moves that give check (or capture) are expanded, so the branching
    factor stays small enough to run this over tens of thousands of positions.
    """
    def rec(b, depth):
        if b.is_checkmate():
            return 0
        if depth == 0:
            return None
        for mv in list(b.legal_moves):
            # only explore forcing-ish moves
            if not (b.gives_check(mv) or b.is_capture(mv)):
                continue
            b.push(mv)
            # opponent must be able to answer; if they cannot, it is mate
            if b.is_checkmate():
                b.pop()
                return 1
            sub = None
            if b.legal_moves.count():
                sub = rec(b, depth - 1)
            b.pop()
            if sub is not None:
                return sub + 1
        return None

    board.turn = color
    return rec(board, max_depth)


def truth_label(board, depth=5, win_margin=300, loss_margin=300) -> int:
    """0 = Black winning, 1 = Draw/balanced, 2 = White winning."""

    # 1) The position is ALREADY over. Must be checked first: a checkmated
    #    position has zero legal moves, so a "mate in one" scan never fires
    #    and the position would fall through to the material search and be
    #    mislabelled "draw".
    if board.is_checkmate():
        return 0 if board.turn == chess.WHITE else 2
    if board.is_stalemate() or board.is_insufficient_material():
        return 1

    # 2) The side to move can mate right now.
    if _has_mating_move(board, board.turn):
        return 2 if board.turn == chess.WHITE else 0

    # 3) Capture search from the REAL side to move. Never force board.turn:
    #    that fabricates an illegal position (the other side has just moved,
    #    so it cannot also be in check) and inflates the score.
    score_white = search_score_white(board, depth)

    if score_white >= win_margin:
        return 2
    if score_white <= -loss_margin:
        return 0
    return 1


# ----------------------------------------------------------------------------
# Dataset build
# ----------------------------------------------------------------------------
def synthetic_mate_positions(n_wanted=400, seed=7):
    """
    Manufacture checkmate positions from random legal positions.

    Motivation: heuristic self-play almost never mates (only 39 across 700
    games), and worse, those 39 happen to have near-balanced material. So
    `is_checkmate` is almost perfectly collinear with the *normal* material
    pattern and the model has no incentive to separate them -- it keeps
    predicting from material and gets a finished mate wrong (verified: the
    Opera Game's final Rd8# was called 99.6% Black, with is_checkmate=1).

    Here we walk random legal positions and, whenever a mating move exists,
    take the position AFTER it. Because the mating position is sampled from
    ordinary play, the material is uncorrelated with who gets mated, which
    finally makes is_checkmate informative on its own.
    """
    rng = random.Random(seed)
    out = []
    b = chess.Board()
    tries = 0
    while len(out) < n_wanted and tries < n_wanted * 400:
        tries += 1
        mv = _pick(b, b.turn, rng.choice([1.0, 2.0, 4.0]))
        if mv is None:
            b = chess.Board()
            continue
        b.push(mv)
        if b.is_game_over(claim_draw=False) or b.halfmove_clock > 40:
            b = chess.Board()
            continue
        if not _has_mating_move(b, b.turn):
            continue
        for m in list(b.legal_moves):
            mover = b.turn          # side to move BEFORE pushing == mated side's opponent
            b.push(m)
            if b.is_checkmate():
                # the side that just moved (mover) delivered mate, so it wins
                out.append((extract_features(b), 2 if mover == chess.WHITE else 0))
                break
            b.pop()
        # walk on from the position regardless
    return out


def build_dataset(n_games=700, every=3, depth=5):
    t0 = time.time()
    print(f"Generating dataset from {n_games} heuristic games (quiescence depth {depth})...")
    X, y = [], []
    counts = []
    term_stats = {"checkmate": 0, "stalemate": 0, "insufficient_material": 0, "other": 0}
    for g in range(n_games):
        if g % 50 == 0:
            print(f"  game {g}/{n_games}  positions so far: {len(X)}  [{time.time()-t0:.0f}s]")
        b = random_game(temperature=random.choice([2.0, 4.0, 6.0, 9.0]))
        moves = list(b.move_stack)
        # NOTE: range(len(moves) + 1), not range(len(moves)). The position
        # after the final move lives at ply == len(moves); excluding it means
        # the model never sees a single checkmate and can never learn to
        # detect one (is_checkmate is all-zero in the dataset).
        for ply in range(len(moves) + 1):
            if ply < 8:
                continue
            pos = chess.Board()
            for mv in moves[:ply]:
                pos.push(mv)

            terminal = pos.is_game_over(claim_draw=False)
            if terminal:
                X.append(extract_features(pos))
                y.append(truth_label(pos, depth=depth))
                if pos.is_checkmate():
                    term_stats["checkmate"] += 1
                elif pos.is_stalemate():
                    term_stats["stalemate"] += 1
                elif pos.is_insufficient_material():
                    term_stats["insufficient_material"] += 1
                else:
                    term_stats["other"] += 1
                break

            if ply % every:
                continue
            if pos.is_insufficient_material():
                break
            X.append(extract_features(pos))
            y.append(truth_label(pos, depth=depth))
        counts.append(len(X))

    # Add synthetic checkmates as their own game-id bucket, so the split below
    # keeps them together (they are near-duplicates of each other, never of a
    # sampled position) and forces the same fraction into train and test.
    mates = synthetic_mate_positions(n_wanted=600)
    mate_ids = []
    for j, (feat, lab) in enumerate(mates):
        X.append(feat)
        y.append(lab)
        mate_ids.append(10_000 + j)
    print(f"  + {len(mates)} synthetic checkmate positions (was {term_stats['checkmate']})")

    print(f"  {len(X)} positions in {time.time()-t0:.0f}s")
    print(f"  label distribution: {dict(collections.Counter(y))}")
    print(f"  terminal positions included: {term_stats}")
    return np.array(X), np.array(y), counts, np.array(mate_ids, dtype=np.int32)


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(str(msg))

    log("=" * 70)
    log("CHESS EVALUATION MODEL - TRAINING REPORT")
    log("=" * 70)
    log(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__}")

    # dataset
    X, y, per_game, mate_ids = build_dataset(n_games=700, every=3, depth=5)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv("positions_dataset.csv", index=False)
    log(f"\nDataset saved: positions_dataset.csv  shape={df.shape}")
    log(f"Class counts: {dict(collections.Counter(y.tolist()))}")
    dist = {k: v / len(y) for k, v in sorted(collections.Counter(y.tolist()).items())}
    log(f"Class fractions: { {0: round(dist.get(0,0),3), 1: round(dist.get(1,3),3), 2: round(dist.get(2,0),3)} }")
    log(f"Majority-class baseline accuracy: {max(dist.values()):.4f}")

    # Split by GAME (not by position). Positions from one game are highly
    # correlated, so a random position split would leak and inflate accuracy.
    # Synthetic mates carry their own ids (>= 10_000) so they split 80/20 too.
    selfplay_ids = np.concatenate([
        np.full(c - prev, i, dtype=np.int32)
        for i, (c, prev) in enumerate(zip(per_game, [0] + per_game[:-1]))
        if c - prev > 0
    ])
    game_id = np.concatenate([selfplay_ids, mate_ids])
    log(f"Game-id vector len={len(game_id)} vs X len={len(X)} match={len(game_id)==len(X)}")

    uniq = np.unique(game_id)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq)
    test_games = set(uniq[: int(0.2 * len(uniq))].tolist())
    test_mask = np.array([g in test_games for g in game_id])

    Xtr, ytr = X[~test_mask], y[~test_mask]
    Xte, yte = X[test_mask], y[test_mask]
    log(f"Train positions: {len(Xtr)} | Test positions: {len(Xte)} (split by game, no leakage)")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Terminal positions are ~2% of the corpus even after synthetic generation.
    # A modest boost keeps them from being drowned out without letting the
    # model over-fit to a hand-built slice of the data.
    i_mat = FEATURE_NAMES.index("is_checkmate")
    i_sta = FEATURE_NAMES.index("is_stalemate")
    w = np.where((Xtr[:, i_mat] > 0) | (Xtr[:, i_sta] > 0), 5.0, 1.0)
    log(f"\nTerminal positions in train: {int((w > 1).sum())} (weight 5x)")

    log("\nTraining XGBoost (300 trees, depth 7)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(Xtr_s, ytr, sample_weight=w)

    ypred = model.predict(Xte_s)
    acc = accuracy_score(yte, ypred)
    log(f"\nTEST ACCURACY: {acc:.4f}   (majority baseline {max(dist.values()):.4f})")
    log("\nClassification report:")
    rep = classification_report(yte, ypred, target_names=["Black", "Draw", "White"], digits=4)
    log(rep)
    log("Confusion matrix (rows=true, cols=pred):")
    log(pd.DataFrame(confusion_matrix(yte, ypred), index=["Black", "Draw", "White"],
                     columns=["pred Black", "pred Draw", "pred White"]).to_string())

    # ---- material monotonicity check (the thing that failed before) ----
    log("\n" + "=" * 70)
    log("MATERIAL MONOTONICITY CHECK (White's predicted win prob vs White material)")
    log("=" * 70)
    log(f"{'scenario':34s} {'mat diff':>9s} {'P(black)':>9s} {'P(draw)':>9s} {'P(white)':>9s}")

    def probe(white_spec=(), black_spec=()):
        b = chess.Board()
        empties = [s for s in chess.SQUARES if b.piece_at(s) is None]

        def place(spec, color):
            for pt in spec:
                for s in empties:
                    if chess.square_rank(s) in (3, 4) and chess.square_file(s) in (2, 3, 4, 5):
                        b.set_piece_at(s, chess.Piece(pt, color))
                        empties.remove(s)
                        break
        place(white_spec, chess.WHITE)
        place(black_spec, chess.BLACK)
        p = model.predict_proba(scaler.transform(extract_features(b).reshape(1, -1)))[0]
        matd = (sum(PIECE_VALUES[q] * len(b.pieces(q, chess.WHITE)) for q in PIECE_VALUES)
                - sum(PIECE_VALUES[q] * len(b.pieces(q, chess.BLACK)) for q in PIECE_VALUES)) / 100.0
        return p, matd

    Q_, R_, B_, N_, P_ = chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN
    w_series = []
    for name, spec in [("start position", ()),
                       ("White + Pawn", (P_,)),
                       ("White + 2 Pawns", (P_, P_)),
                       ("White + Knight", (N_,)),
                       ("White + Bishop", (B_,)),
                       ("White + Rook", (R_,)),
                       ("White + Queen", (Q_,)),
                       ("White + Queen + Rook", (Q_, R_))]:
        p, matd = probe(spec)
        log(f"{name:34s} {matd:+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
        w_series.append(p[2])
    for name, spec in [("Black + Queen", (Q_,)),
                       ("Black + Queen + Rook", (Q_, R_))]:
        p, matd = probe(black_spec=spec)
        log(f"{name:34s} {matd:+9.1f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")

    mono = all(w_series[i] <= w_series[i + 1] + 2e-3 for i in range(len(w_series) - 1))
    log("")
    log("PASS: P(White) rises monotonically with White's material advantage."
        if mono else "FAIL: P(White) is not monotonic in White's material advantage.")

    log("\nTerminal position checks:")

    def play(san_seq):
        bb = chess.Board()
        for m in san_seq.split():
            bb.push_san(m)
        return bb

    for seq, want, lbl in [
        ("f3 e5 g4 Qh4", 0, "Fool's mate (Black won)"),
        ("e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7", 2, "Scholar's mate (White won)"),
    ]:
        bb = play(seq)
        p = model.predict_proba(scaler.transform(extract_features(bb).reshape(1, -1)))[0]
        log(f"  {lbl:34s} mate={bb.is_checkmate()} P(black)={p[0]:.3f} P(draw)={p[1]:.3f} "
            f"P(white)={p[2]:.3f}  {'PASS' if p[want] > 0.5 else 'FAIL'}")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    log("\nTop 15 features by importance:")
    for k, v in imp.head(15).items():
        log(f"  {k:28s} {v:.4f}")

    with open("model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open("scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open("training_report.txt", "w") as fh:
        fh.write("\n".join(log_lines))

    log(f"\nSaved model.pkl, scaler.pkl, positions_dataset.csv, training_report.txt")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
