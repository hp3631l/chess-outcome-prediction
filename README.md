# Real-Time Chess Outcome Prediction Using Machine Learning

A playable 2-player chess web app with a live ML win-probability prediction that
updates after every move.

**Full write-up: [REPORT.md](REPORT.md)**

## Run it

```bash
git clone <this-repo> && cd miniproject
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python app.py      # then open http://127.0.0.1:5000
```

No training step — the trained model is committed, so this is the whole thing.
Play a game and the prediction updates after every ply.

Pinned exactly on purpose: `model.pkl` is a pickled XGBoost booster and will
not unpickle across a major upgrade of xgboost / scikit-learn / numpy.

### Optional extras

```bash
pip install -r requirements-dev.txt
playwright install chromium     # for the browser tests

python train_model.py          # retrain -> model.pkl, scaler.pkl
python make_figures.py         # regenerate figures/
python build_report.py         # REPORT.md -> report.html
```

## Headline results

| Metric | Value |
|---|---|
| Test accuracy | 95.42% (baseline 40.64%) |
| Wrong-winner predictions | 2 / 5,672 |
| Mate-related feature importance | 37.6% |
| Inference latency | 6.8 ms / position |

The interesting part is Section 5 of the report: labelling positions by their parent
game's result yields a model that is *anti-correlated with material* while still
scoring 98.7% accuracy. The fix — a symmetric, search-based oracle plus synthetic
checkmate positions — is the substance of the project.

## Scripts

| Script | Purpose |
|---|---|
| `app.py` | The Flask server and the entire UI (inlined CSS/JS, no CDN) |
| `train_model.py` | Features, oracle, dataset generation, training, sanity checks |
| `sanity_check.py` | Correctness: oracle symmetry, material monotonicity, mate detection, a real game |
| `verify.py` | Headless browser test — drives real clicks, saves `shot_*.png` |
| `verify_anim.py` | Tests the move FLIP, capture fade and board rotation |
| `contrast.py` | Measures rendered contrast in **both** themes, including piece legibility |
| `make_figures.py` | Regenerates everything in `figures/` |
| `build_report.py` | Renders `REPORT.md` -> `report.html` (self-contained) |
| `build_pieces.py` | One-off: traced the piece PNGs into SVG paths (see Attribution) |

## Attribution

Chess piece artwork in `pieces.py` is from the
[Wikipedia chess pieces by Cburnett](https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces),
licensed **CC BY-SA 3.0**. The paths were traced from the PNGs so the app has no
runtime image dependencies; `pieces.py` records the attribution string and the app
surfaces it in the "About the model" panel.

Because that artwork is share-alike, derivative redistribution of `pieces.py` is
subject to CC BY-SA 3.0. The rest of this repository is yours to license as you
like.

Position labels come from a search-based oracle and from synthetic checkmate
positions, both generated in `train_model.py` — there is no third-party game
corpus in this repo.