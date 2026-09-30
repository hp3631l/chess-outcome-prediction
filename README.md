# Real-Time Chess Outcome Prediction Using Machine Learning

A playable 2-player chess web app with a live ML win-probability prediction that
updates after every move.

**Full write-up: [REPORT.md](REPORT.md)** · **PDF: [report.pdf](report.pdf)**

## Run it

```bash
python -m venv venv
source venv/bin/activate
pip install chess xgboost scikit-learn pandas numpy matplotlib flask

python train_model.py    # ~135 s -> model.pkl, scaler.pkl, positions_dataset.csv
python app.py            # then open http://127.0.0.1:5000
```

## Other scripts

| Script | Purpose |
|---|---|
| `sanity_check.py` | Correctness checks: oracle symmetry, material monotonicity, mate detection, a real game |
| `make_figures.py` | Regenerates everything in `figures/` |
| `verify.py` | Headless browser test — drives real clicks, saves `shot_*.png` |
| `build_report.py` | Renders `REPORT.md` -> `report.html` (self-contained) |

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
