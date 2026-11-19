# XAUUSD percentile HMM: first chronological evaluation

- Requested data: 2025-11-12 00:00 to 2026-11-12 00:00 JST (end exclusive).
- 354,366 M1 bars; 338,668 valid feature observations.
- W1 window/lag: 30/30 minutes; slope uses the current 30-minute OHLC window.
- Train: before 2026-08-31 00:00 JST, 269,165 observations in 212 sequences.
- Test origins: from 2026-08-31 01:00 JST after a 60-minute embargo.
- Training percentile quintiles for each signed feature: 25 observable pairs.
- Four hidden states, three initializations (seeds 42/43/44), up to 100 EM
  iterations. Best training likelihood selects seed 42 (58 iterations).
- Causal forward filtering; market gaps split sequences. Future test features
  update the filter only after each feature becomes observable.

| Horizon | Test pairs | HMM accuracy | Persistence accuracy | Observable Markov accuracy |
|---|---:|---:|---:|---:|
| 1 minute | 69,450 | 18.48% | 81.49% | 81.49% |
| 5 minutes | 69,238 | 11.94% | 33.76% | 33.76% |
| 15 minutes | 68,708 | 6.75% | 9.83% | 13.28% |
| 30 minutes | 67,913 | 4.61% | 5.70% | 9.47% |

This initial four-state HMM has **no advantage in joint-class accuracy** over
either baseline at any evaluated horizon. These are observed feature-class
forecasts, not price-direction forecasts or evidence of trading profitability.
At 30 minutes HMM log loss (3.166) is slightly better than observable Markov
(3.200); it is worse at the shorter horizons. See `report.json` for all
probability metrics, changed-class diagnostics and initialization details.
Do not select new hyperparameters against this same holdout and still call
the resulting score an untouched test result; use a training validation
period or a separate future evaluation for model development.

Reproduce from the repository root:

```powershell
python -m pip install -r requirements.txt
python src/train_hmm.py --start 2025-11-12 --end 2026-11-12
```

The committed JSON files are historical checkpoints. The large per-origin
forecast CSVs are generated under `outputs/hmm` by this command and are not
committed. `latest_forecast.json` refers to the last historical observation,
not current prices. Hidden state IDs are arbitrary.
