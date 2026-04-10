# Stage 3 strategy logic review (MR_cointegration.py)

## What the code is doing now

The implementation currently follows this sequence:

1. **Level strategy**
   - Build a panel of prediction and rates assets.
   - Run PCA on the level panel and compute residuals from a low-rank reconstruction.
   - Filter residual series using R² / ADF / AR(1) criteria.
   - Estimate OU parameters on a rolling residual window and trade z-score bands.

2. **Variance and skew strategies**
   - Build dedicated variance/skew panels and align market-side features.
   - Estimate linear relationships (`prediction_asset ~ market_asset`) via `_cointegration_residuals`.
   - Store residuals, then run the same rolling OU + threshold trading engine.

This means your understanding is correct: residual construction happens first (from PCA for level, from linear fit for variance/skew), then OU is estimated on rolling windows of those residuals, and z-score signals are traded.

## Is this logical?

Yes, this is directionally logical for a mean-reversion setup:
- residual extraction creates a spread-like series,
- OU maps spread dynamics to a mean/speed/vol scale,
- z-score thresholds convert that to entry/exit rules.

However, there are consistency and robustness gaps between stages.

## Recommended improvements

1. **Keep transformations consistent between train and trade windows**
   - For any rolling cointegration mode, estimate relationship parameters only with past data and apply out-of-sample.
   - Ensure normalization conventions are identical across level/variance/skew (Stage 3 now normalizes level assets before PCA).

2. **Sharpe and variance treatment**
   - Current Sharpe is trade-level mean/std without annualization and with potential small-sample instability.
   - Add minimum-trade thresholds and report both trade-level and time-aggregated (daily) Sharpe.
   - Include downside-risk metrics (Sortino, expected shortfall) for asymmetric tails.

3. **Avoid mixed timescale estimation**
   - Cointegration over long sample + OU over short sample can create regime mismatch.
   - Consider tying cointegration lookback and OU lookback to a common horizon or using explicit regime segmentation.

4. **Execution realism**
   - Add spread/slippage/fees assumptions before interpreting Sharpe and total PnL.
   - Add position sizing by residual volatility or OU sigma to stabilize risk across assets.

5. **Validation structure**
   - Add walk-forward blocks with strict train/validation/test splits by date.
   - Track signal decay by days-to-decision to verify that alpha survives near event boundaries.
