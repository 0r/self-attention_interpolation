from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "set_seed",
    "TransformerDailyChangePredictor",
    "predict_ast_scores",
    "process_and_interpolate_results",
    "ensemble_scores",
    "smooth_results",
    "make_synthetic_data",
]

# Reproducibility
def set_seed(seed_value: int = 0) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model
class TransformerDailyChangePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 1e-4,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)      # (B, T, d_model)
        out = self.transformer_encoder(x) # (B, T, d_model)
        out = out[:, -1, :]
        out = self.fc(out)                # (B, 1)
        return out

# Core prediction pipeline
# Add this updated function (drop-in replacement) with a new `interpolate_mode` option.
# Modes:
#   - "model"  : original behavior (predict daily deltas; clamp between prev/next actual)
#   - "linear" : force straight-line interpolation between every pair of assessments (pass-through)
#   - "segment_rescaled": use model daily deltas but rescale within each assessment-to-assessment segment
#                         so the cumulative change exactly equals the next assessment delta (pass-through)

def predict_ast_scores(
    sensor_df: pd.DataFrame,
    ast_df: pd.DataFrame,
    patid,
    ast_col: str,
    seed_value: int = 2015,
    lr: float = 1e-4,
    epochs: int = 50,
    interpolate_mode: str = "model",  # "model" | "linear" | "segment_rescaled"
) -> pd.DataFrame:
    set_seed(seed_value)

    sensor_pat = sensor_df[sensor_df['patid'] == patid].copy().reset_index(drop=True)
    if sensor_pat.empty:
        raise ValueError(f"No sensor data found for patid: {patid}")

    ast_pat = ast_df[ast_df['patid'] == patid].copy().reset_index(drop=True)
    if ast_pat.empty:
        raise ValueError(f"No assessment data found for patid: {patid}")

    # dates
    sensor_pat['date'] = pd.to_datetime(sensor_pat['date']).dt.date
    ast_pat['date']    = pd.to_datetime(ast_pat['date']).dt.date

    # assessment-only deltas
    ast_only = ast_pat.dropna(subset=[ast_col]).sort_values("date")[['date', ast_col]].copy()
    if ast_only.empty:
        out = sensor_pat.copy()
        out['predicted_ast_score'] = np.nan
        return out[['patid', 'date', 'predicted_ast_score']]
    ast_only['daily_change'] = ast_only[ast_col].diff().fillna(0.0)

    # merge deltas to sensor timeline (non-assessment -> 0)
    merged_df = (
        sensor_pat.merge(ast_only[['date', 'daily_change']], on='date', how='left')
                  .sort_values('date')
                  .reset_index(drop=True)
    )
    merged_df['daily_change'] = merged_df['daily_change'].fillna(0.0)

    # features/targets
    numeric_features = merged_df.select_dtypes(include=[np.number]).drop(columns=['daily_change'], errors='ignore')
    features = numeric_features.values
    features[~np.isfinite(features)] = 0.0
    targets = merged_df['daily_change'].values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Train model (needed for "model" and "segment_rescaled"; skipped for "linear")
    if interpolate_mode in ("model", "segment_rescaled") and features.shape[0] > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, shuffle=False, random_state=seed_value
        )
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        input_dim = X_train.shape[1] if X_train.shape[0] > 0 else features.shape[1]
        model = TransformerDailyChangePredictor(input_dim)
        criterion = torch.nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor.unsqueeze(1))
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        model.eval()
        with torch.no_grad():
            full_X = torch.tensor(features, dtype=torch.float32)
            daily_changes = model(full_X.unsqueeze(1)).cpu().numpy().flatten()
        pred_change_dict = dict(zip(merged_df['date'].values, daily_changes))
    else:
        pred_change_dict = {}

    # assessment anchors
    ast_obs = ast_pat.dropna(subset=[ast_col]).sort_values('date').copy()
    ast_days = ast_obs['date'].unique()
    days = sensor_pat['date'].dropna().sort_values().unique()

    common_dates = set(days).intersection(set(ast_days))
    if not common_dates:
        out = sensor_pat.copy()
        out["predicted_ast_score"] = np.nan
        return out[["patid", "date", "predicted_ast_score"]]

    # Build segments between consecutive assessments on the sensor day grid
    sorted_days = sorted(days)
    assess_in_grid = [d for d in sorted_days if d in set(ast_days)]
    if len(assess_in_grid) < 2:
        # Not enough anchors to interpolate
        out = sensor_pat.copy()
        # Fill only at the single anchor if present
        out["predicted_ast_score"] = np.nan
        only_day = assess_in_grid[0]
        only_val = float(ast_obs.set_index('date').loc[only_day, ast_col])
        out.loc[out["date"] == only_day, "predicted_ast_score"] = only_val
        return out[["patid", "date", "predicted_ast_score"]]

    ast_lookup = ast_obs.set_index('date')[ast_col].to_dict()

    # Initialize output
    out = sensor_pat.copy()
    out["predicted_ast_score"] = np.nan

    # For each segment [ai, aj] of consecutive assessments, fill per chosen mode
    for i in range(len(assess_in_grid) - 1):
        a0 = assess_in_grid[i]
        a1 = assess_in_grid[i + 1]
        y0 = float(ast_lookup[a0])
        y1 = float(ast_lookup[a1])

        # indices for days in this segment including both endpoints
        seg_days = [d for d in sorted_days if a0 <= d <= a1]
        if len(seg_days) < 2:
            continue
        n = len(seg_days) - 1  # number of steps between endpoints
        target_delta = y1 - y0

        if interpolate_mode == "linear":
            # straight-line interpolation from y0 to y1
            for k, d in enumerate(seg_days):
                val = y0 + (target_delta * (k / n)) if n > 0 else y0
                out.loc[out["date"] == d, "predicted_ast_score"] = val

        elif interpolate_mode == "segment_rescaled":
            # use model daily deltas within the segment, then rescale to match total target_delta
            raw_deltas = []
            for k in range(n):
                day_k = seg_days[k + 0]  # current day
                # take predicted change for NEXT day (k->k+1); fallback to 0
                raw_deltas.append(float(pred_change_dict.get(seg_days[k + 1], 0.0)))
            sum_raw = np.sum(raw_deltas)
            if np.isclose(sum_raw, 0.0):
                # fallback to equal steps if model is flat
                scaled = [target_delta / n] * n if n > 0 else []
            else:
                scale = target_delta / sum_raw
                scaled = [delta * scale for delta in raw_deltas]

            cur = y0
            out.loc[out["date"] == seg_days[0], "predicted_ast_score"] = cur
            for k in range(n):
                cur = cur + scaled[k]
                out.loc[out["date"] == seg_days[k + 1], "predicted_ast_score"] = cur
            # enforce exact endpoint
            out.loc[out["date"] == a1, "predicted_ast_score"] = y1

        else:  # "model" (original behavior with clamping)
            cur = y0
            out.loc[out["date"] == seg_days[0], "predicted_ast_score"] = cur
            for k in range(1, len(seg_days)):
                d = seg_days[k]
                change = float(pred_change_dict.get(d, 0.0))
                raw = cur + change
                # clamp between previous anchor and next anchor value
                lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                cur = max(lo, min(raw, hi))
                out.loc[out["date"] == d, "predicted_ast_score"] = cur
            # enforce anchor at a1
            out.loc[out["date"] == a1, "predicted_ast_score"] = y1

    return out[["patid", "date", "predicted_ast_score"]]



def process_and_interpolate_results(
    sensor_df: pd.DataFrame,
    ast_df: pd.DataFrame,
    score_columns: list[str],
    seed_value: int = 2015,
    **train_kwargs,
) -> pd.DataFrame:
    unique_pats = sensor_df['patid'].unique()
    all_results = []

    sensor_df = sensor_df.copy()
    ast_df = ast_df.copy()
    sensor_df['date'] = pd.to_datetime(sensor_df['date']).dt.date
    ast_df['date'] = pd.to_datetime(ast_df['date']).dt.date

    for patid in unique_pats:
        pat_results = []
        for score_col in score_columns:
            try:
                results = predict_ast_scores(
                    sensor_df, ast_df, patid, score_col, seed_value=seed_value, **train_kwargs
                )
                ast_actual = (
                    ast_df[(ast_df['patid'] == patid)][['date', score_col]]
                    .rename(columns={score_col: f"actual_{score_col}"})
                )
                results = (
                    results.merge(ast_actual, on='date', how='left')
                           .rename(columns={"predicted_ast_score": f"predicted_{score_col}"})
                )
                results[score_col] = results[f"actual_{score_col}"].combine_first(results[f"predicted_{score_col}"])
                results = results.drop(columns=[f"actual_{score_col}", f"predicted_{score_col}"], errors='ignore')
                results = results.drop(columns=['patid'], errors='ignore')
                pat_results.append(results)
            except Exception as e:
                print(f"Failed for patid={patid}, score={score_col}. Error: {e}")

        if pat_results:
            pat_results_df = pat_results[0]
            for df in pat_results[1:]:
                pat_results_df = pat_results_df.merge(df, on='date', how='outer')
            pat_results_df['patid'] = patid
            all_results.append(pat_results_df)

    if not all_results:
        return pd.DataFrame(columns=['patid', 'date'] + score_columns)

    return (
        pd.concat(all_results, ignore_index=True)
          .sort_values(['patid', 'date'])
          .reset_index(drop=True)
    )

def ensemble_scores(interpolated_results_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Average multiple interpolation runs into a single 'ast' trajectory.
    Expects each df to have ['patid','date','ast'].
    """
    dfs = []
    for name, df in interpolated_results_dict.items():
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.rename(columns={'ast': f"{name}_ast"})
        dfs.append(df_copy)

    # merge on patid+date
    merged = dfs[0]
    for df_next in dfs[1:]:
        merged = merged.merge(df_next, on=['patid','date'], how='outer')

    # average across all *_ast columns
    ast_cols = [c for c in merged.columns if c.endswith("_ast")]
    merged['ast'] = merged[ast_cols].mean(axis=1)

    return merged[['patid','date','ast']].sort_values(['patid','date']).reset_index(drop=True)


def smooth_results(results_df: pd.DataFrame, score_columns: list[str], window: int = 3) -> pd.DataFrame:
    smoothed_df = results_df.copy()
    for col in score_columns:
        smoothed_df[col] = (
            smoothed_df.groupby("patid")[col]
            .apply(lambda x: x.rolling(window, min_periods=1, center=True).mean())
            .reset_index(level=0, drop=True)
        )
    return smoothed_df


def make_synthetic_data(
    n_pats: int = 3,
    start_date: str = "2024-01-01",
    *,
    assessment_interval_days: int = 30,
    n_assessments: int = 12,
    seed: int = 42,
    sensor_missing_rate: float = 0.05,
    # assessment dynamics
    base_level: float = 12.0,
    regime_count_range: tuple[int, int] = (3, 5),
    regime_slope_range: tuple[float, float] = (-0.02, 0.03),
    ar_phi: float = 0.75,
    ar_noise_sd: float = 0.18,
    revert_k: float = 0.12,
    shock_prob: float = 0.20,
    shock_scale: float = 1.4,
    shock_sign_bias: float = 0.65,
    plateau_len_range: tuple[int, int] = (1, 3),
    decay_half_life_assess: float = 2.5,
    # NEW: sensor feature controls
    n_features: int = 8,                 # number of synthetic sensor features
    feature_strength: float = 1.0,       # overall coupling strength to assessment dynamics
    feature_noise_sd: float = 0.25,      # idiosyncratic feature noise (per day)
    allow_negative_corr: bool = True,    # allow negatively correlated sensors
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate daily sensors + scheduled assessments. Sensors are constructed to be
    predictive of assessments via controlled correlations with latent components
    (level, deviation, slope/velocity, and shocks). Features are named feat1..featN.
    """
    rng = np.random.default_rng(seed)

    # --- calendar ---
    if n_assessments < 2:
        raise ValueError("n_assessments must be >= 2.")
    start = pd.Timestamp(start_date)
    assess_dates = start + pd.to_timedelta(np.arange(n_assessments) * assessment_interval_days, unit="D")
    end = assess_dates[-1]
    dates = pd.date_range(start, end, freq="D")
    assess_set = set(ad.date() for ad in assess_dates)

    # shock decay fn (in assessment units)
    lam = np.log(2.0) / max(1e-6, decay_half_life_assess)
    def decay(k: int) -> float:
        return float(np.exp(-lam * max(0, k)))

    sensor_rows, ast_rows = [], []

    for patid in range(1, n_pats + 1):
        nT = len(dates)
        nA = len(assess_dates)

        # ----- piecewise baseline over assessments -----
        req_lo, req_hi = regime_count_range
        max_regimes = max(1, min(req_hi, nA - 1))
        min_regimes = max(1, min(req_lo, max_regimes))
        n_regimes = int(rng.integers(min_regimes, max_regimes + 1))

        cps_needed = n_regimes - 1
        avail = np.arange(1, max(1, nA - 1))
        if cps_needed > 0 and avail.size > 0:
            cps = np.sort(rng.choice(avail, size=min(cps_needed, avail.size), replace=False))
        else:
            cps = np.array([], dtype=int)

        knots = np.r_[0, cps, nA - 1]
        slopes = rng.uniform(*regime_slope_range, size=len(knots) - 1)

        assess_axis = np.arange(nA, dtype=float)
        regime_base = np.zeros(nA, dtype=float)
        regime_base[0] = base_level + rng.normal(0, 0.3)

        for r in range(len(slopes)):
            a0, a1 = knots[r], knots[r + 1]
            xs = assess_axis[a0:a1 + 1] - a0
            line = regime_base[a0] + slopes[r] * xs
            regime_base[a0:a1 + 1] = line
            if r + 1 < len(slopes):
                regime_base[a1] = line[-1]

        # interpolate daily baseline from assessment baseline
        day_baseline = np.interp(
            (dates - dates[0]).days,
            (assess_dates - assess_dates[0]).days,
            regime_base
        )

        # ----- daily deviations with mean reversion -----
        dev = np.zeros(nT, dtype=float)
        noise = rng.normal(0, ar_noise_sd, size=nT)
        scores = np.empty(nT, dtype=float)
        scores[0] = day_baseline[0] + rng.normal(0, 0.25)

        for t in range(1, nT):
            dev[t] = (ar_phi - revert_k) * dev[t - 1] + noise[t]
            scores[t] = day_baseline[t] + dev[t]

        # ----- shocks / plateaus / decays on assessment grid -----
        assess_effect = np.zeros(nA, dtype=float)
        a = 0
        while a < nA:
            if rng.random() < shock_prob:
                sign = 1.0 if rng.random() < shock_sign_bias else -1.0
                mag = abs(rng.normal(shock_scale, shock_scale * 0.3))
                plateau_len = int(np.clip(rng.integers(plateau_len_range[0], plateau_len_range[1] + 1),
                                          1, plateau_len_range[1]))
                for j in range(plateau_len):
                    k = a + j
                    if k >= nA: break
                    assess_effect[k] += sign * mag
                k = a + plateau_len
                tail = 0
                while k < nA and tail < 6:
                    assess_effect[k] += sign * mag * decay(tail + 1)
                    k += 1
                    tail += 1
                a += plateau_len
            else:
                a += 1

        # last-assessment index per day, add effect
        day_to_last_ass = np.searchsorted(assess_dates.values, dates.values, side="right") - 1
        day_to_last_ass = np.clip(day_to_last_ass, 0, nA - 1)
        shock_daily = assess_effect[day_to_last_ass]
        scores = scores + shock_daily

        # ----- construct predictive sensor features -----
        # Latent components to mix:
        #   level     = centered scores
        #   dev       = short-term deviation (already centered around baseline)
        #   velocity  = daily first difference of scores
        #   shock_sig = shock_daily (stepwise)
        level = scores - scores.mean()
        dev_c = dev - dev.mean()
        velocity = np.concatenate([[0.0], np.diff(scores)])
        shock_sig = shock_daily

        # standardize components
        def _std(x):
            s = np.std(x)
            return (x / (s if s > 1e-8 else 1.0))

        comps = np.vstack([
            _std(level),
            _std(dev_c),
            _std(velocity),
            _std(shock_sig),
        ])  # shape (4, nT)

        # random linear blends to create features with controlled correlation
        # Each feature = sum_j (w_j * comp_j) * feature_strength + noise
        # Some features can be negatively correlated if allowed.
        W = rng.normal(0, 0.8, size=(n_features, comps.shape[0]))  # weights per component
        if allow_negative_corr:
            sign = rng.choice([-1.0, 1.0], size=W.shape, p=[0.3, 0.7])
            W *= sign
        # normalize rows to unit L2, then scale by feature_strength
        row_norms = np.linalg.norm(W, axis=1, keepdims=True)
        row_norms[row_norms < 1e-8] = 1.0
        W = (W / row_norms) * feature_strength

        # build features
        feats = W @ comps  # (n_features, nT)
        feats += rng.normal(0, feature_noise_sd, size=feats.shape)

        # optional mild nonlinearity to diversify signals
        feats += 0.05 * np.tanh(feats)

        # simulate missing sensor days
        keep = rng.random(nT) > sensor_missing_rate
        for t_idx, (d, kflag) in enumerate(zip(dates, keep)):
            if not kflag:
                continue
            row = {"patid": patid, "date": d.date()}
            # name features feat1..featN
            for j in range(n_features):
                row[f"feat{j+1}"] = float(feats[j, t_idx])
            sensor_rows.append(row)

        # ----- write assessments only on assessment days (NaN elsewhere) -----
        for d_idx, d in enumerate(dates):
            if d.date() in assess_set:
                ast_val = float(scores[d_idx] + rng.normal(0, 0.20))
                ast_rows.append({
                    "patid": patid,
                    "date": d.date(),
                    "ast": ast_val,
                })
            else:
                ast_rows.append({"patid": patid, "date": d.date(), "ast": None})

    sensor_df = pd.DataFrame(sensor_rows)
    ast_df = pd.DataFrame(ast_rows)
    return sensor_df, ast_df




