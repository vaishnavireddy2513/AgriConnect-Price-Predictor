"""
AGRICONNECT - Price Predictor engine (forecasting logic).

Kept separate from app.py, like advisory_data.py / subsidy_data.py, so the model can be
reviewed, retrained and improved without touching routing code.

How it works
------------
* Data      : weekly average modal price (Rs/quintal) per (region, crop) from PriceHistory.
* Target    : log(next_week_price / this_week_price)  -> a scale-free weekly return, so ONE
              pooled model learns from all crops and regions.
* Features  : recent returns (1, 4, 12 weeks), distance from the 26-week and 1-year average,
              what the price did at the same time last year (seasonality), week-of-year
              (sin/cos), and the crop.
* Model     : GradientBoostingRegressor (scikit-learn).
* Forecast  : predict one week, feed it back in, repeat (up to 4 weeks ahead).
* Honesty   : evaluated on the last 26 weeks that the model never saw, against a naive
              "next week = this week" baseline, and shown with an 80% uncertainty band.
"""

import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from advisory_data import CROPS
from models import db, PriceHistory
from price_data import generate_demo_rows
from subsidy_data import REGIONS

PRICE_CROPS = CROPS
PRICE_REGIONS = REGIONS
ALL_REGIONS = 'All regions'          # pseudo-region = average across every region

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'price_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'price_model.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

MIN_HISTORY = 60      # weeks of data needed per series (features look back 53 weeks).
                      # ~5 years (260 weeks) per series gives noticeably better accuracy.
MAX_HORIZON = 4       # weeks ahead
TEST_WEEKS = 26       # hold-out period used for evaluation
Z80 = 1.28            # z-score for an 80% interval

_bundle_cache = {}


# ==========================================================================
# DATA ACCESS
# ==========================================================================
def seed_price_history():
    """Fill PriceHistory with synthetic demo data, only if the table is empty."""
    if PriceHistory.query.first() is not None:
        return
    db.session.bulk_save_objects([PriceHistory(**r) for r in generate_demo_rows()])
    db.session.commit()


def weekly_series(crop, region=None):
    """Weekly price Series (index = Sunday). region None / ALL_REGIONS -> average of regions."""
    q = PriceHistory.query.filter_by(crop=crop)
    if region and region != ALL_REGIONS:
        q = q.filter_by(region=region)
    rows = q.with_entities(PriceHistory.week_end, PriceHistory.modal_price).all()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    s = df.groupby('date')['price'].mean().sort_index()
    return s.resample('W-SUN').mean().interpolate(limit_direction='both')


def data_is_demo():
    """True if any demo (synthetic) rows are present, so the UI can say so."""
    return PriceHistory.query.filter_by(source='demo').first() is not None


# ==========================================================================
# FEATURES
# ==========================================================================
def _features(hist, week_date, crop):
    """
    hist      : list/array of past weekly prices, oldest -> newest (len >= 53)
    week_date : date of the week we are predicting
    """
    h = np.asarray(hist, dtype=float)
    last = h[-1]
    woy = pd.Timestamp(week_date).isocalendar()[1]
    f = [
        np.log(last / h[-2]),                 # last week's return
        np.log(last / h[-5]),                 # 4-week return
        np.log(last / h[-13]),                # 12-week return
        np.log(last / h[-26:].mean()),        # distance from 26-week average
        np.log(h[-52] / last),                # same week last year vs now
        np.log(h[-52] / h[-53]),              # last year's move into the target week
        np.log(h[-51] / h[-52]),              # last year's move after the target week
        np.log(h[-48] / h[-52]),              # last year's move over the following 4 weeks
        np.log(last / h[-52:].mean()),        # distance from the 1-year average
        np.sin(2 * np.pi * woy / 52),
        np.cos(2 * np.pi * woy / 52),
    ]
    f += [1.0 if crop == c else 0.0 for c in PRICE_CROPS]
    return f


def _build_rows(values, index, crop, first=53):
    """Training rows for one series: features at time i-1, target = return into week i."""
    X, y, pos = [], [], []
    for i in range(first, len(values)):
        X.append(_features(values[:i], index[i], crop))
        y.append(np.log(values[i] / values[i - 1]))
        pos.append(i)
    return X, y, pos


def _new_model():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        min_samples_leaf=30, subsample=0.8, random_state=42)


# ==========================================================================
# TRAINING + EVALUATION
# ==========================================================================
def train_all(verbose=True):
    """Train the pooled model, evaluate on the last TEST_WEEKS, save model + metrics."""
    series = {}
    for crop in PRICE_CROPS:
        for region in PRICE_REGIONS:
            s = weekly_series(crop, region)
            if len(s) >= MIN_HISTORY:
                series[(crop, region)] = s
    if not series:
        raise RuntimeError("No price data found. Load data first (see manage_prices.py).")

    X_tr, y_tr, X_te, y_te, naive_te = [], [], [], [], []
    for (crop, region), s in series.items():
        vals, idx = s.values, s.index
        X, y, pos = _build_rows(vals, idx, crop)
        cut = len(vals) - TEST_WEEKS
        for xi, yi, i in zip(X, y, pos):
            if i < cut:
                X_tr.append(xi); y_tr.append(yi)
            else:
                X_te.append(xi); y_te.append(yi)

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_te, y_te = np.array(X_te), np.array(y_te)

    model = _new_model().fit(X_tr, y_tr)

    # ---- 1-step residual spread (used for the uncertainty band) ----
    resid = y_te - model.predict(X_te)
    resid_sd = float(np.std(resid))

    # ---- multi-step back-test vs naive baseline (no peeking: history only) ----
    errs_model = {k: [] for k in range(1, MAX_HORIZON + 1)}
    errs_naive = {k: [] for k in range(1, MAX_HORIZON + 1)}
    for (crop, region), s in series.items():
        vals, idx = s.values, s.index
        n = len(vals)
        for o in range(n - TEST_WEEKS - 1, n - MAX_HORIZON):
            hist = list(vals[:o + 1])
            for k in range(1, MAX_HORIZON + 1):
                x = np.array([_features(hist, idx[o + k], crop)])
                nxt = hist[-1] * float(np.exp(model.predict(x)[0]))
                hist.append(nxt)
                actual = vals[o + k]
                errs_model[k].append(abs(actual - nxt) / actual)
                errs_naive[k].append(abs(actual - vals[o]) / actual)

    metrics = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'series_used': len(series),
        'train_rows': int(len(y_tr)),
        'test_weeks': TEST_WEEKS,
        'resid_sd': round(resid_sd, 5),
        'mape_model': {str(k): round(100 * float(np.mean(v)), 2) for k, v in errs_model.items()},
        'mape_naive': {str(k): round(100 * float(np.mean(v)), 2) for k, v in errs_naive.items()},
    }

    # ---- refit on ALL data for production ----
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    final = _new_model().fit(X_all, y_all)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'model': final, 'resid_sd': resid_sd}, MODEL_PATH)
    with open(METRICS_PATH, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    _bundle_cache.clear()

    if verbose:
        print("Price model trained.")
        print(json.dumps(metrics, indent=2))
    return metrics


def ensure_price_model():
    """Train on first run if no saved model exists."""
    if not os.path.exists(MODEL_PATH):
        train_all(verbose=False)


def _load_bundle():
    if 'bundle' not in _bundle_cache:
        _bundle_cache['bundle'] = joblib.load(MODEL_PATH)
    return _bundle_cache['bundle']


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return {}


# ==========================================================================
# FORECASTING
# ==========================================================================
def get_forecast(crop, region=ALL_REGIONS, weeks=4):
    """
    Returns a dict ready for the template, or None if there isn't enough data / no model.
    Prices are Rs per quintal.
    """
    weeks = max(1, min(int(weeks), MAX_HORIZON))
    s = weekly_series(crop, region)
    if len(s) < MIN_HISTORY or not os.path.exists(MODEL_PATH):
        return None

    bundle = _load_bundle()
    model, sd = bundle['model'], bundle['resid_sd']

    hist = list(s.values)
    last_date = s.index[-1]
    current = float(hist[-1])

    rows = []
    for k in range(1, weeks + 1):
        d = last_date + timedelta(weeks=k)
        x = np.array([_features(hist, d, crop)])
        nxt = hist[-1] * float(np.exp(model.predict(x)[0]))
        hist.append(nxt)
        band = Z80 * sd * np.sqrt(k)
        rows.append({
            'week_end': d.strftime('%d %b %Y'),
            'price': round(nxt, 2),
            'low': round(nxt * float(np.exp(-band)), 2),
            'high': round(nxt * float(np.exp(band)), 2),
            'change_pct': round((nxt - current) / current * 100, 1),
        })

    prices = [r['price'] for r in rows]
    best = max(rows, key=lambda r: r['price'])
    peak_gain = (max(prices) - current) / current * 100
    worst_drop = (min(prices) - current) / current * 100

    if peak_gain >= 4:
        trend = 'up'
        advice = (f"Prices are likely to rise, with the best expected week around {best['week_end']} "
                  f"(about {best['change_pct']:+.1f}% from now). If your produce stores well, "
                  f"consider selling in stages instead of all at once.")
    elif worst_drop <= -4:
        trend = 'down'
        advice = ("Prices are likely to fall over the coming weeks. If your produce is perishable "
                  "or you need cash, consider selling sooner.")
    else:
        trend = 'stable'
        advice = ("Prices look steady for the next few weeks. Choose your selling time based on "
                  "storage, transport and cash needs.")

    recent = s.tail(26)
    return {
        'crop': crop,
        'region': region,
        'weeks': weeks,
        'current_price': round(current, 2),
        'current_week': last_date.strftime('%d %b %Y'),
        'trend': trend,
        'advice': advice,
        'rows': rows,
        'hist_labels': [d.strftime('%d %b') for d in recent.index],
        'hist_values': [round(float(v), 2) for v in recent.values],
        'fc_labels': [r['week_end'][:6] for r in rows],
        'fc_values': prices,
        'fc_low': [r['low'] for r in rows],
        'fc_high': [r['high'] for r in rows],
        'metrics': load_metrics(),
        'is_demo': data_is_demo(),
    }


def get_listing_insight(listing):
    """
    For a marketplace listing: compare the farmer's asking price with the latest market price
    and give the short-term trend. Returns None if the crop isn't covered.
    """
    crop = next((c for c in PRICE_CROPS if c.lower() == listing.crop_name.strip().lower()), None)
    if crop is None:
        return None

    loc = (listing.location or '').lower()
    region = next((r for r in PRICE_REGIONS if r.lower() in loc), ALL_REGIONS)

    fc = get_forecast(crop, region, weeks=4)
    if fc is None:
        return None

    # listing price -> Rs per quintal (1 quintal = 100 kg)
    asking_per_quintal = listing.price_per_unit * (100 if listing.unit == 'kg' else 1)
    diff = (asking_per_quintal - fc['current_price']) / fc['current_price'] * 100

    return {
        'crop': crop,
        'region': region,
        'market_price': fc['current_price'],
        'asking_price': round(asking_per_quintal, 2),
        'diff_pct': round(diff, 1),
        'trend': fc['trend'],
        'is_demo': fc['is_demo'],
    }
