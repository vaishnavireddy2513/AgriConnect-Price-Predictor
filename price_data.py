"""
AGRICONNECT - Price Predictor: data helpers.

Two ways to fill the PriceHistory table:

  1. generate_demo_rows()  -> realistic *synthetic* weekly prices (seasonality + trend +
                              correlated noise). Used automatically on first run so the
                              module works out of the box. Rows are tagged source='demo'.

  2. load_csv_rows(path)   -> real mandi prices downloaded from Agmarknet / data.gov.in
                              (daily, per market). They are aggregated to weekly averages
                              per (state, crop). Rows are tagged source='agmarknet'.

All prices are in Rs per quintal (the unit Agmarknet uses).
"""

import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from advisory_data import CROPS
from subsidy_data import REGIONS

# --------------------------------------------------------------------------
# 1. SYNTHETIC DEMO DATA
# --------------------------------------------------------------------------
# base price (Rs/quintal), seasonal amplitude, peak week of year, weekly noise, yearly trend
CROP_PROFILE = {
    'Rice':      dict(base=2300,  amp=0.06, peak=44, vol=0.010, trend=0.05),
    'Wheat':     dict(base=2400,  amp=0.07, peak=16, vol=0.009, trend=0.05),
    'Tomato':    dict(base=2200,  amp=0.35, peak=28, vol=0.070, trend=0.00),
    'Cotton':    dict(base=7200,  amp=0.08, peak=5,  vol=0.012, trend=0.03),
    'Maize':     dict(base=2200,  amp=0.10, peak=10, vol=0.014, trend=0.04),
    'Chilli':    dict(base=13000, amp=0.12, peak=8,  vol=0.020, trend=0.00),
    'Sugarcane': dict(base=350,   amp=0.03, peak=1,  vol=0.004, trend=0.03),
    'Groundnut': dict(base=6200,  amp=0.07, peak=40, vol=0.012, trend=0.04),
}

REGION_FACTOR = {
    'Telangana': 1.00, 'Andhra Pradesh': 0.98, 'Karnataka': 1.02,
    'Maharashtra': 1.04, 'Punjab': 0.97, 'Uttar Pradesh': 0.96,
}


def last_sunday(today=None):
    today = today or date.today()
    return today - timedelta(days=(today.weekday() + 1) % 7)


def generate_demo_rows(weeks=260, seed=42):
    """Return a list of dicts (week_end, region, crop, modal_price, source)."""
    rng = np.random.default_rng(seed)
    end = last_sunday()
    week_ends = [end - timedelta(weeks=weeks - 1 - i) for i in range(weeks)]
    rows = []

    for crop in CROPS:
        p = CROP_PROFILE[crop]
        # one national shock series shared by all regions (so regions move together)
        national = _ar1(rng, weeks, phi=0.85, sd=p['vol'])
        for region in REGIONS:
            local = _ar1(rng, weeks, phi=0.6, sd=p['vol'] * 0.5)
            for i, d in enumerate(week_ends):
                woy = d.isocalendar()[1]
                seasonal = p['amp'] * np.cos(2 * np.pi * (woy - p['peak']) / 52)
                trend = 1 + p['trend'] * i / 52
                price = (p['base'] * REGION_FACTOR[region] * trend
                         * np.exp(seasonal + national[i] + local[i]))
                rows.append(dict(week_end=d, region=region, crop=crop,
                                 modal_price=round(float(price), 2), source='demo'))
    return rows


def _ar1(rng, n, phi, sd):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sd)
    return x


# --------------------------------------------------------------------------
# 2. REAL DATA FROM CSV (Agmarknet format)
# --------------------------------------------------------------------------
# Agmarknet commodity names differ from our crop names, so map by prefix.
COMMODITY_ALIASES = {
    'paddy': 'Rice', 'rice': 'Rice',
    'wheat': 'Wheat',
    'tomato': 'Tomato',
    'cotton': 'Cotton',
    'maize': 'Maize',
    'dry chilli': 'Chilli', 'green chilli': 'Chilli', 'chilli': 'Chilli',
    'sugarcane': 'Sugarcane',
    'groundnut': 'Groundnut',
}


def _map_crop(name):
    name = str(name).strip().lower()
    for prefix, crop in COMMODITY_ALIASES.items():
        if name.startswith(prefix):
            return crop
    return None


def load_csv_rows(path):
    """
    Expected columns (case-insensitive, extra columns are ignored):
        State, Commodity, Arrival_Date (dd/mm/yyyy), Modal_Price (Rs/quintal)
    Returns weekly-aggregated rows for our REGIONS and CROPS only.
    """
    df = pd.read_csv(path)
    df.columns = [re.sub(r'[\s\-]+', '_', c.strip().lower()) for c in df.columns]

    # tolerate a few common column spellings
    rename = {'arrival date': 'arrival_date', 'modal price': 'modal_price'}
    df = df.rename(columns=rename)
    need = {'state', 'commodity', 'arrival_date', 'modal_price'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df['date'] = pd.to_datetime(df['arrival_date'], dayfirst=True, errors='coerce')
    df['price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['region'] = df['state'].astype(str).str.strip().str.title()
    df['crop'] = df['commodity'].map(_map_crop)

    df = df.dropna(subset=['date', 'price', 'crop'])
    df = df[(df['price'] > 0) & (df['region'].isin(REGIONS))]
    if df.empty:
        raise ValueError("No usable rows: check state/commodity names and date format.")

    # drop absurd outliers per (crop, region): beyond 5x / 0.2x of the median
    med = df.groupby(['crop', 'region'])['price'].transform('median')
    df = df[(df['price'] < 5 * med) & (df['price'] > 0.2 * med)]

    df['week_end'] = df['date'].dt.to_period('W-SUN').dt.end_time.dt.date
    weekly = (df.groupby(['region', 'crop', 'week_end'])['price']
                .mean().round(2).reset_index())
    return [dict(week_end=r.week_end, region=r.region, crop=r.crop,
                 modal_price=float(r.price), source='agmarknet')
            for r in weekly.itertuples()]
