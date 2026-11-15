#!/usr/bin/env python3
"""
The El Nino Alibi - Trend Court for Indonesia
NASA Space Apps Challenge 2026 - "Be An Earth System Trend Detective!"

Every trend found in NASA data is put on trial. It is only called real
("GUILTY") if it survives three alibis:
  1. El Nino alibi     - the trend is still there after removing ENSO (NOAA ONI)
  2. Noise alibi       - modified Mann-Kendall test (Hamed & Rao 1998), which
                         corrects for year-to-year autocorrelation
  3. Cherry-pick alibi - the trend keeps its sign and stays significant no
                         matter which start year you choose

Data:
  NASA POWER daily point API (MERRA-2 meteorology)
  NOAA CPC Oceanic Nino Index (ONI)

Usage:
  pip install numpy scipy requests
  python trend_court.py              # download (cached in ./cache) + analyse
  python trend_court.py --offline    # only use files already in ./cache

Output:
  data.js   (loaded by index.html - just open index.html in a browser)
  data.json (same content, for anyone who wants the raw results)
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

SITES = [
    # id, name, lat, lon, note
    ("banda-aceh", "Banda Aceh", 5.55, 95.32, "Aceh, northern tip of Sumatra"),
    ("medan", "Medan", 3.59, 98.67, "North Sumatra"),
    ("padang", "Padang", -0.95, 100.35, "West Sumatra coast"),
    ("palembang", "Palembang", -2.98, 104.75, "South Sumatra, peatland fires"),
    ("jakarta", "Jakarta", -6.21, 106.85, "Capital, Java"),
    ("surabaya", "Surabaya", -7.25, 112.75, "East Java"),
    ("denpasar", "Denpasar", -8.65, 115.22, "Bali"),
    ("kupang", "Kupang", -10.17, 123.61, "East Nusa Tenggara, driest region"),
    ("pontianak", "Pontianak", -0.03, 109.33, "West Kalimantan, on the equator"),
    ("balikpapan", "Balikpapan", -1.27, 116.83, "East Kalimantan"),
    ("makassar", "Makassar", -5.15, 119.43, "South Sulawesi"),
    ("manado", "Manado", 1.47, 124.84, "North Sulawesi"),
    ("ambon", "Ambon", -3.70, 128.18, "Maluku, wet season in mid-year"),
    ("jayapura", "Jayapura", -2.53, 140.72, "Papua"),
]

PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX"]
FIRST_YEAR = 1981
LAST_YEAR = dt.date.today().year - 1      # last complete calendar year
BASELINE = (1981, 2010)                   # for the "hot day" threshold
MIN_WINDOW = 20                           # shortest record in the cherry-pick test
ALPHA = 0.05                              # significance level
ROBUST_SHARE = 0.70                       # share of start years that must stay significant
MAX_MISSING = 0.05                        # max share of missing days per window

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Seasons differ across Indonesia (Ambon is wettest in mid-year, Java is driest),
# so each site's dry season is found from its own 1981-2010 rainfall.
# enso = which months of the El Nino index to compare with:
#   "year" = Jan-Dec, "dry4" = the site's driest 4 months,
#   "spell" = the site's dry half-year, "onset" = the 3 months where the search starts
VARS = {
    "rain_annual": dict(
        name="Annual rainfall", short="Rain / year", unit="mm", digits=0, enso="year",
        up="the area is getting wetter", down="the area is getting drier"),
    "rain_dry": dict(
        name="Dry-season rainfall (driest 4 months)", short="Dry-season rain", unit="mm", digits=0, enso="dry4",
        up="the dry season is getting wetter", down="the dry season is getting drier"),
    "dry_spell": dict(
        name="Longest dry spell (dry half-year)", short="Dry spell", unit="days", digits=1, enso="spell",
        up="dry spells are getting longer", down="dry spells are getting shorter"),
    "onset": dict(
        name="Rainy-season start (BMKG rule)", short="Rain start", unit="days", digits=1, enso="onset",
        up="the rainy season is starting later", down="the rainy season is starting earlier"),
    "t2m": dict(
        name="Mean air temperature", short="Temp.", unit="\u00b0C", digits=2, enso="year",
        up="it is warming", down="it is cooling"),
    "hot_days": dict(
        name="Hot days (above 1981-2010 90th percentile)", short="Hot days", unit="days", digits=1, enso="year",
        up="hot days are becoming more common", down="hot days are becoming rarer"),
}

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ONI_URLS = [
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "https://psl.noaa.gov/data/correlation/oni.data",
]
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]   # centre month = index + 1

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Downloading (with cache)
# ----------------------------------------------------------------------------

def _get(url, params=None, tries=4):
    import requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:          # network hiccup - retry
            last = str(e)
        wait = 5 * (k + 1)
        print(f"    retry in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"Download failed: {url}\n{last}")


def fetch_power(site, cache_dir, offline):
    sid, name, lat, lon, _ = site
    path = os.path.join(cache_dir, f"power_{sid}.json")
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if offline or age_days < 7:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    if offline:
        sys.exit(f"--offline given but {path} is missing")
    end = dt.date.today() - dt.timedelta(days=10)
    print(f"  NASA POWER: {name} ({lat}, {lon}) {FIRST_YEAR}-{end} ...")
    r = _get(POWER_URL, params={
        "parameters": ",".join(PARAMS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{FIRST_YEAR}0101",
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    })
    js = r.json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f)
    return js


def fetch_oni(cache_dir, offline):
    path = os.path.join(cache_dir, "oni.txt")
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if offline or age_days < 7:
            with open(path, encoding="utf-8") as f:
                return parse_oni(f.read())
    if offline:
        sys.exit(f"--offline given but {path} is missing")
    for url in ONI_URLS:
        try:
            print(f"  NOAA ONI: {url}")
            text = _get(url).text
            monthly = parse_oni(text)
            if len(monthly) > 100:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                return monthly
        except Exception as e:
            print(f"    failed: {e}")
    sys.exit("Could not download the ONI index. Download "
             f"{ONI_URLS[0]} by hand and save it as {path}")


def parse_oni(text):
    """Return {(year, month): anomaly} from CPC oni.ascii.txt or PSL oni.data."""
    monthly = {}
    if "SEAS" in text:                                   # CPC format
        for line in text.splitlines():
            p = line.split()
            if len(p) == 4 and p[0] in SEASONS:
                monthly[(int(p[1]), SEASONS.index(p[0]) + 1)] = float(p[3])
    else:                                                # PSL format
        for line in text.splitlines():
            p = line.split()
            if len(p) == 13 and p[0].isdigit() and 1800 < int(p[0]) < 2200:
                for m, v in enumerate(p[1:], start=1):
                    v = float(v)
                    if v > -90:
                        monthly[(int(p[0]), m)] = v
    return monthly


# ----------------------------------------------------------------------------
# Turning daily data into yearly indicators
# ----------------------------------------------------------------------------

def to_daily(js):
    par = js["properties"]["parameter"]
    keys = sorted(par[PARAMS[0]].keys())
    d = {"dates": [dt.date(int(k[:4]), int(k[4:6]), int(k[6:8])) for k in keys]}
    for p in PARAMS:
        src = par.get(p, {})
        arr = np.array([src.get(k, -999.0) for k in keys], dtype=float)
        arr[arr <= -990] = np.nan
        d[p] = arr
    d["year"] = np.array([x.year for x in d["dates"]])
    d["month"] = np.array([x.month for x in d["dates"]])
    return d


def _window(arr, mask, expected):
    v = arr[mask]
    if v.size < expected * (1 - MAX_MISSING) or np.isnan(v).mean() > MAX_MISSING:
        return None
    return v


def longest_run(boolean):
    best = cur = 0
    for b in boolean:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def dasarian_starts(date_from, date_to):
    """BMKG dasarian = 10-day blocks: days 1-10, 11-20, 21-end of month."""
    out = []
    y, m = date_from.year, date_from.month
    while dt.date(y, m, 1) <= date_to:
        for day in (1, 11, 21):
            out.append(dt.date(y, m, day))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def onset_day(daily, year, index_of, start_month):
    """Rainy-season onset (BMKG): first dasarian with >= 50 mm that is followed
    by two more dasarians with >= 50 mm. Searched for 9 months from the 1st of
    `start_month` in `year`. Returns days after that date, or None."""
    rain = daily["PRECTOTCORR"]
    begin = dt.date(year, start_month, 1)
    end_m = start_month + 8
    end = dt.date(year + (end_m > 12), (end_m - 1) % 12 + 1, 28)
    starts = dasarian_starts(begin, end)
    sums = []
    for s in starts:
        if s.day == 21:
            nxt = dt.date(s.year + (s.month == 12), s.month % 12 + 1, 1)
        else:
            nxt = s + dt.timedelta(days=10)
        i0, i1 = index_of.get(s), index_of.get(nxt - dt.timedelta(days=1))
        if i0 is None or i1 is None:
            sums.append(np.nan)
            continue
        block = rain[i0:i1 + 1]
        sums.append(np.nan if np.isnan(block).mean() > 0.2 else np.nanmean(block) * block.size)
    for i in range(len(sums) - 2):
        a, b, c = sums[i:i + 3]
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            return None                      # gap in the data - can't judge this year
        if a >= 50 and b >= 50 and c >= 50:
            return (starts[i] - begin).days
    return None


def site_seasons(daily):
    """Find this site's dry season from its 1981-2010 average monthly rainfall."""
    yr, mo, rain = daily["year"], daily["month"], daily["PRECTOTCORR"]
    base = (yr >= BASELINE[0]) & (yr <= BASELINE[1])
    clim = np.array([np.nanmean(rain[base & (mo == m)]) for m in range(1, 13)])
    # driest 4 consecutive months that stay inside one calendar year
    s4 = min(range(1, 10), key=lambda m: clim[m - 1:m + 3].sum())
    dry4 = list(range(s4, s4 + 4))
    s6 = min(max(s4 - 1, 1), 7)
    spell = list(range(s6, s6 + 6))
    onset_start = dry4[2]                      # start looking in the 3rd dry month
    return dict(dry4=dry4, spell=spell, onset_start=onset_start,
                onset_enso=[onset_start, onset_start + 1, onset_start + 2],
                clim=[round(float(v) * 30.4, 0) for v in clim])


def yearly_indicators(daily, seasons):
    yr, mo = daily["year"], daily["month"]
    rain, t2m, tmax = daily["PRECTOTCORR"], daily["T2M"], daily["T2M_MAX"]
    index_of = {d: i for i, d in enumerate(daily["dates"])}
    base = (yr >= BASELINE[0]) & (yr <= BASELINE[1])
    hot_thr = float(np.nanpercentile(tmax[base], 90)) if np.any(base) else np.nan
    n_dry = sum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1] for m in seasons["dry4"])
    n_spell = sum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1] for m in seasons["spell"])

    out = {k: {} for k in VARS}
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        ny = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        m_year = yr == y
        m_dry = m_year & np.isin(mo, seasons["dry4"])
        m_spell = m_year & np.isin(mo, seasons["spell"])

        v = _window(rain, m_year, ny)
        out["rain_annual"][y] = None if v is None else float(np.nanmean(v) * v.size)
        v = _window(rain, m_dry, n_dry)
        out["rain_dry"][y] = None if v is None else float(np.nanmean(v) * v.size)
        v = _window(rain, m_spell, n_spell)
        out["dry_spell"][y] = None if v is None else float(longest_run(np.nan_to_num(v, nan=99.0) < 1.0))
        v = _window(t2m, m_year, ny)
        out["t2m"][y] = None if v is None else float(np.nanmean(v))
        v = _window(tmax, m_year, ny)
        out["hot_days"][y] = None if (v is None or np.isnan(hot_thr)) else float(np.nansum(v > hot_thr))
        out["onset"][y] = onset_day(daily, y, index_of, seasons["onset_start"])
    return out, hot_thr


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def sen_slope(t, y):
    i, j = np.triu_indices(len(t), 1)
    slopes = (y[j] - y[i]) / (t[j] - t[i])
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * t))
    return slope, intercept


def mk_modified(t, y):
    """Mann-Kendall trend test with the Hamed & Rao (1998) variance correction
    for autocorrelation. Only positive autocorrelation is allowed to change the
    variance (correction factor >= 1), so the test can only get stricter."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 8:
        return dict(n=n, slope=0.0, intercept=float(np.mean(y)) if n else 0.0,
                    p=1.0, p_plain=1.0, z=0.0, factor=1.0)
    i, j = np.triu_indices(n, 1)
    s = float(np.sign(y[j] - y[i]).sum())
    _, counts = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    slope, intercept = sen_slope(t, y)

    ranks = stats.rankdata(y - slope * t)
    ranks = ranks - ranks.mean()
    denom = float(np.sum(ranks ** 2))
    acc = 0.0
    if denom > 0:
        lim = 1.96 / math.sqrt(n)
        for k in range(1, n - 2):
            rk = float(np.sum(ranks[:-k] * ranks[k:]) / denom)
            if abs(rk) > lim:
                acc += (n - k) * (n - k - 1) * (n - k - 2) * rk
    factor = max(1.0, 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * acc)
    var_star = var_s * factor
    if var_star <= 0 or s == 0:
        z, p, p_plain = 0.0, 1.0, 1.0
    else:
        z = (s - math.copysign(1, s)) / math.sqrt(var_star)
        p = float(2 * stats.norm.sf(abs(z)))
        z0 = (s - math.copysign(1, s)) / math.sqrt(var_s)
        p_plain = float(2 * stats.norm.sf(abs(z0)))   # without the autocorrelation fix
    return dict(n=n, slope=slope, intercept=intercept, p=p, p_plain=p_plain,
                z=float(z), factor=float(factor))


def remove_enso(t, y, e):
    """Fit y = a + b*time + c*ONI. Remove only the ONI part (c*ONI), keep the rest.
    Time is in the fit so the ENSO effect is not confused with a trend."""
    n = len(y)
    X = np.column_stack([np.ones(n), t - t.mean(), e])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - 3, 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = float(math.sqrt(max(cov[2, 2], 1e-30)))
        p_enso = float(2 * stats.t.sf(abs(beta[2] / se), dof))
    except np.linalg.LinAlgError:
        p_enso = 1.0
    adjusted = y - beta[2] * (e - e.mean())

    # how much of the year-to-year wobble ENSO explains (after removing linear trends)
    def detrend(v):
        return v - np.polyval(np.polyfit(t, v, 1), t)
    yd, ed = detrend(y), detrend(e)
    if np.std(yd) > 0 and np.std(ed) > 0:
        r = float(np.corrcoef(yd, ed)[0, 1])
    else:
        r = 0.0
    return adjusted, float(beta[2]), p_enso, r


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, p[idx] * n / rank)
        q[idx] = prev
    return q


def judge(var_key, site_name, years, raw, enso_series, enso_label, unit_note=""):
    meta = VARS[var_key]
    t = np.array(years, float)
    y = np.array(raw, float)
    e = np.array(enso_series, float)

    raw_test = mk_modified(t, y)
    adj, enso_coef, enso_p, enso_r = remove_enso(t, y, e)
    adj_test = mk_modified(t, adj)

    # cherry-pick alibi: change the start year
    windows = []
    for start in range(int(t[0]), int(t[-1]) - MIN_WINDOW + 2):
        m = t >= start
        if m.sum() < MIN_WINDOW * 0.9:
            continue
        w = mk_modified(t[m], adj[m])
        windows.append(dict(start=start, slope10=round(w["slope"] * 10, 4), p=round(w["p"], 4)))
    full_sign = np.sign(adj_test["slope"])
    if windows and full_sign != 0:
        same_sign = all(np.sign(w["slope10"]) in (full_sign, 0) for w in windows)
        sig_share = float(np.mean([w["p"] < ALPHA for w in windows]))
    else:
        same_sign, sig_share = False, 0.0
    robust = bool(same_sign and sig_share >= ROBUST_SHARE)

    raw_sig = raw_test["p"] < ALPHA
    adj_sig = adj_test["p"] < ALPHA
    if adj_sig and robust:
        verdict = "GUILTY"
    elif adj_sig:
        verdict = "NOT PROVEN"
    elif raw_sig:
        verdict = "ACQUITTED"
    else:
        verdict = "NO CASE"
    unmasked = bool(adj_sig and not raw_sig)

    # plain-language sentence
    d = meta["digits"]
    slope10 = adj_test["slope"] * 10
    direction = meta["up"] if slope10 > 0 else meta["down"]
    period = f"{years[0]}-{years[-1]}"
    if verdict == "GUILTY":
        text = (f"In {site_name}, {direction}: about {slope10:+.{d}f} {meta['unit']} per decade "
                f"({period}). The trend survives all three alibis — it is not El Niño, "
                f"not random noise, and not a cherry-picked start year.")
        if unmasked:
            text += " El Niño was actually hiding this trend in the raw data."
    elif verdict == "NOT PROVEN":
        text = (f"In {site_name}, the data suggest {direction} ({slope10:+.{d}f} {meta['unit']} per decade, "
                f"{period}), but the trend depends on which start year you pick, "
                f"so the court cannot convict.")
    elif verdict == "ACQUITTED":
        rs = raw_test["slope"] * 10
        dir_raw = meta["up"] if rs > 0 else meta["down"]
        text = (f"In {site_name}, the raw data look like {dir_raw} ({rs:+.{d}f} {meta['unit']} per decade), "
                f"but once El Niño and La Nina are removed the trend is no longer significant. "
                f"El Niño's alibi holds.")
    else:
        text = (f"In {site_name}, there is no significant trend in {meta['name'].lower()} "
                f"({period}). Changes are within natural ups and downs.")

    return dict(
        verdict=verdict, unmasked=unmasked, text=text, enso_label=enso_label, unit_note=unit_note,
        years=[int(v) for v in years],
        raw=[round(float(v), 4) for v in y],
        adj=[round(float(v), 4) for v in adj],
        enso=[round(float(v), 3) for v in e],
        raw_test={k: (round(v, 6) if isinstance(v, float) else v) for k, v in raw_test.items()},
        adj_test={k: (round(v, 6) if isinstance(v, float) else v) for k, v in adj_test.items()},
        enso_coef=round(enso_coef, 5), enso_p=round(enso_p, 6), enso_r=round(enso_r, 4),
        windows=windows, robust=robust, same_sign=bool(same_sign),
        sig_share=round(sig_share, 3),
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="use only files already in ./cache")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    print("1/3 Downloading data")
    oni = fetch_oni(args.cache, args.offline)

    def season_oni(year, months):
        # months above 12 belong to the next year (e.g. Nov-Dec-Jan)
        vals = [oni.get((year + (m - 1) // 12, (m - 1) % 12 + 1)) for m in months]
        return None if any(v is None for v in vals) else float(np.mean(vals))

    print("2/3 Building yearly indicators and holding trials")
    cases = {}
    site_meta = []
    hot_thresholds = {}
    for site in SITES:
        sid, name, lat, lon, note = site
        daily = to_daily(fetch_power(site, args.cache, args.offline))
        sea = site_seasons(daily)
        ind, thr = yearly_indicators(daily, sea)
        hot_thresholds[sid] = round(thr, 2) if not np.isnan(thr) else None
        mrange = lambda ms: f"{MONTHS[(ms[0] - 1) % 12]}-{MONTHS[(ms[-1] - 1) % 12]}"
        enso_months = {"year": list(range(1, 13)), "dry4": sea["dry4"],
                       "spell": sea["spell"], "onset": sea["onset_enso"]}
        site_meta.append(dict(id=sid, name=name, lat=lat, lon=lon, note=note,
                              dry_season=mrange(sea["dry4"]), monthly_rain=sea["clim"],
                              onset_from=f"1 {MONTHS[sea['onset_start'] - 1]}"))
        for vk, meta in VARS.items():
            months = enso_months[meta["enso"]]
            years, raw, en = [], [], []
            for y in range(FIRST_YEAR, LAST_YEAR + 1):
                v = ind[vk].get(y)
                e = season_oni(y, months)
                if v is not None and e is not None and not np.isnan(v):
                    years.append(y)
                    raw.append(v)
                    en.append(e)
            if len(years) < MIN_WINDOW + 5:
                print(f"   skip {name} / {vk}: only {len(years)} usable years")
                continue
            if vk == "onset" and np.mean(np.array(raw) == 0) >= 0.5:
                print(f"   skip {name} / onset: rain all year, no clear dry season")
                continue
            note_u = f"counted from {site_meta[-1]['onset_from']}" if vk == "onset" else ""
            cases[f"{sid}|{vk}"] = judge(vk, name, years, raw, en,
                                         f"{mrange(months)} ONI", note_u)
        print(f"   {name}: done")

    # field check: Benjamini-Hochberg false discovery rate over all trials
    keys = list(cases)
    if keys:
        q = benjamini_hochberg([cases[k]["adj_test"]["p"] for k in keys])
        for k, qv in zip(keys, q):
            cases[k]["q"] = round(float(qv), 6)

    tally = {}
    for c in cases.values():
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1

    result = dict(
        title="The El Niño Alibi — Trend Court for Indonesia",
        generated=dt.datetime.now().isoformat(timespec="minutes"),
        period=[FIRST_YEAR, LAST_YEAR],
        settings=dict(alpha=ALPHA, min_window=MIN_WINDOW, robust_share=ROBUST_SHARE,
                      baseline=list(BASELINE)),
        sites=site_meta,
        vars={k: dict(name=v["name"], short=v["short"], unit=v["unit"], digits=v["digits"])
              for k, v in VARS.items()},
        hot_thresholds=hot_thresholds,
        cases=cases,
        tally=tally,
        sources=[
            dict(name="NASA POWER daily point API (MERRA-2)",
                 url="https://power.larc.nasa.gov/"),
            dict(name="NOAA CPC Oceanic Niño Index (ONI)",
                 url="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"),
        ],
    )

    print("3/3 Writing results")
    with open(os.path.join(HERE, "data.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    with open(os.path.join(HERE, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.TREND_COURT = ")
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"   {len(cases)} trials: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print("Done. Open index.html in your browser.")


if __name__ == "__main__":
    main()
