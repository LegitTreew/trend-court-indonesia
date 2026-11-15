# The El Niño Alibi: Trend Court for Indonesia

NASA Space Apps Challenge 2026: **Be An Earth System Trend Detective!**

Is Indonesia really changing, or is it just El Niño? Each trend we find in NASA data, in 14 cities from Banda Aceh to Jayapura, is put on trial.
It is called **GUILTY** (a real trend) only if it survives three alibis:

1. **Noise alibi.** Modified Mann-Kendall test with the Hamed & Rao (1998) correction for year-to-year autocorrelation.
2. **El Niño alibi.** We fit `y = a + b·year + c·ONI`, subtract only the El Niño part (`c·ONI`), and test again.
   Sometimes El Niño was *hiding* a trend (★ in the app).
3. **Cherry-picking alibi.** We repeat the test for every start year (minimum 20 years of record). A real trend keeps its
   direction and stays significant for at least 70% of the start years.

Bonus: a Benjamini-Hochberg false-discovery-rate check across all 84 trials.

**Ask the court:** type a question in English or Bahasa Indonesia ("Is Jakarta getting hotter?", "kekeringan di Kupang") and the app opens the matching case.

| Verdict | Meaning |
|---|---|
| GUILTY | The trend is real after all three alibis |
| NOT PROVEN | Significant, but it depends on the start year |
| ACQUITTED | Looked like a trend, but El Niño explains it |
| NO CASE | No significant trend |

## Challenge questions → where the app answers them
- **What is changing?** 6 indicators: annual rainfall, dry-season rainfall, longest dry spell, rainy-season start (BMKG rule), temperature, hot days. Each city's dry season is its own driest 4 months, because seasons differ across Indonesia (Ambon is wettest mid-year).
- **Where?** Map and docket of 14 cities across Indonesia.
- **How much?** Sen's slope per decade, before and after removing El Niño.
- **Is it significant?** The three alibis, plus the false-discovery-rate check.

## Run it
```bash
python3 -m pip install --user numpy scipy requests
python3 trend_court.py     # downloads NASA POWER + NOAA ONI (cached in ./cache), writes data.js
```
Then open `index.html` in a browser. No server is needed. The map tiles need internet; everything else works offline.

## Data
- NASA POWER daily point API (MERRA-2 meteorology): https://power.larc.nasa.gov/
- NOAA CPC Oceanic Niño Index (ONI): https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

## Limits
- NASA POWER values are grid averages (about 50 km), not rain-gauge readings.
- The El Niño effect is treated as linear and same-season. The Indian Ocean Dipole is not removed yet.
- "Guilty" means the change is statistically real. It does not say what caused it.

## Files
- `trend_court.py`: download, indicators, statistics, verdicts → `data.js` / `data.json`
- `index.html`: the app (Chart.js and Leaflet are included in `vendor/`)
YOLO achievement test
