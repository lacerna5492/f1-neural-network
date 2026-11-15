"""Build one pre-race feature table from the F1 CSVs (no information from the race itself).
Usage: python3 -I build_table.py <data_dir> <out_csv>
Target: podium (1 if the driver finished 1st-3rd, else 0).
"""
import sys
import numpy as np
import pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
res = pd.read_csv(f'{D}/f1_results.csv')
races = pd.read_csv(f'{D}/f1_races.csv')
drivers = pd.read_csv(f'{D}/f1_drivers.csv')
dstand = pd.read_csv(f'{D}/f1_driver_standings.csv')
cstand = pd.read_csv(f'{D}/f1_constructor_standings.csv')

df = res.merge(races[['season', 'round', 'circuit_id', 'date']], on=['season', 'round'], how='left')
df = df.merge(drivers[['driver_id', 'dob']], on='driver_id', how='left')
df['date'] = pd.to_datetime(df['date'])
df['driver_age'] = ((df['date'] - pd.to_datetime(df['dob'])).dt.days / 365.25).round(1)
df = df.sort_values(['date', 'grid']).reset_index(drop=True)

# Target
df['podium'] = (df['position'] <= 3).astype(int)
df['points_finish'] = (df['points'] > 0).astype(int)

# Grid 0 means pit-lane start: treat as back of the field
field = df.groupby(['season', 'round'])['driver_id'].transform('count')
df['grid_pos'] = np.where(df['grid'] == 0, field, df['grid'])
df['field_size'] = field

# Recent form from PREVIOUS races only (shift(1) keeps the current race out)
def rolling_prev(g, col, n):
    return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

df['finish_pos'] = df['position']
gd = df.groupby('driver_id')
df['driver_avg_finish_last5'] = rolling_prev(gd, 'finish_pos', 5)
df['driver_podium_rate_last10'] = rolling_prev(gd, 'podium', 10)
df['driver_career_races'] = gd.cumcount()
gc = df.groupby('constructor_id')
df['team_avg_finish_last10'] = rolling_prev(gc, 'finish_pos', 10)
df['team_podium_rate_last10'] = rolling_prev(gc, 'podium', 10)

# Previous season's championship standings (known before the season starts)
ds = dstand[['season', 'driver_id', 'position', 'points']].copy()
ds['season'] += 1
ds = ds.rename(columns={'position': 'driver_prev_champ_pos', 'points': 'driver_prev_season_pts'})
ds = ds.drop_duplicates(['season', 'driver_id'])
df = df.merge(ds, on=['season', 'driver_id'], how='left')
cs = cstand[['season', 'constructor_id', 'position']].copy()
cs['season'] += 1
cs = cs.rename(columns={'position': 'team_prev_champ_pos'}).drop_duplicates(['season', 'constructor_id'])
df = df.merge(cs, on=['season', 'constructor_id'], how='left')

cols = ['season', 'round', 'race_name', 'circuit_id', 'driver_name', 'constructor', 'grid_pos', 'field_size',
        'driver_age', 'driver_career_races', 'driver_avg_finish_last5', 'driver_podium_rate_last10',
        'team_avg_finish_last10', 'team_podium_rate_last10', 'driver_prev_champ_pos', 'driver_prev_season_pts',
        'team_prev_champ_pos']
TARGET = sys.argv[3] if len(sys.argv) > 3 else 'podium'
cols = cols + [TARGET]
out = df[cols].copy()
# No blanks, so the template's 'drop empty values' step can't silently remove a third of the rows.
# Debut / unranked cases get explicit 'no history' values.
out['driver_prev_champ_pos'] = out['driver_prev_champ_pos'].fillna(40)   # 40 = not classified last season
out['driver_prev_season_pts'] = out['driver_prev_season_pts'].fillna(0)
out['team_prev_champ_pos'] = out['team_prev_champ_pos'].fillna(25)       # 25 = not classified / new team
out['driver_avg_finish_last5'] = out['driver_avg_finish_last5'].fillna(out['field_size'])
out['team_avg_finish_last10'] = out['team_avg_finish_last10'].fillna(out['field_size'])
out[['driver_podium_rate_last10', 'team_podium_rate_last10']] = out[['driver_podium_rate_last10', 'team_podium_rate_last10']].fillna(0)
out['driver_age'] = out['driver_age'].fillna(out['driver_age'].median())
for c in ['driver_avg_finish_last5', 'driver_podium_rate_last10', 'team_avg_finish_last10', 'team_podium_rate_last10']:
    out[c] = out[c].round(3)
out.to_csv(OUT, index=False)
print(out.shape, TARGET, 'share', round(out[TARGET].mean(), 3))
print((out.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())
