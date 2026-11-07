#!/usr/bin/env python3
"""Download public IFI inventory and daily historical weather for selected locations."""
import argparse, json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IFI_URL = "https://zenodo.org/records/16994648/files/India_Flood_Inventory_v3.csv?download=1"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--locations', default=str(ROOT/'locations_template.csv'))
    p.add_argument('--start', default='2017-01-01')
    p.add_argument('--end', default='2023-12-31')
    args=p.parse_args()
    DATA.mkdir(exist_ok=True)
    loc=pd.read_csv(args.locations)
    required={'village_id','name','district','state','latitude','longitude'}
    missing=required-set(loc.columns)
    if missing: raise SystemExit(f'Locations file missing columns: {sorted(missing)}')
    loc=loc.dropna(subset=['village_id','latitude','longitude']).drop_duplicates('village_id')
    if loc.empty: raise SystemExit('No locations with coordinates found.')

    # Public source download. Failure is surfaced; no fabricated replacement data.
    out=DATA/'India_Flood_Inventory_v3.csv'
    try:
        with requests.get(IFI_URL, timeout=90, stream=True, headers={'User-Agent':'FloodGuardAI research prototype contact: local user'}) as r:
            r.raise_for_status()
            tmp=out.with_suffix('.part')
            with open(tmp,'wb') as f:
                for chunk in r.iter_content(1024*128):
                    if chunk: f.write(chunk)
            tmp.replace(out)
        # Parse once to validate file is actually CSV.
        inv=pd.read_csv(out, low_memory=False)
        print(f'Downloaded IFI: {len(inv):,} rows; columns: {list(inv.columns)}')
        (DATA/'ifi_columns.json').write_text(json.dumps({'rows':len(inv),'columns':list(inv.columns),'source':IFI_URL},indent=2),encoding='utf-8')
    except Exception as e:
        print(f'WARNING: IFI download failed: {e}\nYou can retry later. No replacement data was generated.')

    frames=[]
    for i,row in loc.reset_index(drop=True).iterrows():
        vid=str(row.village_id)
        print(f'[{i+1}/{len(loc)}] Downloading historical weather for {row["name"]} ({vid})...')
        params={'latitude':float(row.latitude),'longitude':float(row.longitude),'start_date':args.start,'end_date':args.end,'daily':'precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_hours','timezone':'Asia/Kolkata'}
        try:
            resp=requests.get(ARCHIVE_URL,params=params,timeout=90)
            resp.raise_for_status(); d=resp.json()
            if 'daily' not in d: raise ValueError(d.get('reason','No daily data returned'))
            df=pd.DataFrame(d['daily'])
            df['date']=pd.to_datetime(df['time']).dt.date.astype(str)
            df.drop(columns=['time'],inplace=True)
            df['village_id']=vid; df['name']=row['name']; df['district']=row['district']; df['state']=row['state']
            df['latitude']=float(row.latitude); df['longitude']=float(row.longitude)
            elev=row.get('elevation_m',None)
            try:
                if pd.isna(elev):
                    er=requests.get(ELEVATION_URL,params={'latitude':float(row.latitude),'longitude':float(row.longitude)},timeout=30); er.raise_for_status(); elev=er.json().get('elevation',[None])[0]
            except Exception as ee:
                print(f'  Elevation lookup unavailable: {ee}')
            df['elevation_m']=float(elev) if elev is not None and pd.notna(elev) else float('nan')
            frames.append(df)
            print(f'  {len(df)} daily rows')
        except Exception as e:
            print(f'  ERROR: {e}')
        time.sleep(0.25)
    if not frames: raise SystemExit('No weather data downloaded. Check internet/API and retry.')
    weather=pd.concat(frames,ignore_index=True)
    weather.to_csv(DATA/'village_daily_weather.csv',index=False)
    print(f'Wrote {DATA/"village_daily_weather.csv"} ({len(weather):,} rows).')
    print('NEXT: supply verified village-level labels in village_flood_labels.csv. District labels are not silently converted to village labels.')
if __name__=='__main__': main()
