from __future__ import annotations
import requests
ODDS_API_BASE='https://api.the-odds-api.com/v4'; SPORT_KEY='mma_mixed_martial_arts'
class OddsAPIError(RuntimeError): pass

def fetch_mma_moneylines(api_key:str,regions='us,us2',timeout=20):
    if not api_key: raise OddsAPIError('Add your The Odds API key in Settings first.')
    r=requests.get(f'{ODDS_API_BASE}/sports/{SPORT_KEY}/odds',params={'apiKey':api_key,'regions':regions,'markets':'h2h','oddsFormat':'american','dateFormat':'iso'},timeout=timeout)
    if r.status_code!=200: raise OddsAPIError(f'Odds API HTTP {r.status_code}: {r.text[:240]}')
    return r.json()

def flatten_moneylines(events):
    rows=[]
    for e in events:
      for b in e.get('bookmakers',[]):
       for m in b.get('markets',[]):
        if m.get('key')!='h2h':continue
        for o in m.get('outcomes',[]): rows.append({'event_id':e.get('id'),'commence_time':e.get('commence_time'),'fighter_a':e.get('home_team'),'fighter_b':e.get('away_team'),'selection':o.get('name'),'odds':o.get('price'),'book':b.get('title',b.get('key','Unknown')),'book_key':b.get('key'),'last_update':b.get('last_update') or m.get('last_update')})
    return rows

def event_pairs(rows):
    seen=set(); out=[]
    for r in rows:
      k=(r.get('event_id'),r.get('commence_time'),r.get('fighter_a'),r.get('fighter_b'))
      if k not in seen: seen.add(k); out.append(k)
    return out

def find_event_lines(rows,a,b):
    n=lambda s:' '.join(str(s or '').lower().replace('-',' ').replace("'",'').split())
    return [r for r in rows if {n(r.get('fighter_a')),n(r.get('fighter_b'))}=={n(a),n(b)}]

def best_line(lines,name):
    n=lambda s:' '.join(str(s or '').lower().replace('-',' ').replace("'",'').split())
    xs=[r for r in lines if n(r.get('selection'))==n(name) and r.get('odds') is not None]
    return max(xs,key=lambda r:float(r['odds'])) if xs else None
