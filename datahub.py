from __future__ import annotations
import os, re, requests
from datetime import datetime, timezone
from typing import Any
from models import FighterProfile

FF_BASE='https://fightforensics.com/api/v1'

class DataHubError(RuntimeError): pass

def _headers():
    key=os.getenv('FIGHT_FORENSICS_API_KEY','').strip()
    return {'X-Api-Key':key} if key else {}

def _get(path, params=None, timeout=15):
    r=requests.get(f'{FF_BASE}{path}',params=params or {},headers=_headers(),timeout=timeout)
    if r.status_code!=200: raise DataHubError(f'Fight Forensics HTTP {r.status_code}: {r.text[:180]}')
    return r.json()

def _items(obj, keys):
    if isinstance(obj,list): return obj
    if isinstance(obj,dict):
        for k in keys:
            if isinstance(obj.get(k),list): return obj[k]
    return []

def search_fighter(name:str):
    data=_get('/fighters',{'q':name,'limit':5})
    rows=_items(data,['fighters','results','data'])
    if not rows: raise DataHubError(f"No all-MMA fighter record found for '{name}'.")
    target=_norm(name)
    rows.sort(key=lambda x: (0 if _norm(str(x.get('name','')))==target else 1, abs(len(_norm(str(x.get('name',''))))-len(target))))
    return rows[0]

def fighter_detail(name:str):
    hit=search_fighter(name)
    fid=hit.get('id') or hit.get('fighter_id')
    if fid is None: return hit
    return _get(f'/fighters/{fid}')

def _norm(s): return re.sub(r'[^a-z0-9 ]','',str(s).lower()).strip()

def _walk(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            yield str(k).lower(),v
            yield from _walk(v)
    elif isinstance(obj,list):
        for v in obj: yield from _walk(v)

def _first(obj, names, default=None):
    names={n.lower() for n in names}
    for k,v in _walk(obj):
        if k in names and not isinstance(v,(dict,list)) and v not in (None,''):
            return v
    return default

def _num(v, default=None):
    if v is None: return default
    try:
        if isinstance(v,str): v=v.replace('%','').strip()
        return float(v)
    except: return default

def _pct(v):
    x=_num(v)
    if x is None:return None
    return x/100 if x>1 else x

def _record_from(obj):
    # Supports either explicit W/L/D fields or record strings such as 18-4-0.
    w=_num(_first(obj,['wins','win']))
    l=_num(_first(obj,['losses','loss']))
    d=_num(_first(obj,['draws','draw']))
    if w is not None and l is not None: return int(w),int(l),int(d or 0)
    rec=_first(obj,['record','pro_record','pro_mma'])
    if isinstance(rec,dict): rec=rec.get('value')
    m=re.search(r'(\d+)\s*[-–]\s*(\d+)(?:\s*[-–]\s*(\d+))?',str(rec or ''))
    return tuple(map(int,[m.group(1),m.group(2),m.group(3) or 0])) if m else (0,0,0)

def profile_from_all_mma(name:str):
    raw=fighter_detail(name)
    w,l,d=_record_from(raw); fights=w+l+d
    elo=_num(_first(raw,['elo','career_elo','rating']))
    age=_num(_first(raw,['age']))
    height=_num(_first(raw,['height_in','height_inches','height']))
    reach=_num(_first(raw,['reach_in','reach_inches','reach']))
    # Only use rate stats if provider actually supplies them; otherwise simulation shrinkage handles missingness.
    p=FighterProfile(name=str(_first(raw,['name','fighter_name'],name)), age=age,height_in=height,reach_in=reach,
        stance=str(_first(raw,['stance'],'Unknown')), slpm=_num(_first(raw,['slpm','sig_strikes_landed_per_minute'])),
        sapm=_num(_first(raw,['sapm','sig_strikes_absorbed_per_minute'])), str_acc=_pct(_first(raw,['str_acc','striking_accuracy'])),
        str_def=_pct(_first(raw,['str_def','striking_defense'])), td_avg=_num(_first(raw,['td_avg','takedown_average'])),
        td_acc=_pct(_first(raw,['td_acc','takedown_accuracy'])), td_def=_pct(_first(raw,['td_def','takedown_defense'])),
        sub_avg=_num(_first(raw,['sub_avg','submission_average'])), pro_fights=fights or None, elo=elo)
    # Record-derived finish shares when available.
    kos=_num(_first(raw,['ko_wins','wins_by_ko','knockout_wins']))
    subs=_num(_first(raw,['submission_wins','wins_by_submission','sub_wins']))
    if w>0:
        if kos is not None:p.ko_win_rate=min(1,kos/w)
        if subs is not None:p.sub_win_rate=min(1,subs/w)
        if kos is not None or subs is not None:p.decision_win_rate=max(0,1-(p.ko_win_rate or 0)-(p.sub_win_rate or 0))
    return p, {'source':'Fight Forensics','record':f'{w}-{l}-{d}','raw':raw}

def merge_profiles(primary:FighterProfile, enrich:FighterProfile):
    # Enrichment wins for granular rate stats; all-MMA record/Elo remains where enrichment lacks it.
    out=FighterProfile(**primary.as_dict())
    for field in out.__dataclass_fields__:
        if field=='name': continue
        v=getattr(enrich,field,None)
        if v is not None and (not isinstance(v,float) or v!=0.0): setattr(out,field,v)
    out.name=primary.name or enrich.name
    return out
