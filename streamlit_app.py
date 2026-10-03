from __future__ import annotations
import os, math
from datetime import datetime
import pandas as pd
import streamlit as st
from models import FighterProfile,FightContext
from simulation import simulate_fight
from betting import implied_probability
from analysis import simulation_table,evaluate_moneyline_market,expert_breakdown
from odds import fetch_mma_moneylines,flatten_moneylines,event_pairs,find_event_lines,best_line
from datahub import profile_from_all_mma,merge_profiles,DataHubError
from ufcstats import get_fighter_by_name,UFCStatsError

st.set_page_config(page_title='CageCash',page_icon='🥊',layout='wide')
st.markdown('''<style>.block-container{padding-top:1rem;max-width:1450px}.cc{border:1px solid #353843;border-radius:16px;padding:16px;margin:10px 0}.muted{opacity:.7}.edge{font-size:1.25rem;font-weight:800}.stMetric{background:#171920;padding:10px;border-radius:12px}</style>''',unsafe_allow_html=True)
st.title('🥊 CageCash')
st.caption('MMA intelligence engine • automatic fight board • independent simulations • market edges • plain-English fight reads')

def secret(name,default=''):
    try:return st.secrets.get(name,os.getenv(name,default))
    except:return os.getenv(name,default)
if 'board' not in st.session_state: st.session_state.board=[]
if 'results' not in st.session_state: st.session_state.results={}

def load_profile(name):
    notes=[]
    try:
        p,meta=profile_from_all_mma(name); notes.append(f"All-MMA record: {meta.get('record','?')} • Fight Forensics")
    except Exception as e:
        p=FighterProfile(name=name); notes.append('All-MMA record source unavailable; using conservative priors.')
    # UFCStats is enrichment, never the universe of supported fighters.
    try:
        up,_=get_fighter_by_name(name); p=merge_profiles(p,up); notes.append('UFCStats rate-stat enrichment available.')
    except Exception: pass
    return p,notes

def confidence_label(x): return 'HIGH' if x>=.72 else 'MODERATE' if x>=.48 else 'LOW'
def fair_american(p):
    p=max(.001,min(.999,p)); return round(-100*p/(1-p)) if p>=.5 else round(100*(1-p)/p)
def fmt_odds(x): return f'+{int(x)}' if x is not None and x>=0 else str(int(x)) if x is not None else '—'

def fight_read(s,a,b):
    d=s.diagnostics; strike=d.get('strike_edge_a',0); grap=d.get('grapple_edge_a',0); sub=d.get('submission_edge_a',0)
    fav=a if s.p_a_win>=s.p_b_win else b; dog=b if fav==a else a; fp=max(s.p_a_win,s.p_b_win)
    parts=[]
    if abs(strike)>.25: parts.append(f"{a if strike>0 else b} has the cleaner projected striking interaction")
    else: parts.append('the striking matchup grades relatively close')
    if abs(grap)>.25: parts.append(f"{a if grap>0 else b} owns the stronger wrestling/control pathway")
    else: parts.append('wrestling is not projected as a dominant separator')
    if abs(sub)>.35: parts.append(f"{a if sub>0 else b} carries the more meaningful submission threat")
    finish=1-s.p_distance
    shape='finish-heavy' if finish>.62 else 'decision-leaning' if s.p_distance>.60 else 'mixed finish/decision'
    return f"CageCash leans {fav} ({fp:.1%}). In matchup terms, {parts[0]}, while {parts[1]}. {parts[2]}. The overall fight shape is {shape}. {dog}'s upset path is most credible where the projected phase advantages break down; this is why the probability is a range, not a certainty."

def analyze_event(event,rows,n_sims):
    eid,when,a,b=event; lines=find_event_lines(rows,a,b)
    pa,na=load_profile(a); pb,nb=load_profile(b)
    # Auto context is intentionally neutral when the odds feed does not identify rounds/division.
    ctx=FightContext(scheduled_rounds=3,weight_class='Unknown')
    s=simulate_fight(pa,pb,ctx,n_sims=n_sims,seed=abs(hash(eid))%(2_000_000_000-1)+1)
    market=evaluate_moneyline_market(s,lines)
    ba,bb=best_line(lines,a),best_line(lines,b)
    return {'event':event,'summary':s,'lines':lines,'market':market,'notes':na+nb,'best_a':ba,'best_b':bb,'read':fight_read(s,a,b)}

with st.sidebar:
    st.header('CageCash settings')
    odds_key=st.text_input('The Odds API key',value=secret('ODDS_API_KEY'),type='password')
    regions=st.text_input('Sportsbook regions',value='us,us2')
    sims=st.select_slider('Simulations per fight',options=[10000,25000,50000,100000],value=25000)
    st.caption('25k is recommended for scanning a card. Re-run an individual fight at 100k for deeper analysis.')
    st.divider(); st.caption('Fight discovery/lines: The Odds API. All-MMA records/Elo: Fight Forensics. UFCStats is optional enrichment only for fighters it covers.')

board_tab,detail_tab,lab_tab,method_tab=st.tabs(['🔥 Fight Board','🧠 Full Fight Read','🧪 Fight Lab','ℹ️ Model'])
with board_tab:
    st.subheader('Upcoming MMA board')
    st.write('CageCash discovers the fights, loads available fighter evidence, runs its own model, then compares that independent price with sportsbooks.')
    if st.button('🔄 Scan upcoming MMA & run models',type='primary',use_container_width=True):
        try:
            with st.spinner('Pulling upcoming MMA lines…'):
                ev=fetch_mma_moneylines(odds_key,regions); rows=flatten_moneylines(ev); pairs=event_pairs(rows)
            st.session_state.board=(pairs,rows); st.session_state.results={}
            prog=st.progress(0,text=f'Analyzing {len(pairs)} fights…')
            for i,e in enumerate(pairs):
                try: st.session_state.results[e[0]]=analyze_event(e,rows,sims)
                except Exception as ex: st.session_state.results[e[0]]={'event':e,'error':str(ex)}
                prog.progress((i+1)/max(1,len(pairs)),text=f'Analyzing fight {i+1}/{len(pairs)}')
            prog.empty()
        except Exception as e: st.error(str(e))
    if not st.session_state.results: st.info('Add your Odds API key, then tap **Scan upcoming MMA & run models**. The old manual simulator is now secondary.')
    else:
        ranked=[]
        for r in st.session_state.results.values():
            if 'summary' not in r: continue
            s=r['summary']; m=r['market']; best=None
            if not m.empty:
                m2=m.sort_values('ev_roi',ascending=False); best=m2.iloc[0].to_dict()
            ranked.append((best.get('ev_roi',-9) if best else -9,r,best))
        ranked.sort(key=lambda x:x[0],reverse=True)
        for _,r,best in ranked:
            eid,when,a,b=r['event']; s=r['summary']; fav=a if s.p_a_win>=s.p_b_win else b; p=max(s.p_a_win,s.p_b_win)
            dt=pd.to_datetime(when,utc=True,errors='coerce'); when_txt=dt.strftime('%a %b %d • %I:%M %p UTC') if not pd.isna(dt) else str(when)
            st.markdown(f"### {a} vs {b}"); st.caption(when_txt)
            c1,c2,c3,c4=st.columns(4); c1.metric('Model lean',fav); c2.metric('Win probability',f'{p:.1%}'); c3.metric('Fair line',fmt_odds(fair_american(p))); c4.metric('Data confidence',confidence_label(s.data_confidence))
            if best:
                edge=(best.get('model_p',0)-best.get('no_vig_market_p',best.get('implied_p',0)))*100
                st.markdown(f"**Best current price read:** {best['selection']} {fmt_odds(best['odds'])} at {best['book']} • model EV {best['ev_roi']:.1%} • probability edge {edge:+.1f} pts • **{best['verdict']}**")
            else: st.warning('No complete two-sided sportsbook market was available for EV comparison.')
            st.write(r['read'])
            if st.button('Open full fight read',key=f'open_{eid}'):
                st.session_state.selected=eid; st.info('Fight selected. Tap the **Full Fight Read** tab above.')
            st.divider()
with detail_tab:
    eid=st.session_state.get('selected')
    if not eid or eid not in st.session_state.results: st.info('Choose **Open full fight read** from the Fight Board.')
    else:
        r=st.session_state.results[eid]; s=r['summary']; a,b=s.fighter_a,s.fighter_b
        st.header(f'{a} vs {b}')
        st.markdown(f"### CageCash fight read\n{r['read']}")
        for h,body in expert_breakdown(s).items(): st.markdown(f'**{h}:** {body}')
        st.subheader('Simulation distribution'); st.dataframe(simulation_table(s),hide_index=True,use_container_width=True)
        st.subheader('Sportsbook value board')
        if r['market'].empty: st.info('No current lines available.')
        else: st.dataframe(r['market'],hide_index=True,use_container_width=True)
        st.caption('Source/data notes: '+' • '.join(r['notes']))
with lab_tab:
    st.subheader('Custom Fight Lab')
    st.caption('This is the old manual workflow, moved out of the way. Use it for hypothetical matchups or fights missing from the live board.')
    c1,c2=st.columns(2); aa=c1.text_input('Fighter A'); bb=c2.text_input('Fighter B')
    rounds=st.selectbox('Rounds',[3,5])
    if st.button('Run custom matchup',use_container_width=True) and aa and bb:
        pa,na=load_profile(aa); pb,nb=load_profile(bb); ss=simulate_fight(pa,pb,FightContext(scheduled_rounds=rounds,weight_class='Unknown'),n_sims=sims,seed=42)
        st.session_state.lab=ss
    if 'lab' in st.session_state:
        ss=st.session_state.lab; st.metric(f'{ss.fighter_a} win',f'{ss.p_a_win:.1%}'); st.metric(f'{ss.fighter_b} win',f'{ss.p_b_win:.1%}'); st.write(fight_read(ss,ss.fighter_a,ss.fighter_b)); st.dataframe(simulation_table(ss),hide_index=True,use_container_width=True)
with method_tab:
    st.subheader('What the engine is doing')
    st.write('CageCash does not treat the sportsbook favorite as the model pick. The simulation uses available fighter evidence, shrinks sparse samples toward MMA priors, models striking/grappling/finish interactions, and simulates round and finish paths. The sportsbook line is compared afterward for price/value analysis.')
    st.write('Coverage is intentionally source-aware: the fight board is MMA-wide wherever the odds feed lists a bout. Fight Forensics supplies cross-promotion records and career Elo. UFCStats can enrich UFC-experienced fighters with granular rate statistics, but it no longer determines which fights CageCash supports.')
    st.warning('Sparse-data regional fights can appear on the board with LOW confidence. CageCash should not manufacture precision when fighter evidence is weak; those fights require a larger edge or a pass.')
    st.markdown('Data attribution: [Fight Forensics](https://fightforensics.com) • market data via The Odds API')
