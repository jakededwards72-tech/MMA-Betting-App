from __future__ import annotations
import os, hashlib
import pandas as pd
import streamlit as st
from models import FighterProfile, FightContext
from simulation import simulate_fight
from analysis import simulation_table, evaluate_moneyline_market, expert_breakdown
from odds import fetch_mma_moneylines, flatten_moneylines, event_pairs, find_event_lines, best_line
from datahub import profile_from_all_mma

st.set_page_config(page_title="CageCash", page_icon="🥊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--cc-bg:#090b0f;--cc-card:#11141a;--cc-line:#272c35;--cc-muted:#9198a5;--cc-edge:#73d9aa;--cc-loss:#ff7b7b;--cc-gold:#d8bd82}
.stApp{background:var(--cc-bg)} .block-container{padding-top:1.2rem;padding-bottom:5rem;max-width:1180px}
[data-testid="stHeader"]{background:transparent}.cc-brand{font-size:2.65rem;font-weight:850;letter-spacing:-.04em;margin:0}.cc-kicker{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--cc-muted)}
.cc-hero,.cc-card{background:var(--cc-card);border:1px solid var(--cc-line);border-radius:22px;padding:20px;margin:12px 0}.cc-hero{padding:24px}
.cc-fight{font-size:1.45rem;font-weight:760;letter-spacing:-.02em}.cc-meta{color:var(--cc-muted);font-size:.82rem}.cc-pill{display:inline-block;border:1px solid #3a3f49;border-radius:999px;padding:4px 9px;font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;margin-right:6px}.cc-edge{color:var(--cc-edge)}.cc-loss{color:var(--cc-loss)}
[data-testid="stMetric"]{background:#0c0f14;border:1px solid var(--cc-line);padding:12px;border-radius:14px}[data-testid="stMetricLabel"]{color:var(--cc-muted)}
.stButton>button{border-radius:12px;min-height:44px}.stTabs [data-baseweb="tab-list"]{gap:18px}.stTabs [data-baseweb="tab"]{padding-left:0;padding-right:0}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}.cc-brand{font-size:2.25rem}.cc-hero{padding:18px}.cc-fight{font-size:1.2rem}}
</style>
""", unsafe_allow_html=True)


def secret(name, default=""):
    try: return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception: return os.getenv(name, default)

def norm(s): return " ".join(str(s or "").lower().replace("-", " ").replace("'", "").split())
def fmt_odds(x):
    if x is None or pd.isna(x): return "—"
    x=int(round(float(x))); return f"+{x}" if x>=0 else str(x)
def fair_american(p):
    p=max(.001,min(.999,float(p))); return round(-100*p/(1-p)) if p>=.5 else round(100*(1-p)/p)
def conf_label(x): return "HIGH" if x>=.72 else "MODERATE" if x>=.48 else "LOW"
def safe(v, default=0.0):
    try: return float(v)
    except Exception: return default

def deterministic_seed(event_id, a, b):
    h=hashlib.sha256(f"{event_id}|{a}|{b}".encode()).hexdigest()[:8]
    return int(h,16) % 2_000_000_000 + 1

@st.cache_data(ttl=21600, show_spinner=False)
def load_profile(name):
    try:
        p,meta=profile_from_all_mma(name)
        return p,[f"All-MMA record {meta.get('record','?')} · Fight Forensics"],True
    except Exception as e:
        return FighterProfile(name=name),[f"Cross-promotion profile unavailable ({type(e).__name__}); conservative priors used."],False

@st.cache_data(ttl=180, show_spinner=False)
def load_odds(api_key, regions):
    return flatten_moneylines(fetch_mma_moneylines(api_key,regions))

def fight_read(s,a,b):
    d=s.diagnostics or {}; strike=safe(d.get('strike_edge_a')); grap=safe(d.get('grapple_edge_a')); sub=safe(d.get('submission_edge_a'))
    fav, dog = (a,b) if s.p_a_win>=s.p_b_win else (b,a); fp=max(s.p_a_win,s.p_b_win)
    phase=[]
    phase.append(f"{a if strike>0 else b} projects as the cleaner striking side" if abs(strike)>.25 else "the striking phase projects close")
    phase.append(f"{a if grap>0 else b} has the stronger wrestling/control route" if abs(grap)>.25 else "wrestling is not a major separator")
    if abs(sub)>.35: phase.append(f"{a if sub>0 else b} owns the more meaningful submission threat")
    finish=1-s.p_distance
    shape="finish-heavy" if finish>.62 else "decision-heavy" if s.p_distance>.60 else "mixed between finishes and decisions"
    method_a=max([(s.p_a_ko,'KO/TKO'),(s.p_a_sub,'submission'),(s.p_a_dec,'decision')])[1]
    method_b=max([(s.p_b_ko,'KO/TKO'),(s.p_b_sub,'submission'),(s.p_b_dec,'decision')])[1]
    text=f"CageCash leans **{fav} at {fp:.1%}**. Stylistically, {phase[0]}; {phase[1]}."
    if len(phase)>2: text+=f" {phase[2]}."
    text+=f" The simulation is **{shape}**. {a}'s most common win route is {method_a}; {b}'s is {method_b}."
    if s.data_confidence<.48: text+=f" The biggest warning is data quality: this is a **LOW-confidence** read, so {dog}'s upset paths carry more uncertainty than the headline probability suggests."
    elif abs(s.p_a_win-s.p_b_win)<.10: text+=" This is a price-sensitive matchup; a small change in assumptions can move the betting conclusion."
    return text

def analyze_event(event, rows, n_sims):
    eid,when,a,b=event; lines=find_event_lines(rows,a,b)
    pa,na,found_a=load_profile(a); pb,nb,found_b=load_profile(b)
    s=simulate_fight(pa,pb,FightContext(scheduled_rounds=3,weight_class='Unknown'),n_sims=n_sims,seed=deterministic_seed(eid,a,b))
    market=evaluate_moneyline_market(s,lines)
    return {'event':event,'summary':s,'lines':lines,'market':market,'notes':na+nb,'profiles_found':found_a+found_b,'read':fight_read(s,a,b)}

def top_market(r):
    m=r.get('market')
    if m is None or m.empty:return None
    return m.sort_values('ev_roi',ascending=False).iloc[0].to_dict()

def render_fight(r, compact=False):
    eid,when,a,b=r['event']; s=r['summary']; best=top_market(r)
    dt=pd.to_datetime(when,utc=True,errors='coerce'); when_txt=dt.strftime('%a %b %d · %I:%M %p UTC') if not pd.isna(dt) else str(when)
    fav=a if s.p_a_win>=s.p_b_win else b; p=max(s.p_a_win,s.p_b_win)
    st.markdown(f'<div class="cc-card"><div class="cc-meta">{when_txt}</div><div class="cc-fight">{a} <span style="color:#6f7682">vs</span> {b}</div><div style="margin-top:8px"><span class="cc-pill">{conf_label(s.data_confidence)} DATA</span><span class="cc-pill">{s.n_sims:,} SIMS</span></div></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3); c1.metric('Model lean',fav); c2.metric('Win probability',f'{p:.1%}'); c3.metric('Fair price',fmt_odds(fair_american(p)))
    if best:
        market_p=safe(best.get('no_vig_market_p'),safe(best.get('implied_p'))); edge=(safe(best.get('model_p'))-market_p)*100
        ev=safe(best.get('ev_roi')); klass='cc-edge' if ev>0 else 'cc-loss'
        st.markdown(f'<div class="{klass}"><b>{best.get("verdict","PRICE CHECK")}</b> · {best.get("selection")} {fmt_odds(best.get("odds"))} @ {best.get("book")} · EV {ev:+.1%} · edge {edge:+.1f} pts</div>',unsafe_allow_html=True)
    else: st.caption('No complete two-sided sportsbook price was available for EV comparison.')
    st.markdown(r['read'])
    if not compact and st.button('Open full fight read →',key=f'open_{eid}',use_container_width=True): st.session_state.selected=eid

if 'results' not in st.session_state: st.session_state.results={}
if 'scan_meta' not in st.session_state: st.session_state.scan_meta={}

st.markdown('<div class="cc-kicker">MMA INTELLIGENCE DESK</div><div class="cc-brand">🥊 CageCash</div>',unsafe_allow_html=True)
st.caption('Independent simulations · live market comparison · fight reads · full-MMA board')

with st.sidebar:
    st.header('Desk settings')
    odds_key=st.text_input('The Odds API key',value=secret('ODDS_API_KEY'),type='password')
    regions=st.text_input('Sportsbook regions',value='us,us2')
    sims=st.select_slider('Board simulations / fight',options=[5000,10000,25000,50000],value=10000)
    st.caption('10k is the fast board setting. Use Fight Lab for deeper reruns.')
    st.divider(); st.caption('Odds: The Odds API · cross-promotion fighter evidence: Fight Forensics · model: CageCash Monte Carlo engine')

board_tab,scanner_tab,detail_tab,lab_tab,method_tab=st.tabs(['DESK','SCANNER','FIGHT READ','LAB','MODEL'])

with board_tab:
    st.markdown('<div class="cc-hero"><div class="cc-kicker">LIVE DESK</div><div style="font-size:2rem;font-weight:800;margin-top:6px">Upcoming MMA board</div><div class="cc-meta" style="margin-top:8px">Pull the market once, model each fight independently, then compare CageCash fair prices against the books.</div></div>',unsafe_allow_html=True)
    if st.button('↻  Refresh board & run models',type='primary',use_container_width=True):
        if not odds_key: st.error('The Odds API key is missing from Streamlit Secrets.')
        else:
            st.session_state.results={}; status=st.status('Building CageCash desk…',expanded=True)
            try:
                status.write('Loading upcoming MMA markets…'); rows=load_odds(odds_key,regions); pairs=event_pairs(rows)
                st.session_state.scan_meta={'fights':len(pairs),'rows':len(rows),'time':pd.Timestamp.utcnow().isoformat()}
                status.write(f'Found {len(pairs)} fights. Running {sims:,} paths per fight…')
                prog=st.progress(0); live=st.empty()
                for i,e in enumerate(pairs):
                    try: st.session_state.results[e[0]]=analyze_event(e,rows,sims)
                    except Exception as ex: st.session_state.results[e[0]]={'event':e,'error':f'{type(ex).__name__}: {ex}'}
                    prog.progress((i+1)/max(1,len(pairs))); live.caption(f'Priced {i+1}/{len(pairs)} · {e[2]} vs {e[3]}')
                prog.empty(); live.empty(); status.update(label='Desk ready',state='complete',expanded=False)
            except Exception as e: status.update(label='Board refresh failed',state='error'); st.error(str(e))
    results=list(st.session_state.results.values()); ok=[r for r in results if 'summary' in r]; bad=[r for r in results if 'summary' not in r]
    if not results: st.info('Tap **Refresh board & run models**. Your API key stays in Streamlit Secrets; it is not shown on the board.')
    else:
        meta=st.session_state.scan_meta; c1,c2,c3=st.columns(3); c1.metric('Fights found',meta.get('fights',len(results))); c2.metric('Modeled',len(ok)); c3.metric('Data errors',len(bad))
        if bad:
            with st.expander(f'{len(bad)} fights could not be modeled'):
                for r in bad: st.code(f"{r['event'][2]} vs {r['event'][3]} — {r.get('error')}")
        ranked=sorted(ok,key=lambda r:safe((top_market(r) or {}).get('ev_roi'),-99),reverse=True)
        if ranked:
            st.subheader('Best seats')
            positives=[r for r in ranked if safe((top_market(r) or {}).get('ev_roi'),-1)>0]
            if positives:
                for r in positives[:3]: render_fight(r,compact=True); st.divider()
            else: st.caption('No positive-EV moneyline survived the current model/price comparison.')
            st.subheader("Tonight's board")
            for r in ranked: render_fight(r); st.divider()

with scanner_tab:
    st.header('Value scanner'); st.caption('Every modeled moneyline ranked by expected value. Positive model edge is not the same thing as a guaranteed winning bet.')
    rows_out=[]
    for r in st.session_state.results.values():
        if 'summary' not in r: continue
        s=r['summary']; m=r['market']
        if m.empty: continue
        for _,x in m.iterrows(): rows_out.append({'Fight':f'{s.fighter_a} vs {s.fighter_b}','Selection':x['selection'],'Book':x['book'],'Price':fmt_odds(x['odds']),'Model':f"{safe(x['model_p']):.1%}",'EV':safe(x['ev_roi']),'Verdict':x['verdict']})
    if not rows_out: st.info('Run the Desk first to populate the scanner.')
    else:
        df=pd.DataFrame(rows_out).sort_values('EV',ascending=False); df['EV']=df['EV'].map(lambda x:f'{x:+.1%}'); st.dataframe(df,hide_index=True,use_container_width=True)

with detail_tab:
    eid=st.session_state.get('selected'); r=st.session_state.results.get(eid) if eid else None
    if not r or 'summary' not in r: st.info('Open a fight from the Desk to load its full read.')
    else:
        s=r['summary']; st.header(f'{s.fighter_a} vs {s.fighter_b}'); st.markdown(r['read']); st.divider()
        for h,body in expert_breakdown(s).items(): st.markdown(f'**{h}**  \n{body}')
        st.subheader('Simulation distribution'); st.dataframe(simulation_table(s),hide_index=True,use_container_width=True)
        st.subheader('Sportsbook board'); st.dataframe(r['market'],hide_index=True,use_container_width=True) if not r['market'].empty else st.info('No current two-sided lines.')
        st.caption(' · '.join(r['notes']))

with lab_tab:
    st.header('Fight Lab'); st.caption('Deep-dive a hypothetical matchup or rerun a listed fight at higher simulation depth.')
    c1,c2=st.columns(2); aa=c1.text_input('Fighter A'); bb=c2.text_input('Fighter B'); rounds=st.selectbox('Scheduled rounds',[3,5]); lab_sims=st.select_slider('Deep simulations',[25000,50000,100000,250000],value=50000)
    if st.button('Run Fight Lab',use_container_width=True) and aa.strip() and bb.strip():
        with st.spinner('Building matchup…'):
            pa,_,_=load_profile(aa.strip()); pb,_,_=load_profile(bb.strip()); st.session_state.lab=simulate_fight(pa,pb,FightContext(scheduled_rounds=rounds),n_sims=lab_sims,seed=deterministic_seed('lab',aa,bb))
    if st.session_state.get('lab'):
        s=st.session_state.lab; st.markdown(fight_read(s,s.fighter_a,s.fighter_b)); st.dataframe(simulation_table(s),hide_index=True,use_container_width=True)

with method_tab:
    st.header('Model desk')
    st.markdown('**Prediction first, price second.** CageCash builds an independent matchup distribution before it looks at sportsbook price. It then removes two-way vig where possible and evaluates the offered line against the model.')
    st.markdown('**Uncertainty is part of the output.** Sparse cross-promotion data is shrunk toward priors and lowers the data-confidence label instead of being presented as false precision.')
    st.markdown('**Current limitation.** The generic MMA odds feed does not reliably identify weight class or scheduled rounds. The automatic board therefore uses a neutral 3-round context until event metadata is enriched. Five-round fights should be rerun in Fight Lab before treating the board probability as final.')
    st.caption('CageCash is an analytical tool, not a guarantee of outcomes or profit.')
