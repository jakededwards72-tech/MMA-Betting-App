from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "mma_quant.db"

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS bets (
      id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
      event TEXT, fight TEXT, market TEXT, selection TEXT, book TEXT,
      odds REAL, stake REAL, model_p REAL, model_ev REAL,
      result TEXT DEFAULT 'OPEN', profit REAL DEFAULT 0
    )""")
    return con

def add_bet(event, fight, market, selection, book, odds, stake, model_p, model_ev):
    with _conn() as con:
        con.execute("INSERT INTO bets(created_at,event,fight,market,selection,book,odds,stake,model_p,model_ev) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), event, fight, market, selection, book, float(odds), float(stake), float(model_p), float(model_ev)))

def list_bets():
    with _conn() as con:
        return pd.read_sql_query("SELECT * FROM bets ORDER BY id DESC", con)

def settle_bet(bet_id: int, result: str):
    result = result.upper()
    with _conn() as con:
        row = con.execute("SELECT odds,stake FROM bets WHERE id=?", (int(bet_id),)).fetchone()
        if not row: return
        odds, stake = row
        if result == 'WIN': profit = stake * (odds/100 if odds > 0 else 100/abs(odds))
        elif result == 'LOSS': profit = -stake
        else: profit = 0.0
        con.execute("UPDATE bets SET result=?,profit=? WHERE id=?", (result, profit, int(bet_id)))

def delete_bet(bet_id: int):
    with _conn() as con: con.execute("DELETE FROM bets WHERE id=?", (int(bet_id),))
