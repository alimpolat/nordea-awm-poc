"""Mark the Bergström portfolio to market with real prices and real FX.

The client *identity* is fictitious by design (name, quantities held, cost basis,
IPS, meeting narrative). Everything market-derived becomes REAL:

  current_mv     = quantity x live price x live FX->SEK   (Yahoo, free)
  ytd_return_pct = real YTD price return from Jan-1 close
  aum_sek        = sum of marked MVs
  as_of          = today

The five holdings with no market feed (two sovereign bonds, Emirates REIT/Dubai,
two illiquid private funds) keep their last-statement valuation — exactly how a
real wealth platform reports: liquid marked daily, privates lag the quarterly
statement.

Usage:  python scripts/mark_to_market.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
PORTFOLIO = REPO / "data" / "bergstrom_portfolio.json"

# portfolio code -> tradable Yahoo symbol
TICKER_OVERRIDES = {"IEAC": "IEAC.AS"}
# no daily market price: bonds (yield-quoted), Dubai REIT (not on Yahoo), private funds
STATEMENT_VALUED = {"DE10Y-BUND", "FR10Y-OAT", "REIT.DI", "NORD-PE-VII", "GLOB-INFRA"}
# listing currency -> Yahoo FX cross to SEK ("" = already SEK)
FX_SYMBOL = {"SEK": "", "USD": "SEK=X", "EUR": "EURSEK=X", "AED": "AEDSEK=X", "SAR": "SARSEK=X"}

UA = {"User-Agent": "Mozilla/5.0"}


def yahoo_ytd(symbol: str) -> tuple[float, float]:
    """(last_price, ytd_return_pct) from real Jan-1-to-now daily closes."""
    r = httpx.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "1d", "range": "ytd"}, headers=UA, timeout=8,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = [c for c in res["indicators"]["quote"][0]["close"] if c is not None]
    if len(closes) < 2:
        raise ValueError(f"{symbol}: not enough closes")
    return float(closes[-1]), (closes[-1] / closes[0] - 1.0) * 100.0


def fx_to_sek(ccy: str, cache: dict) -> float:
    if ccy == "SEK":
        return 1.0
    if ccy not in cache:
        r = httpx.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{FX_SYMBOL[ccy]}",
            params={"interval": "1d", "range": "5d"}, headers=UA, timeout=8,
        )
        r.raise_for_status()
        cache[ccy] = float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    return cache[ccy]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--rebase-quantities", action="store_true",
        help="One-time: re-derive share counts from the DESIGNED MVs at REAL prices "
        "(the fictitious quantities were back-computed from made-up prices). "
        "After rebasing, plain runs just mark the fixed quantities to market.",
    )
    args = ap.parse_args()

    p = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    fx_cache: dict[str, float] = {}
    marked, kept = 0, 0

    print(f"{'TICKER':14} {'PRICE':>10} {'FX':>8} {'QTY':>9} {'MV (SEK M)':>11} {'old MV':>8} {'YTD %':>7}")
    for h in p["holdings"]:
        t = h["ticker"]
        if t in STATEMENT_VALUED:
            kept += 1
            print(f"{t:14} {'—':>10} {'—':>8} {'—':>9} {h['current_mv']/1e6:>11.1f} {h['current_mv']/1e6:>8.1f}   (statement-valued)")
            continue
        sym = TICKER_OVERRIDES.get(t, t)
        price, ytd = yahoo_ytd(sym)
        fx = fx_to_sek(h["fx_exposure"], fx_cache)
        if args.rebase_quantities:
            h["quantity"] = max(100, round(h["current_mv"] / (price * fx) / 100) * 100)
        mv = round(h["quantity"] * price * fx)
        print(f"{t:14} {price:>10.2f} {fx:>8.4f} {h['quantity']:>9,} {mv/1e6:>11.1f} {h['current_mv']/1e6:>8.1f} {ytd:>+7.1f}")
        h["current_mv"] = mv
        h["ytd_return_pct"] = round(ytd, 1)
        marked += 1

    p["aum_sek"] = sum(h["current_mv"] for h in p["holdings"])
    p["as_of"] = date.today().isoformat()
    p["valuation_note"] = (
        f"Liquid positions marked to market {date.today().isoformat()} "
        "(live prices x live FX); bonds/illiquid funds at last statement valuation."
    )

    print(f"\nAUM: SEK {p['aum_sek']/1e6:,.1f}M  ({marked} marked live, {kept} statement-valued)")
    print("FX used:", {k: round(v, 4) for k, v in fx_cache.items()})

    if args.dry_run:
        print("\n--dry-run: not written")
        return 0
    PORTFOLIO.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Written -> {PORTFOLIO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
