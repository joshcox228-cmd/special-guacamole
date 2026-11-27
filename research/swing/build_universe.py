"""Build the pool of stocks to download, from Wikipedia's index membership tables.

Usage: python -I build_universe.py <dir with saved html pages> <output dir>

The pool is every company in the S&P 500 / 400 / 600 today (about 1,500 US companies),
plus a hand-written list of heavily traded names that sit outside those indexes.

Every name in the pool still trades today. Companies that were delisted (bought out or
bankrupt) are missing, which flatters any backtest; the study measures and stress-tests that.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

# Heavily traded stocks that are not (or were not) in the US indexes above: foreign
# companies listed in the US, recent listings and retail favourites.
EXTRAS = """
BABA TSM ASML NVO SHOP SE MELI PDD JD BIDU NIO XPEV LI SAP TM SONY AZN NVS GSK SNY BP SHEL TTE RIO BHP VALE
PBR ITUB INFY UL DEO BTI NTES BILI FUTU GRAB CPNG SPOT ARM STLA RACE CCJ CNQ SU ENB TD RY BNS BMO CP CNI AEM
WPM FNV TECK NTR MSTR COIN HOOD PLTR SOFI AFRM UPST RBLX U SNOW NET DDOG ZS MDB OKTA TWLO ZM DOCU PINS SNAP
LYFT CVNA W CHWY ETSY PTON GME AMC PLUG FCEL RIVN LCID DKNG ROKU TTD APP CRWD PANW RDDT CRWV IREN MARA RIOT
CLSK HUT SMCI VRT CEG NRG VST OKLO SMR NNE LEU UEC ASTS RKLB LUNR JOBY ACHR IONQ RGTI QBTS SOUN AI PATH BBAI
SYM TEM HIMS CELH ELF ONON DUOL CAVA TOST BROS SG WING NESR ESI LITE CEVA RMBS ALLY BB NOK ERIC TEVA GOLD B
KGC AG PAAS HL CDE BTG NEM FCX CLF X AA MT NUE STLD VKTX MRNA BNTX NVAX SAVA CRSP NTLA BEAM EDIT ARKK
ARES CG OWL TPG LPLA IBKR MRVL TEAM WDAY VEEV ALNY BMRN EXAS ILMN RPRX LNG CQP ET EPD MPLX WES PAA FLUT FERG
CRH XYZ EQH FNF RKT Z OPEN CART DASH ABNB BIRK GTLB CFLT ESTC HUBS BILL FOUR NU XP STNE MNDY WIX CYBR CHKP
NICE GLBE ICLR ARGX LEGN GMAB TCOM BEKE YUMC ZTO VIPS TAL EDU IQ WB KWEB FXI EWZ GDX GDXJ SLV GLD USO UNG
SMH SOXX XBI IBB XOP XME XHB XRT KRE KBE ITB IYR HYG LQD EEM EFA DIA VTI VOO
""".split()

BENCHMARKS = ["SPY", "QQQ", "IWM", "MDY", "SHY", "TLT", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI",
              "XLB", "XLU", "XLRE", "XLC", "^VIX", "^VIX3M"]


def tables(path: Path) -> list[pd.DataFrame]:
    return pd.read_html(StringIO(path.read_text(encoding="utf-8", errors="replace")))


def flat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(p) for p in c if "Unnamed" not in str(p)).strip() if isinstance(c, tuple) else str(c)
                  for c in df.columns]
    return df


def members(path: Path, source: str) -> pd.DataFrame:
    """The first table on the page that has a ticker column and a company column."""
    for raw in tables(path):
        df = flat(raw)
        tick = next((c for c in df.columns if c.lower() in ("symbol", "ticker", "ticker symbol")), None)
        name = next((c for c in df.columns if c.lower() in ("security", "company", "company name")), None)
        if tick and name and len(df) > 50:
            sector = next((c for c in df.columns if "sector" in c.lower() or "industry" in c.lower()), None)
            out = pd.DataFrame({"ticker": df[tick].astype(str).str.strip(), "name": df[name].astype(str).str.strip(),
                                "sector": df[sector].astype(str).str.strip() if sector else "", "source": source})
            return out
    raise ValueError(f"no membership table found in {path.name}")


def main(src: Path, out: Path) -> None:
    frames = [members(src / f"{k}.html", k) for k in ("sp500", "sp400", "sp600")]
    for f in frames:
        print(f"{f['source'].iloc[0]:>6}: {len(f)} rows")
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["ticker"].str.fullmatch(r"[A-Z]{1,5}([.\-][A-Z])?")]
    extra = pd.DataFrame({"ticker": EXTRAS, "name": "", "sector": "", "source": "extra"})
    bench = pd.DataFrame({"ticker": BENCHMARKS, "name": "", "sector": "ETF/Index", "source": "benchmark"})
    pool = pd.concat([pool, extra, bench], ignore_index=True)
    # keep the first (most senior index) row per ticker, but remember every index it is in
    sources = pool.groupby("ticker")["source"].agg(lambda s: "+".join(dict.fromkeys(s)))
    pool = pool.drop_duplicates("ticker").set_index("ticker")
    pool["source"] = sources
    pool["yahoo"] = pool.index.str.replace(".", "-", regex=False)
    out.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out / "pool.csv")
    print(f"pool: {len(pool)} tickers")
    print(pool["source"].str.split("+").str[0].value_counts().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
