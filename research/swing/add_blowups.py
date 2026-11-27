"""Adversarial check: add stocks that are known to have soared and then crashed (and still trade),
which the index-based pool leaves out. This deliberately tilts the data AGAINST momentum ideas."""
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from engine import STOCK_DIR
from download import fetch, FIELDS

NAMES = """BHC TLRY CGC ACB SPCE QS FUBO SKLZ CLOV TDOC BYND GPRO GRPN LC DDD SSYS LKNCY DIDIY GOTU RLX TIGR YELP
ANGI BLDP SOHU JOYY WKHS HYLN LAZR MVIS SNDL KODK SFIX LMND ROOT BARK DNA VXRT INO OCGN PACB FSLY BOX DBX VRNS RPD
TENB UPWK FVRR BITF CAN RIG NBR TDW BTU IAG MBI FNMA FMCC FRCB AMBC GNW IRBT WOLF BLNK CHPT NKLA GOEV XOS PSNY
FFIE MULN AMRN ACAD OPK MNKD ICPT AGIO BLUE NVTA SDC RIDE FSR ARVL CLVS PBYI SPWR SEDG NOVA MAXN STEM FCEL
BIGC APPS MGNI PUBM IS ZI BMBL MTCH IAC TRIP EXPE CAR HTZ NCLH SABR AAL JBLU SAVE ULCC ALGT""".split()

pool = pd.read_csv(STOCK_DIR / "pool.csv", index_col=0)
new = [n for n in dict.fromkeys(NAMES) if n not in pool.index]
with ThreadPoolExecutor(max_workers=6) as ex:
    got = {k: v for k, v in zip(new, ex.map(fetch, new)) if v is not None and len(v) >= 250}
print(f"added {len(got)} of {len(new)} new names; not available: {sorted(set(new) - set(got))}")
for f in FIELDS:
    wide = pd.read_parquet(STOCK_DIR / f"{f}.parquet")
    extra = pd.concat({k: v[f] for k, v in got.items()}, axis=1).astype("float32")
    wide = wide.drop(columns=[c for c in extra.columns if c in wide.columns]).join(extra, how="left")
    wide.to_parquet(STOCK_DIR / f"{f}.parquet")
add = pd.DataFrame({"name": "", "sector": "", "source": "blowup", "yahoo": list(got)}, index=list(got))
pd.concat([pool[~pool.index.isin(add.index)], add]).to_csv(STOCK_DIR / "pool.csv")
