import streamlit as st
import pandas as pd
import io, unicodedata
from datetime import date, datetime

st.set_page_config(page_title="MCDA Patient Pathway Finder", layout="wide")
st.title("MCDA Patient Pathway Finder")

# ------------------------- Helperek -------------------------
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm(s: str) -> str:
    return strip_accents(str(s)).strip().lower()

def read_any_csv(file_like):
    # Próbáljuk auto-sep-et (',' vagy ';'), és UTF-8, majd latin-1
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            file_like.seek(0)
            return pd.read_csv(file_like, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    file_like.seek(0)
    return pd.read_csv(file_like)  # utolsó esély

def coerce_price(series: pd.Series) -> pd.Series:
    # számot csinál: törli a 'HUF', 'Ft', szóköz, NBSP, ezres pont, stb.; ',' -> '.'
    def to_num(x):
        if pd.isna(x):
            return float("nan")
        s = str(x)
        s = s.replace("\xa0", " ").replace("HUF", "").replace("Ft", "").replace("ft", "")
        s = s.replace(" ", "").replace(".", "")
        s = s.replace(",", ".")
        keep = "".join(ch for ch in s if ch.isdigit() or ch==".")
        try:
            return float(keep) if keep not in ["", "."] else float("nan")
        except Exception:
            return float("nan")
    return series.apply(to_num)

def parse_time_any(x):
    if pd.isna(x): return pd.NaT
    if isinstance(x, datetime): return x.time()
    if hasattr(x, "to_pydatetime"):
        dt = x.to_pydatetime()
        return dt.time()
    s = str(x).strip()
    for fmt in ["%H:%M", "%H.%M", "%H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    return pd.NaT

def resolve_column(df: pd.DataFrame, aliases) -> str | None:
    # aliases: list of possible names; returns the matched original column name
    cols_norm = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in cols_norm:
            return cols_norm[norm(a)]
    return None

# Magyar+angol aliasok
ALIASES = {
    "institution": ["institution", "intezmeny", "intézmény", "intézmény neve", "intezmeny neve"],
    "service":     ["service", "szolgaltatas", "szolgáltatás"],
    "price":       ["price", "ar", "ár", "dij", "díj", "koltseg", "költség"],
    "location":    ["location", "helyszin", "helyszín", "cim", "cím", "varos", "város"],
    "date":        ["date", "datum", "dátum"],
    "time":        ["time", "idopont", "időpont", "ora", "óra"],
    "start_dt":    ["start_dt", "kezdet", "kezdes", "kezdes_dt", "idopont_dt"],
}

def load_df(uploaded):
    # 1) Feltöltött fájl
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            return read_any_csv(uploaded)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
        else:
            st.error("Csak CSV vagy XLSX fájl tölthető fel.")
            st.stop()

    # 2) Fallback repo fájlok
    for local in ["egyseges.xlsx", "egyseges.csv", "data/egyseges.xlsx", "data/egyseges.csv"]:
        try:
            if local.endswith(".csv"):
                with open(local, "rb") as f:
                    df = read_any_csv(io.BytesIO(f.read()))
            else:
                df = pd.read_excel(local)
            st.info(f"A repóban lévő „{local}” fájlt használod.")
            return df
        except Exception:
            continue

    st.error("Nem találtam adatfájlt. Tölts fel `egyseges.csv` vagy `egyseges.xlsx` fájlt.")
    st.stop()

# ---------------------- Adat betöltés és validálás ----------------------
uploaded = st.file_uploader("Tölts fel CSV/XLSX fájlt", type=["csv", "xlsx"])
df = load_df(uploaded).copy()

# Dobáljuk az üres 'Unnamed: ...' index-oszlopokat
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

# Oszlopok feloldása
inst_col = resolve_column(df, ALIASES["institution"])
svc_col  = resolve_column(df, ALIASES["service"])
prc_col  = resolve_column(df, ALIASES["price"])
loc_col  = resolve_column(df, ALIASES["location"])
date_col = resolve_column(df, ALIASES["date"])
time_col = resolve_column(df, ALIASES["time"])
start_col= resolve_column(df, ALIASES["start_dt"])

# Debug infó
with st.expander("🔍 Diagnosztika (talált oszlopok)"):
    st.write("Eredeti oszlopok:", list(df.columns))
    st.write("Felismert oszlopok:", {
        "institution": inst_col, "service": svc_col, "price": prc_col, "location": loc_col,
        "date": date_col, "time": time_col, "start_dt": start_col
    })

required_missing = [("institution", inst_col), ("service", svc_col), ("price", prc_col), ("location", loc_col)]
missing = [name for name, col in required_missing if col is None]
if missing:
    st.error("Hiányzó kötelező oszlop(ok): " + ", ".join(missing))
    st.stop()

# Ár -> numerikus
df["__price"] = coerce_price(df[prc_col])

# start_dt előállítása, ha kell
if start_col is None:
    if date_col is not None and time_col is not None:
        d = pd.to_datetime(df[date_col], errors="coerce").dt.date
        t = df[time_col].apply(parse_time_any)
        df["start_dt"] = pd.to_datetime(d.astype(str) + " " + t.astype(str), errors="coerce")
    else:
        df["start_dt"] = pd.NaT
else:
    df["start_dt"] = pd.to_datetime(df[start_col], errors="coerce")

# Várakozási napok
df["waiting_days"] = (df["start_dt"].dt.date - date.today()).dt.days

st.subheader("Forrásadat (részlet)")
preview_cols = [c for c in [inst_col, svc_col, prc_col, loc_col, date_col, time_col, "start_dt"] if c is not None]
st.dataframe(df[preview_cols].head(50))

# ----------------------------- Szűrők -----------------------------
c1, c2, c3, c4 = st.columns([1,1,1,1])
services = sorted(df[svc_col].dropna().astype(str).unique())
selected_services = c1.multiselect("Szolgáltatások", services)
prefer_city = c2.text_input("Preferált város (opcionális)", value="Budapest")
max_wait = c3.slider("Max. várakozás (nap)", 0, 60, 14)
max_price_filter = c4.number_input("Max. ár (HUF, opcionális)", min_value=0, value=0, step=1000)

filtered = df.copy()
if selected_services:
    filtered = filtered[ filtered[svc_col].astype(str).isin(selected_services) ]
if max_wait:
    filtered = filtered[ filtered["waiting_days"].fillna(9999) <= max_wait ]
if max_price_filter and max_price_filter > 0:
    filtered = filtered[ filtered["__price"].fillna(1e12) <= max_price_filter ]

if filtered.empty:
    st.warning("Nincs találat a beállított szűrőkre.")
if not df.empty:
    st.markdown("## Válaszd ki, milyen szolgáltatást szeretnél és add meg a preferenciáid!")
    c1, c2 = st.columns([2,2])
    with c1:
        services = sorted(df[svc_col].dropna().astype(str).unique())
        selected_services = st.multiselect("Milyen szolgáltatást keresel?", services)
        st.caption("Többet is kiválaszthatsz. Pl. Anyajegyszűrés, Ultrahang, stb.")
    with c2:
        st.markdown("**Preferenciák:**")
        st.caption("Állítsd be, hogy mi a legfontosabb számodra az optimális betegút kiválasztásához!")
        price_w = st.slider("Legyen minél olcsóbb? (Ár fontossága %)", 0, 100, 40)
        loc_w = st.slider("Legyen minél közelebb? (Helyszín fontossága %)", 0, 100-price_w, 30)
        time_w = 100 - price_w - loc_w
        st.caption(f"Legyen minél korábbi időpont? (Időpont fontossága: **{time_w}%**)\nA három preferencia összesen 100%-ot ad ki.")
        prefer_city = st.text_input("Preferált város (opcionális)", value="Budapest")
        max_wait = st.slider("Max. várakozás (nap)", 0, 60, 14)
        max_price_filter = st.number_input("Max. ár (HUF, opcionális)", min_value=0, value=0, step=1000)

    filtered = df.copy()
    if selected_services:
        filtered = filtered[ filtered[svc_col].astype(str).isin(selected_services) ]
    if max_wait:
        filtered = filtered[ filtered["waiting_days"].fillna(9999) <= max_wait ]
    if max_price_filter and max_price_filter > 0:
        filtered = filtered[ filtered["__price"].fillna(1e12) <= max_price_filter ]

    if filtered.empty:
        st.warning("Nincs találat a beállított szűrőkre.")
    st.stop()

# ----------------------------- MCDA -----------------------------
st.markdown("### Súlyok (összesen 100%)")
w1, w2 = st.columns(2)
price_w = w1.slider("Ár fontossága (%)", 0, 100, 40)
loc_w = w2.slider("Helyszín fontossága (%)", 0, 100 - price_w, 30)
time_w = 100 - price_w - loc_w
st.caption(f"Elérhető időpont fontossága: **{time_w}%**")

# Ár hasznosság: olcsóbb = jobb
pmin, pmax = filtered["__price"].min(), filtered["__price"].max()
if pd.isna(pmin) or pd.isna(pmax) or pmax == pmin:
if not df.empty:
    st.subheader("Forrásadat (részlet)")
    preview_cols = [c for c in [inst_col, svc_col, prc_col, loc_col, date_col, time_col, "start_dt"] if c is not None]
    st.dataframe(df[preview_cols].head(50))
else:
    st.info("Nincs betöltött adat. Tölts fel egy CSV-t vagy XLSX-et, vagy ellenőrizd a repóban lévő fájlt!")
    filtered["price_util"] = 1 - price_norm

# Helyszín hasznosság: prefer_city találat = 1, különben 0
if prefer_city:
    filtered["location_util"] = filtered[loc_col].astype(str).str.contains(prefer_city, case=False, na=False).astype(int)
else:
    filtered["location_util"] = 0.5

# Idő hasznosság: korábban = jobb
if filtered["start_dt"].notna().any():
    wmin, wmax = filtered["waiting_days"].min(), filtered["waiting_days"].max()
    if pd.isna(wmin) or pd.isna(wmax) or wmax == wmin:
        filtered["time_util"] = 0.5
    else:
        wait_norm = (filtered["waiting_days"] - wmin) / (wmax - wmin)
        filtered["time_util"] = 1 - wait_norm
else:
    filtered["time_util"] = 0.5

# Súlyok és végső pontszám
pw, lw, tw = price_w/100, loc_w/100, time_w/100
filtered["score"] = pw*filtered["price_util"] + lw*filtered["location_util"] + tw*filtered["time_util"]

# Eredmények
st.markdown("### Eredmények (magasabb pontszám = jobb)")
show_cols = [inst_col, svc_col, prc_col, "start_dt", "waiting_days", loc_col,
             "price_util", "location_util", "time_util", "score"]
st.dataframe(filtered.sort_values("score", ascending=False)[show_cols].reset_index(drop=True))
