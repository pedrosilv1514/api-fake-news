"""Auditable temporal article/group splits; never fall back to cleaned text."""
from collections import Counter
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from ml.preprocessing import compor_texto, normalizar_texto


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_articles(path: Path):
    df = pd.read_csv(path)
    required = {"Titulo", "Noticia", "Classe", "Data", "URL"}
    if not required <= set(df):
        raise ValueError(f"Exige colunas de artigo {sorted(required)}; texto_limpo não é original")
    if not df.Classe.isin([0, 1]).all():
        raise ValueError("Classe deve ser 0=FAKE ou 1=REAL; feedback não revisado não é rótulo")
    df = df.copy()
    df["text"] = [compor_texto(str(t), str(b)) for t,b in zip(df.Titulo.fillna(""), df.Noticia.fillna(""))]
    df["published"] = pd.to_datetime(df.Data.astype(str).str.extract(r"(\d{1,2}/\d{1,2}/\d{4})")[0], format="%d/%m/%Y", errors="coerce")
    df["article_id"] = [digest(str(u) + "\n" + normalizar_texto(t)) for u,t in zip(df.URL.fillna(""), df.text)]
    df["source_domain"] = df.URL.fillna("").map(lambda u: urlparse(u).hostname or "")
    excluded = {"undated": int(df.published.isna().sum()), "empty_body": int(df.Noticia.fillna("").str.strip().eq("").sum())}
    df = df[df.published.notna() & df.Noticia.fillna("").str.strip().ne("")].copy()
    parent = {i:i for i in df.index}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    tokens = {i:set(re.findall(r"\w+", normalizar_texto(str(t)))) for i,t in df.Titulo.fillna("").items()}
    frequencies = Counter(t for ts in tokens.values() for t in ts)
    bodies, urls, buckets = {}, {}, {}
    for i,row in df.iterrows():
        # Group exact copies/URLs and approximate titles before choosing date boundaries.
        body_key = digest(normalizar_texto(str(row.Noticia)))
        for key, index in [(body_key,bodies),(str(row.URL),urls)]:
            if key and key != "nan":
                if key in index: parent[find(i)] = find(index[key])
                else: index[key] = i
        anchors = sorted(tokens[i], key=lambda t:(frequencies[t],t))[:3]
        candidates = {j for token in anchors for j in buckets.get(token, [])}
        for j in candidates:
            union = tokens[i] | tokens[j]
            if union and len(tokens[i] & tokens[j])/len(union) >= .85:
                parent[find(i)] = find(j)
        for anchor in anchors: buckets.setdefault(anchor, []).append(i)
    df["group_id"] = [digest(str(find(i))) for i in df.index]
    before = len(df)
    df["text_hash"] = df.text.map(lambda t:digest(normalizar_texto(t)))
    conflicting = df.groupby("text_hash").Classe.nunique()
    df = df[~df.text_hash.isin(conflicting[conflicting>1].index)].drop_duplicates("text_hash").copy()
    excluded["conflicting_or_duplicate_texts"] = before - len(df)
    return df, excluded


def temporal_splits(df):
    groups = df.groupby("group_id").published.agg(["min", "max"]).sort_values("max")
    dates = sorted(groups["max"].unique())
    if len(dates) < 5:
        raise ValueError("Datas insuficientes para split temporal")
    train_end = dates[int((len(dates)-1)*.60)]
    calibration_end = dates[int((len(dates)-1)*.80)]
    group_to_split = {}
    for g,row in groups.iterrows():
        if row["max"] <= train_end: group_to_split[g] = "train"
        elif row["min"] > train_end and row["max"] <= calibration_end: group_to_split[g] = "calibration"
        elif row["min"] > calibration_end: group_to_split[g] = "test"
        else: group_to_split[g] = "purged_boundary_group"
    assigned = df.assign(split=df.group_id.map(group_to_split))
    parts = {s:assigned[assigned.split==s].copy() for s in ["train","calibration","test"]}
    for name,part in parts.items():
        if part.Classe.nunique()!=2: raise ValueError(f"Split {name} não contém as duas classes")
    for left,right in [("train","calibration"),("train","test"),("calibration","test")]:
        assert not set(parts[left].group_id)&set(parts[right].group_id)
        assert parts[left].published.max() < parts[right].published.min()
    return parts, {"train_end":str(train_end), "calibration_end":str(calibration_end), "purged_rows":int((assigned.split=="purged_boundary_group").sum()), "grouping_limit":"exact body/URL and title Jaccard >=0.85 over rare-token candidates; narrative independence requires review"}
