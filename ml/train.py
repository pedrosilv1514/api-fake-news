"""Train a candidate only. The legacy served artifact is never replaced."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.contracts import LABELS
from ml.dataset import load_articles, temporal_splits
from ml.preprocessing import PREPROCESSING_VERSION, normalizar_texto

ROOT = Path(__file__).resolve().parents[1]


def run_training(data_path=None, output_dir=None):
    source = Path(data_path or ROOT / "fake_recogna_limpo.csv")
    version = datetime.now(timezone.utc).strftime("candidate-%Y%m%dT%H%M%S%fZ")
    df, excluded = load_articles(source)
    parts, split_info = temporal_splits(df)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalizar_texto, lowercase=False, token_pattern=r"(?u)R\$|[€$£]|\b\w+(?:[.,/-]\w+)*(?:%|‰|°[cCfF])?", ngram_range=(1,2), max_features=50_000, max_df=.95)),
        ("clf", LinearSVC(C=1, max_iter=5000, random_state=42)),
    ])
    model.fit(parts["train"].text, parts["train"].Classe)
    candidate = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    candidate.fit(parts["calibration"].text, parts["calibration"].Classe)
    out = Path(output_dir or ROOT / "models/candidates") / version
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(candidate, out / "model.joblib")
    predictions = candidate.predict(parts["test"].text)
    p_fake = candidate.predict_proba(parts["test"].text)[:,list(candidate.classes_).index(0)]
    test = parts["test"].Classe.to_numpy()
    cm = confusion_matrix(test, predictions, labels=[0,1])
    metrics = {"n":len(test), "accuracy":accuracy_score(test,predictions), "classification_report":classification_report(test,predictions,output_dict=True), "confusion_matrix":cm.tolist(), "matrix_label_order":["FAKE","REAL"], "fpr":float(cm[1,0]/cm[1].sum()), "fnr":float(cm[0,1]/cm[0].sum()), "brier":brier_score_loss((test==0).astype(int),p_fake), "scope":"historical transformed corpus; not independent recent validation"}
    manifest = {"version":version, "status":"candidate_not_promoted", "preprocessing_version":PREPROCESSING_VERSION, "input_contract":"title + body", "label_mapping":LABELS, "fake_label":0, "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(), "model_sha256":hashlib.sha256((out/"model.joblib").read_bytes()).hexdigest(), "source_quality":"Noticia in local FakeRecogna is already transformed; lost facts cannot be restored", "excluded":excluded, "split_info":split_info, "splits":{s:{"n":len(p), "from":str(p.published.min()), "to":str(p.published.max()), "classes":{str(k):int(v) for k,v in p.Classe.value_counts().items()}} for s,p in parts.items()}, "versions":{n:importlib.metadata.version(n) for n in ["scikit-learn","numpy","pandas","joblib"]}, "recent_validation":False, "production_approved":False}
    for name,obj in [("manifest.json",manifest),("metrics.json",metrics)]:
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    for name,part in parts.items():
        part[["article_id","group_id","text_hash","published","source_domain","Classe"]].to_csv(out/(name+"_split.csv"),index=False)
    parts["test"].assign(predicted=predictions,prob_fake=p_fake)[["article_id","Classe","predicted","prob_fake"]].to_csv(out/"test_predictions.csv",index=False)
    print(json.dumps({"candidate":str(out),"metrics":metrics},ensure_ascii=False),flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    run_training(args.data,args.output)
