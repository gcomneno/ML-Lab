#!/usr/bin/env python3
# ML-Lab — GridSearch su dati misti (numeriche + categoriche) senza leakage
# + Appunti di fine-run (riassunti compatti in console)
#
# Logit con soglia OOF e appunti finali
# python gridsearch_mixed.py --model logit --auto-threshold --print-cheatsheet
#
# RF con soglia a costo (FN=10x FP) e appunti
# python gridsearch_mixed.py --model rf --auto-threshold --thr-mode cost --cost-fn 10 --print-cheatsheet

import argparse
import numpy as np
from typing import Optional, Dict, Tuple

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# --------------------- Appunti in console (mini-modulo inline) ---------------------

def _fmt(x, nd=3):
    return "nan" if x is None else f"{x:.{nd}f}"

def _print_line():
    print("-" * 72)

def _print_block(title: str):
    _print_line()
    print(title)
    _print_line()

def _fmt_cm_tuple(cm_tuple):
    tn, fp, fn, tp = cm_tuple
    return (
        f"  TN (sani corretti): {tn}\n"
        f"  FP (falsi allarmi): {fp}\n"
        f"  FN (positivi persi): {fn}\n"
        f"  TP (positivi presi): {tp}"
    )

def print_run_summary(
    model_label: str,
    threshold: Optional[float] = None,
    best_params: Optional[Dict] = None,
    seed: Optional[int] = None,
    train: Optional[Dict] = None,   # dict(acc, prec, rec, f1, auc, cm=(tn,fp,fn,tp))
    test: Optional[Dict] = None,
    cv_auc: Optional[Tuple[float, float]] = None,  # (mean, std)
    cv_f1: Optional[Tuple[float, float]] = None,   # (mean, std)
    calibrated: Optional[str] = None,              # 'isotonic' | 'sigmoid' | None
    class_weight: Optional[str] = None             # 'balanced' | None
):
    _print_block(f"RUN SUMMARY — {model_label}")

    print("» Setup")
    if seed is not None:
        print(f"  seed: {seed}")
    if threshold is not None:
        print(f"  soglia decisione: {threshold:.3f}")
    if calibrated:
        print(f"  calibrazione probabilità: {calibrated}")
    if class_weight:
        print(f"  class_weight: {class_weight}")
    if best_params:
        print(f"  best params: {best_params}")

    if train:
        print("\n» Train metrics")
        print(f"  Acc={_fmt(train.get('acc'))} | Prec={_fmt(train.get('prec'))} | "
              f"Rec={_fmt(train.get('rec'))} | F1={_fmt(train.get('f1'))} | AUC={_fmt(train.get('auc'))}")
        if train.get("cm") is not None:
            print(_fmt_cm_tuple(train["cm"]))

    if test:
        print("\n» Test metrics")
        print(f"  Acc={_fmt(test.get('acc'))} | Prec={_fmt(test.get('prec'))} | "
              f"Rec={_fmt(test.get('rec'))} | F1={_fmt(test.get('f1'))} | AUC={_fmt(test.get('auc'))}")
        if test.get("cm") is not None:
            print(_fmt_cm_tuple(test["cm"]))

    if cv_auc or cv_f1:
        print("\n» Cross-Validation (train-only)")
        if cv_auc:
            mu, sd = cv_auc
            print(f"  ROC-AUC (mean±std): {_fmt(mu)} ± {_fmt(sd)}")
        if cv_f1:
            mu, sd = cv_f1
            print(f"  F1      (mean±std): {_fmt(mu)} ± {_fmt(sd)}")

    _print_block("REGOLE FLASH")
    print("1) Mai usare il TEST per scegliere iperparametri o soglia (test è sacro).")
    print("2) Con sbilanciamento valuta F1/Recall e la confusion; accuracy da sola inganna.")
    print("3) La soglia 0.5 non è legge: prova auto-threshold (F1/Youden) o soglia a costo.")
    print("4) Scaling per Logit/SVM; non serve per Alberi/Random Forest.")
    print("5) RF: scegli bene la soglia; calibra (isotonic/sigmoid) se vuoi probabilità affidabili.")
    print("6) Importanze: guarda impurity + permutation; se togli top-1 e l’AUC non scende, c’è ridondanza.")
    print("7) Pipeline sempre: imputazione/scaling/encoding dentro la CV (no leakage).")

def print_cheatsheet():
    _print_block("ML — CHEAT-SHEET (compatto)")
    print("• ML = imparare f(input→output) dai dati. Test separato e SACRO.")
    print("• Overfit: alto su train, basso su test → riduci complessità / più dati.")
    print("• Underfit: scarso ovunque → modello più ricco / nuove feature.")
    print("• Metriche: Precision, Recall, F1; ROC-AUC per il ranking globale.")
    print("• Soglia: muoverla scambia FP↔FN. Auto-threshold o soglia a costo.")
    print("• Scaling: Logit/SVM sì; Alberi/RF no.")
    print("• Random Forest: ottima ma score spesso poco calibrati; valuta calibrazione.")
    print("• Importanze: non fidarti solo di impurity; usa permutation + ablation.")
    print("• Pipeline + GridSearchCV sulla pipeline: no leakage.")
    _print_line()

def pack_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred),
        rec=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_proba),
        cm=cm
    )

# ---------------- Dataset sintetico (numeriche + categoriche) ----------------

def make_mixed_dataset(n=1200, n_num=10, n_cat=3, pos_ratio=0.35, missing_num=0.10, missing_cat=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X_num, y = make_classification(n_samples=n, n_features=n_num, n_informative=6,
                                   n_redundant=2, n_repeated=0, weights=[1-pos_ratio, pos_ratio],
                                   class_sep=1.5, random_state=seed)

    # Derivo categoriche da numeriche (binnate) + un po’ di rumore
    cats = []
    for j in range(n_cat):
        col = X_num[:, j % n_num]
        bins = np.quantile(col, [0.0, 0.33, 0.66, 1.0])
        lab = np.digitize(col, bins[1:-1], right=False)
        labels = np.array(["low", "mid", "high"])[lab]
        flip = rng.rand(n) < 0.05
        labels[flip] = rng.choice(["low", "mid", "high"], size=flip.sum())
        cats.append(labels.reshape(-1, 1))
    X_cat = np.hstack(cats) if n_cat > 0 else np.empty((n, 0), dtype=object)

    # Missing
    if missing_num > 0:
        m = rng.rand(*X_num.shape) < missing_num
        X_num = X_num.astype(float); X_num[m] = np.nan
    if n_cat > 0 and missing_cat > 0:
        m = rng.rand(*X_cat.shape) < missing_cat
        X_cat = X_cat.astype(object); X_cat[m] = None

    X = np.hstack([X_num, X_cat]) if n_cat > 0 else X_num
    num_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_num + n_cat))
    return X, y, num_idx, cat_idx

# ---------------- Utility stampa ----------------

def speak_confusion(cm):
    tn, fp, fn, tp = cm.ravel()
    print(f"  Sani corretti (TN): {tn}")
    print(f"  Falsi allarmi (FP): {fp}")
    print(f"  Positivi persi (FN): {fn}")
    print(f"  Positivi presi (TP): {tp}")

def report_metrics(y_true, y_pred, y_prob=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n== {label} ==")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print("Confusion matrix (righe=vero, colonne=predetto):")
    print(cm)
    speak_confusion(cm)

# ---------------- Scelta soglia onesta (OOF) ----------------

def choose_threshold_from_oof(y_true, oof_scores, mode="f1", cost_fp=1.0, cost_fn=5.0):
    best_t, best_val = 0.5, -1e9
    for t in np.linspace(0.0, 1.0, 1001):
        yhat = (oof_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        if mode == "f1":
            val = f1_score(y_true, yhat)
        elif mode == "youden":
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            val = tpr - fpr
        elif mode == "cost":
            val = -(cost_fp * fp + cost_fn * fn)
        else:
            val = f1_score(y_true, yhat)
        if val > best_val:
            best_val, best_t = val, t
    return best_t, best_val

# ---------------- Feature names dopo il preproc ----------------

def get_feature_names(preprocessor, num_idx, cat_idx):
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        out = [f"num__x{i}" for i in num_idx] + [f"cat__x{j}" for j in cat_idx]
        return out

# ---------------- Main ----------------

def main(model="logit", seed=0, n=1200, missing=0.10,
         auto_threshold=True, thr_mode="f1", cost_fp=1.0, cost_fn=5.0,
         print_cheat=False):

    X, y, num_idx, cat_idx = make_mixed_dataset(n=n, n_num=10, n_cat=3, missing_num=missing, seed=seed)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_idx),
        ("cat", cat_pipe, cat_idx)
    ])

    if model == "logit":
        clf = LogisticRegression(max_iter=5000, solver="liblinear")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        param_grid = {
            "clf__C": [0.1, 0.3, 1, 3, 10],
            "clf__class_weight": [None, "balanced"]
        }
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        param_grid = {
            "clf__max_depth": [None, 8, 12],
            "clf__max_features": ["sqrt", "log2"],
            "clf__class_weight": [None, "balanced"]
        }
    else:
        raise ValueError("--model deve essere 'logit' o 'rf'")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1", cv=cv, n_jobs=-1, return_train_score=False
    )
    grid.fit(Xtr, ytr)

    print("\n[GridSearch] Top configurazioni (ordinate per F1 CV):")
    results = grid.cv_results_
    order = np.argsort(-results["mean_test_f1"])
    for i in order[:5]:
        params = results["params"][i]
        mean_f1 = results["mean_test_f1"][i]; std_f1 = results["std_test_f1"][i]
        mean_auc = results["mean_test_roc_auc"][i]
        print(f"  F1={mean_f1:.3f}±{std_f1:.3f}  AUC={mean_auc:.3f}  params={params}")

    best = grid.best_estimator_
    print(f"\n[Scelgo] {grid.best_params_} (refit su F1)")

    # ---- Soglia onesta: OOF sul TRAIN con pipeline migliore ----
    oof_prob = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    if auto_threshold:
        thr, val = choose_threshold_from_oof(ytr, oof_prob, mode=thr_mode, cost_fp=cost_fp, cost_fn=cost_fn)
        tag = thr_mode if thr_mode != "cost" else f"cost(fp={cost_fp},fn={cost_fn})"
        print(f"\n[Soglia] scelta onesta su OOF: thr={thr:.3f} (criterio={tag})")
    else:
        thr = 0.5
        print("\n[Soglia] uso default 0.5")

    # ---- Fit finale su TRAIN completo e valutazione su TEST ----
    best.fit(Xtr, ytr)
    prob_tr = best.predict_proba(Xtr)[:, 1]
    prob_te = best.predict_proba(Xte)[:, 1]
    yhat_tr = (prob_tr >= thr).astype(int)
    yhat_te = (prob_te >= thr).astype(int)

    # Report classico
    report_metrics(ytr, yhat_tr, prob_tr, f"{model.upper()} (TRAIN, soglia={thr:.3f})")
    report_metrics(yte, yhat_te, prob_te, f"{model.upper()} (TEST, soglia={thr:.3f})")

    # Extra: nomi feature + importanze/coef
    preproc = best.named_steps["pre"]
    feat_names = get_feature_names(preproc, num_idx, cat_idx)

    if model == "rf":
        rf = best.named_steps["clf"]
        try:
            importances = rf.feature_importances_
            idx = np.argsort(importances)[::-1][:10]
            print("\n[RF] Top 10 feature importanti:")
            for i in idx:
                name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
                print(f"- {name:35s} {importances[i]:.3f}")
        except Exception as e:
            print(f"\n[RF] Impossibile leggere importances: {e}")
    else:
        logit = best.named_steps["clf"]
        try:
            coefs = np.ravel(logit.coef_)
            idx_pos = np.argsort(-coefs)[:6]
            idx_neg = np.argsort(coefs)[:6]
            print("\n[Logit] Top coefficienti (positivi = aumentano rischio; negativi = lo riducono)")
            for i in idx_pos:
                nm = feat_names[i] if i < len(feat_names) else f"feat_{i}"
                print(f"+ {nm:35s} {coefs[i]: .3f}")
            for i in idx_neg:
                nm = feat_names[i] if i < len(feat_names) else f"feat_{i}"
                print(f"- {nm:35s} {coefs[i]: .3f}")
        except Exception as e:
            print(f"\n[Logit] Impossibile leggere coeff: {e}")

    # ---- CV extra per riassunto (AUC/F1 sul train) ----
    cv_auc_scores = cross_val_score(best, Xtr, ytr, cv=cv, scoring="roc_auc")
    cv_f1_scores  = cross_val_score(best, Xtr, ytr, cv=cv, scoring="f1")

    # ---- Appunti di fine-run (riassunti compatti) ----
    train_metrics = pack_metrics(ytr, yhat_tr, prob_tr)
    test_metrics  = pack_metrics(yte, yhat_te, prob_te)

    class_weight = None
    if model == "logit":
        # prova a leggere dai parametri del best estimator
        params = grid.best_params_
        if "clf__class_weight" in params and params["clf__class_weight"] == "balanced":
            class_weight = "balanced"
    else:
        params = grid.best_params_
        if "clf__class_weight" in params and params["clf__class_weight"] == "balanced":
            class_weight = "balanced"

    print_run_summary(
        model_label=f"{model.upper()} — best grid",
        threshold=thr,
        best_params=grid.best_params_,
        seed=seed,
        train=train_metrics,
        test=test_metrics,
        cv_auc=(cv_auc_scores.mean(), cv_auc_scores.std()),
        cv_f1=(cv_f1_scores.mean(),  cv_f1_scores.std()),
        calibrated=None,
        class_weight=class_weight
    )

    if print_cheat:
        print_cheatsheet()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GridSearch sicura (Pipeline, numeriche+categoriche) con soglia onesta e appunti finali")
    ap.add_argument("--model", choices=["logit", "rf"], default="logit", help="Modello: logit o rf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1200, help="Campioni totali sintetici")
    ap.add_argument("--missing", type=float, default=0.10, help="Quota missing nelle numeriche")
    ap.add_argument("--auto-threshold", action="store_true", help="Scegli soglia da OOF (default: False)")
    ap.add_argument("--thr-mode", choices=["f1", "youden", "cost"], default="f1", help="Criterio soglia OOF")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    ap.add_argument("--print-cheatsheet", action="store_true", help="Stampa anche il mini cheat-sheet a fine run")
    args = ap.parse_args()

    main(model=args.model, seed=args.seed, n=args.n, missing=args.missing,
         auto_threshold=args.auto_threshold, thr_mode=args.thr_mode,
         cost_fp=args.cost_fp, cost_fn=args.cost_fn, print_cheat=args.print_cheatsheet)
