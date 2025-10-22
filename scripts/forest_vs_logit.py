#!/usr/bin/env python3
# ML-Lab — Random Forest vs Logistic (parla umano) + calibrazione RF (fix importances) + appunti di fine-run
#
# python forest_vs_logit.py
#
# con soglie auto + appunti + cheat-sheet
# python forest_vs_logit.py --auto-threshold --print-cheatsheet

import argparse
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.base import clone

# --------------------- Appunti in console (mini-modulo inline) ---------------------

from typing import Optional, Dict, Tuple

def _fmt(x, nd=3):
    return "nan" if x is None else f"{x:.{nd}f}"

def _print_line():
    print("-" * 72)

def _print_block(title: str):
    _print_line()
    print(title)
    _print_line()

def _fmt_cm_tuple(cm_tuple: Tuple[int, int, int, int]):
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

# --------------------- Funzioni esistenti ---------------------

def speak_confusion(cm):
    tn, fp, fn, tp = cm.ravel()
    print(f"  Sani corretti (TN): {tn}")
    print(f"  Falsi allarmi (FP): {fp}")
    print(f"  Positivi persi (FN): {fn}")
    print(f"  Positivi presi (TP): {tp}")

def metrics_report(y_true, y_pred, y_score=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n== {label} ==")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | ROC-AUC: {roc:.3f}")
    print("Confusion matrix (righe=vero, colonne=predetto):")
    print(cm)
    speak_confusion(cm)

def summarize_scores(y_true, scores):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    def qdesc(a):
        qs = np.quantile(a, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        return " ".join(f"{q:.3f}" for q in qs)
    print("\n[Distribuzione punteggi]")
    print("Negativi (benigni) quantili:", qdesc(neg))
    print("Positivi (maligni) quantili:", qdesc(pos))
    near = np.abs(scores - 0.5) < 0.02
    print(f"Casi vicini alla soglia 0.5 ±0.02: {near.sum()}")

def sweep_thresholds(y_true, scores, start=0.3, stop=0.7, step=0.05):
    print("\n[Sweep soglia]   thr   FP   FN   Precision  Recall     F1")
    best_f1, best_t = -1.0, 0.5
    t = start
    while t <= stop + 1e-12:
        yhat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"                {t:4.2f}  {fp:3d}  {fn:3d}    {prec:8.3f}  {rec:7.3f}  {f1:7.3f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
        t += step
    print(f"[Sweep] Miglior F1 a soglia {best_t:.3f}")

def pick_threshold_on_validation(proba_model, Xtr, ytr, seed, metric="f1"):
    from sklearn.metrics import f1_score
    Xsub, Xval, ysub, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=seed, stratify=ytr)
    proba_model.fit(Xsub, ysub)
    scores = proba_model.predict_proba(Xval)[:, 1]
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 201):
        yhat = (scores >= t).astype(int)
        if metric == "f1":
            score = f1_score(yval, yhat)
        else:
            tn, fp, fn, tp = confusion_matrix(yval, yhat).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
        if score > best_score:
            best_score, best_t = score, t
    return best_t

# Helper per riassunti
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

def main(seed=0, threshold=0.5, auto_threshold=False, metric="f1",
         rf_tune=False, rf_class_weight=None, rf_n=300, rf_depth=None, rf_mf="sqrt",
         rf_calibrate=None, C=1.0, print_cheat=False):

    data = load_breast_cancer()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    yhat_d = dummy.predict(Xte)
    metrics_report(yte, yhat_d, None, "Baseline (classe maggioritaria)")

    logit_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))

    rf_base = RandomForestClassifier(
        n_estimators=rf_n, max_depth=rf_depth, max_features=rf_mf,
        random_state=seed, n_jobs=-1, class_weight=rf_class_weight
    )

    if rf_tune:
        print("\n[Tuning RF] Valutazione su TRAIN (k=5, metrica ROC-AUC):")
        grid = {"n_estimators": [200, 400], "max_depth": [None, 8, 12], "max_features": ["sqrt", "log2"]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        best_auc, best_cfg = -1.0, None
        for n in grid["n_estimators"]:
            for d in grid["max_depth"]:
                for mf in grid["max_features"]:
                    model = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=mf,
                                                   random_state=seed, n_jobs=-1, class_weight=rf_class_weight)
                    aucs = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc")
                    print(f"  n={n:3d} depth={str(d):>4} mf={mf:>4}  -> ROC-AUC mean={aucs.mean():.3f} ± {aucs.std():.3f}")
                    if aucs.mean() > best_auc:
                        best_auc, best_cfg = aucs.mean(), (n, d, mf)
        rf_n, rf_depth, rf_mf = best_cfg
        print(f"[Tuning RF] Scelgo n={rf_n}, depth={rf_depth}, max_features={rf_mf}")
        rf_base = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, max_features=rf_mf,
                                         random_state=seed, n_jobs=-1, class_weight=rf_class_weight)

    if rf_calibrate:
        print(f"\n[RF] Calibrazione probabilità: metodo={rf_calibrate} (CV=5 sul TRAIN)")
        rf = CalibratedClassifierCV(rf_base, method=rf_calibrate, cv=5)
    else:
        rf = rf_base

    # Fit
    logit_pipe.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)

    # Soglie auto (su TRAIN)
    thr_logit = threshold
    thr_rf = threshold
    if auto_threshold:
        thr_logit = pick_threshold_on_validation(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C)),
                                                Xtr, ytr, seed, metric)
        thr_rf = pick_threshold_on_validation(rf, Xtr, ytr, seed, metric)
        print(f"\n[Soglia auto] logit={thr_logit:.3f}  rf={thr_rf:.3f}  (ottimizzate su validation dal TRAIN)")

    # Probabilità TEST
    proba_logit = logit_pipe.predict_proba(Xte)[:, 1]
    proba_rf    = rf.predict_proba(Xte)[:, 1]

    print("\n--- Random Forest: analisi punteggi ---")
    summarize_scores(yte, proba_rf)
    sweep_thresholds(yte, proba_rf, start=0.3, stop=0.7, step=0.05)

    yhat_logit = (proba_logit >= thr_logit).astype(int)
    yhat_rf    = (proba_rf    >= thr_rf).astype(int)

    metrics_report(yte, yhat_logit, proba_logit, f"Logistic Regression (C={C}, soglia={thr_logit:.3f})")
    rf_desc_n, rf_desc_dep, rf_desc_mf = rf_base.n_estimators, rf_base.max_depth, rf_base.max_features
    metrics_report(yte, yhat_rf, proba_rf,
                   f"Random Forest (n={rf_desc_n}, depth={rf_desc_dep}, mf={rf_desc_mf}, calibrate={rf_calibrate}, soglia={thr_rf:.3f})")

    # >>> FIX: se RF è calibrata, rf_base NON è fit. Fittiamo un clone per le importanze.
    rf_for_importance = clone(rf_base).fit(Xtr, ytr)
    importances = rf_for_importance.feature_importances_
    names = load_breast_cancer().feature_names
    order = np.argsort(importances)[::-1]
    print("\n[RF] Top 10 feature importanti:")
    for i in order[:10]:
        print(f"- {names[i]:30s}  {importances[i]:.3f}")

    # CV (AUC e F1) su TRAIN per entrambi i modelli
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    logit_cv_auc = cross_val_score(logit_pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
    rf_cv_auc    = cross_val_score(rf,          Xtr, ytr, cv=cv, scoring="roc_auc")
    logit_cv_f1  = cross_val_score(logit_pipe, Xtr, ytr, cv=cv, scoring="f1")
    rf_cv_f1     = cross_val_score(rf,          Xtr, ytr, cv=cv, scoring="f1")

    print(f"\n[CV ROC-AUC su TRAIN]  logit mean={logit_cv_auc.mean():.3f} ± {logit_cv_auc.std():.3f} | "
          f"rf mean={rf_cv_auc.mean():.3f} ± {rf_cv_auc.std():.3f}")

    # --------------------- Appunti di fine-run (riassunti compatti) ---------------------

    # Metriche TRAIN (con soglia applicata al train)
    proba_logit_tr = logit_pipe.predict_proba(Xtr)[:, 1]
    yhat_logit_tr  = (proba_logit_tr >= thr_logit).astype(int)
    logit_train_metrics = pack_metrics(ytr, yhat_logit_tr, proba_logit_tr)

    proba_rf_tr = rf.predict_proba(Xtr)[:, 1]
    yhat_rf_tr  = (proba_rf_tr >= thr_rf).astype(int)
    rf_train_metrics = pack_metrics(ytr, yhat_rf_tr, proba_rf_tr)

    # Metriche TEST già pronte
    logit_test_metrics = pack_metrics(yte, yhat_logit, proba_logit)
    rf_test_metrics    = pack_metrics(yte, yhat_rf,    proba_rf)

    # Riassunto LOGIT
    print_run_summary(
        model_label=f"Logistic Regression (C={C})",
        threshold=thr_logit,
        best_params={"C": C},
        seed=seed,
        train=logit_train_metrics,
        test=logit_test_metrics,
        cv_auc=(logit_cv_auc.mean(), logit_cv_auc.std()),
        cv_f1=(logit_cv_f1.mean(), logit_cv_f1.std()),
        calibrated=None,
        class_weight=None
    )

    # Riassunto RF
    rf_best_params = {"n_estimators": rf_desc_n, "max_depth": rf_desc_dep, "max_features": rf_desc_mf}
    print_run_summary(
        model_label="Random Forest" + (f" (calibrated={rf_calibrate})" if rf_calibrate else ""),
        threshold=thr_rf,
        best_params=rf_best_params,
        seed=seed,
        train=rf_train_metrics,
        test=rf_test_metrics,
        cv_auc=(rf_cv_auc.mean(), rf_cv_auc.std()),
        cv_f1=(rf_cv_f1.mean(), rf_cv_f1.std()),
        calibrated=rf_calibrate,
        class_weight=("balanced" if rf_base.class_weight == "balanced" else None)
    )

    if print_cheat:
        print_cheatsheet()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random Forest vs Logistic — confronto chiaro con soglia, importanze e calibrazione.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--auto-threshold", action="store_true")
    p.add_argument("--metric", choices=["f1", "youden"], default="f1")
    p.add_argument("--rf-tune", action="store_true")
    p.add_argument("--rf-class-weight", choices=[None, "balanced"], default=None)
    p.add_argument("--rf-n", type=int, default=300)
    p.add_argument("--rf-depth", type=lambda s: None if s.lower()=="none" else int(s), default=None)
    p.add_argument("--rf-mf", choices=["sqrt","log2"], default="sqrt")
    p.add_argument("--rf-calibrate", choices=[None, "isotonic", "sigmoid"], default=None)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--print-cheatsheet", action="store_true", help="Stampa anche il mini cheat-sheet a fine run")
    args = p.parse_args()
    rf_class_weight = None if args.rf_class_weight in (None, "None") else "balanced"
    main(seed=args.seed, threshold=args.threshold, auto_threshold=args.auto_threshold, metric=args.metric,
         rf_tune=args.rf_tune, rf_class_weight=rf_class_weight,
         rf_n=args.rf_n, rf_depth=args.rf_depth, rf_mf=args.rf_mf,
         rf_calibrate=args.rf_calibrate, C=args.C, print_cheat=args.print_cheatsheet)
