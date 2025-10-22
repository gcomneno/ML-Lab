#!/usr/bin/env python3

# ML-Lab — Imbalanced v3 (parla umano, con soglia & analisi)
# - Conta errori in chiaro (TN, FP, FN, TP)
# - Soglia regolabile (--threshold) o automatica da validation (--auto-threshold)
# - Riassunto probabilità, sweep soglia e soglia per costo (FP vs FN)
# - Modello: Logistic Regression con scaling (qui serve!)
# - Appunti di fine-run: riassunto compatto + cheat-sheet opzionale
#
# Uso:
#   python imbalance.py
#   python imbalance.py --threshold 0.6
#   python imbalance.py --auto-threshold
#   python imbalance.py --auto-threshold --metric youden
#   python imbalance.py --seed 13 --C 0.5
#   python imbalance.py --auto-threshold --print-cheatsheet
#   python imbalance.py --auto-threshold --thr-cost --cost-fp 1 --cost-fn 10

import argparse
import numpy as np
from typing import Optional, Dict, Tuple

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

# ---------- Appunti in console (riassunti compatti) ----------
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
    calibrated: Optional[str] = None,              # non usato qui
    class_weight: Optional[str] = None,            # non usato qui
    cost_fp: Optional[float] = None,
    cost_fn: Optional[float] = None
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
            if cost_fp is not None and cost_fn is not None:
                tn, fp, fn, tp = test["cm"]
                costo = cost_fp * fp + cost_fn * fn
                print(f"  Costo(test) con cost_fp={cost_fp} cost_fn={cost_fn}: {costo:.3f}")

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
    print("5) Importanze/coefficienti: spiega sempre cosa spinge su/giù il rischio.")
    print("6) Pipeline/CV per evitare leakage (tutto fittato solo sul TRAIN).")

def print_cheatsheet():
    _print_block("ML — CHEAT-SHEET (compatto)")
    print("• ML = imparare f(input→output) dai dati. Test separato e SACRO.")
    print("• Overfit: alto su train, basso su test → riduci complessità / più dati.")
    print("• Underfit: scarso ovunque → modello più ricco / nuove feature.")
    print("• Metriche: Precision, Recall, F1; ROC-AUC per il ranking globale.")
    print("• Soglia: muoverla scambia FP↔FN. Auto-threshold o soglia a costo.")
    print("• Scaling: Logit/SVM sì; Alberi/RF no.")
    print("• Leakage: imputazione/scaling/encoding sempre dentro la CV.")
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

# ---------- Utility "parla umano" ----------
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

def pick_threshold_cost(scores, y_true, cost_fp=1.0, cost_fn=5.0):
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.0, 1.0, 201):
        yhat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost

def pick_threshold_on_validation(pipe, Xtr, ytr, seed, metric="f1"):
    # separo un piccolo validation dal TRAIN (il TEST non si tocca)
    Xsub, Xval, ysub, yval = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=seed, stratify=ytr
    )
    pipe.fit(Xsub, ysub)
    proba_val = pipe.predict_proba(Xval)[:, 1]
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 201):  # passo 0.005
        yhat = (proba_val >= t).astype(int)
        if metric == "f1":
            score = f1_score(yval, yhat)
        else:
            # Youden J = TPR - FPR
            tn, fp, fn, tp = confusion_matrix(yval, yhat).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score

# ---------- Main ----------
def main(seed=0, C=1.0, threshold=0.5, auto_threshold=False, metric="f1",
         thr_cost=False, cost_fp=1.0, cost_fn=5.0, print_cheat=False):
    data = load_breast_cancer()
    X, y = data.data, data.target  # 1 = maligno (positivo), 0 = benigno

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Baseline pigra: predice sempre la classe più frequente
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    yhat_d = dummy.predict(Xte)
    metrics_report(yte, yhat_d, None, "Baseline (classe maggioritaria)")

    # Pipeline logistica + scaling
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))

    # Soglia: fissa o stimata su validation (sempre solo col TRAIN)
    chosen_thr = threshold
    if auto_threshold:
        chosen_thr, score = pick_threshold_on_validation(pipe, Xtr, ytr, seed, metric=metric)
        print(f"\n[Soglia auto] Scelta soglia={chosen_thr:.3f} ottimizzata per {metric} sul validation (dal TRAIN)")

    # Alleno su tutto il TRAIN e valuto sul TEST
    pipe.fit(Xtr, ytr)
    proba_te = pipe.predict_proba(Xte)[:, 1]
    proba_tr = pipe.predict_proba(Xtr)[:, 1]

    # Analisi punteggi e soglie (non cambia il modello, ti aiuta a capire)
    summarize_scores(yte, proba_te)
    sweep_thresholds(yte, proba_te, start=0.3, stop=0.7, step=0.05)

    # Soglia per costo (sul TEST per capire tradeoff empirico)
    t_cost, tot_cost = pick_threshold_cost(proba_te, yte, cost_fp=cost_fp, cost_fn=cost_fn)
    print(f"\n[Soglia per costo] cost_fp={cost_fp}, cost_fn={cost_fn}  -> t*={t_cost:.3f}, costo={tot_cost:.0f}")

    # Decisione finale con la soglia scelta
    yhat_te = (proba_te >= chosen_thr).astype(int)
    metrics_report(yte, yhat_te, proba_te, f"Logistic Regression (C={C}, soglia={chosen_thr:.3f})")

    # CV su TRAIN (metriche di ranking e classificazione)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_roc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
    cv_f1  = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1")
    print(f"\n[CV ROC-AUC su TRAIN] mean={cv_roc.mean():.3f}  std={cv_roc.std():.3f}")

    # ---------- Appunti di fine-run ----------
    yhat_tr = (proba_tr >= chosen_thr).astype(int)
    train_metrics = pack_metrics(ytr, yhat_tr, proba_tr)
    test_metrics  = pack_metrics(yte, yhat_te, proba_te)

    print_run_summary(
        model_label=f"Logistic Regression (scaled, C={C})",
        threshold=chosen_thr,
        best_params={"C": C},
        seed=seed,
        train=train_metrics,
        test=test_metrics,
        cv_auc=(cv_roc.mean(), cv_roc.std()),
        cv_f1=(cv_f1.mean(),  cv_f1.std()),
        cost_fp=cost_fp,
        cost_fn=cost_fn
    )

    if print_cheat:
        print_cheatsheet()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Imbalanced v3 — report umano + soglia regolabile/automatica + analisi soglie + appunti finali")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--C", type=float, default=1.0, help="Forza del modello logit (C più alto = meno regolarizzazione).")
    p.add_argument("--threshold", type=float, default=0.5, help="Soglia decisione (default 0.5).")
    p.add_argument("--auto-threshold", action="store_true", help="Stima soglia dal TRAIN (validation) per massimizzare F1/Youden.")
    p.add_argument("--metric", choices=["f1", "youden"], default="f1", help="Criterio per auto-soglia.")
    p.add_argument("--thr-cost", action="store_true", help="(Deprecated: la soglia a costo viene solo riportata; usa --cost-fp/--cost-fn)")
    p.add_argument("--cost-fp", type=float, default=1.0, help="Costo di un falso positivo (per report del costo su TEST).")
    p.add_argument("--cost-fn", type=float, default=5.0, help="Costo di un falso negativo (per report del costo su TEST).")
    p.add_argument("--print-cheatsheet", action="store_true", help="Stampa anche il mini cheat-sheet a fine run")
    args = p.parse_args()

    main(seed=args.seed, C=args.C, threshold=args.threshold,
         auto_threshold=args.auto_threshold, metric=args.metric,
         thr_cost=args.thr_cost, cost_fp=args.cost_fp, cost_fn=args.cost_fn,
         print_cheat=args.print_cheatsheet)
