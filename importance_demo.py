#!/usr/bin/env python3
# ML-Lab — Feature Importance demo (RF): impurity vs permutation, correlazioni, ablation
# + Appunti di fine-run (riassunto compatto)

import argparse
import numpy as np
from typing import Optional, Dict, Tuple

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# --------------------- Appunti in console (mini-modulo inline, binary) ---------------------

def _fmt(x, nd=3): return "nan" if x is None else f"{x:.{nd}f}"
def _print_line(): print("-" * 72)
def _print_block(title: str): _print_line(); print(title); _print_line()

def _fmt_cm_tuple(cm_tuple):
    tn, fp, fn, tp = cm_tuple
    return (f"  TN (sani corretti): {tn}\n"
            f"  FP (falsi allarmi): {fp}\n"
            f"  FN (positivi persi): {fn}\n"
            f"  TP (positivi presi): {tp}")

def print_run_summary(
    model_label: str,
    threshold: Optional[float] = None,
    seed: Optional[int] = None,
    train: Optional[Dict] = None,   # dict(acc, auc, cm)
    test: Optional[Dict] = None,
    cv_auc: Optional[Tuple[float, float]] = None
):
    _print_block(f"RUN SUMMARY — {model_label}")

    print("» Setup")
    if seed is not None:
        print(f"  seed: {seed}")
    if threshold is not None:
        print(f"  soglia decisione: {threshold:.3f}")

    if train:
        print("\n» Train metrics")
        print(f"  Acc={_fmt(train.get('acc'))} | AUC={_fmt(train.get('auc'))}")
        if train.get("cm") is not None:
            print(_fmt_cm_tuple(train["cm"]))

    if test:
        print("\n» Test metrics")
        print(f"  Acc={_fmt(test.get('acc'))} | AUC={_fmt(test.get('auc'))}")
        if test.get("cm") is not None:
            print(_fmt_cm_tuple(test["cm"]))

    if cv_auc:
        mu, sd = cv_auc
        print("\n» Cross-Validation (train-only)")
        print(f"  ROC-AUC (mean±std): {_fmt(mu)} ± {_fmt(sd)}")

    _print_block("REGOLE FLASH")
    print("1) Impurity importance è veloce ma può favorire feature con molte soglie e si diluisce con feature correlate.")
    print("2) Permutation importance misura la perdita reale su TEST: spesso più onesta.")
    print("3) Correlazioni alte ⇒ importanze si “spartiscono”: controlla coppie |corr|≥0.9.")
    print("4) Ablation (drop top-k) per testare la robustezza del segnale.")

def print_cheatsheet():
    _print_block("Feature Importance — CHEAT")
    print("• Impurity: utile per una vista rapida; occhio a bias e correlazioni.")
    print("• Permutation: misura la caduta di metrica quando rompi la feature (su TEST).")
    print("• Correlazioni alte: più feature raccontano la stessa storia.")
    print("• Ablation: togli top-1/top-k e verifica quanto scende l’AUC.")
    _print_line()

def pack_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        acc=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_proba),
        cm=cm
    )

# --------------------- Utility stampa ---------------------

def print_top_k(names, values, k=12, title="[Top]"):
    order = np.argsort(values)[::-1]
    print(f"\n{title} Top {k}")
    for i in order[:k]:
        print(f"- {names[i]:30s} {values[i]:.4f}")

def correlated_pairs(X, names, thr=0.9, max_show=10):
    C = np.corrcoef(X, rowvar=False)
    pairs = []
    p = C.shape[0]
    for i in range(p):
        for j in range(i+1, p):
            r = C[i, j]
            if np.isnan(r): 
                continue
            if abs(r) >= thr:
                pairs.append((i, j, r))
    pairs.sort(key=lambda t: -abs(t[2]))
    if len(pairs) == 0:
        print("\n[Coppie molto correlate] nessuna con |corr|>=", thr)
        return []
    print("\n[Coppie molto correlate (|corr|>=0.9) — attenzione a interpretazione]")
    for i, (a, b, r) in enumerate(pairs[:max_show]):
        print(f"- {names[a]:30s} ~ {names[b]:30s} corr={r:.3f}")
    return pairs

# --------------------- Main ---------------------

def main(seed=0, n_estimators=300, max_depth=None, threshold=0.5, top=12, print_cheat=False):
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = data.feature_names

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1
    )
    rf.fit(Xtr, ytr)

    proba_tr = rf.predict_proba(Xtr)[:, 1]
    proba_te = rf.predict_proba(Xte)[:, 1]
    yhat_te  = (proba_te >= threshold).astype(int)
    yhat_tr  = (proba_tr >= threshold).astype(int)

    auc_te = roc_auc_score(yte, proba_te)
    acc_te = accuracy_score(yte, yhat_te)
    print(f"RandomForest base (soglia {threshold:.1f}):  ROC-AUC={auc_te:.3f}  |  Accuracy={acc_te:.3f}")

    # Impurity importance
    imp = rf.feature_importances_
    print_top_k(names, imp, k=top, title="[RF impurity importance]")

    # Permutation importance (TEST, su ROC-AUC)
    pi = permutation_importance(
        rf, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1
    )
    print_top_k(names, pi.importances_mean, k=top, title="[Permutation importance (ROC-AUC, TEST)]")

    # Correlazioni molto alte (per attenzione interpretativa)
    _ = correlated_pairs(X, names, thr=0.9, max_show=10)

    # Ablation test: rimuovo top-1 e top-3 per impurity
    order = np.argsort(imp)[::-1]
    base_auc = auc_te
    for k in [1, 3]:
        drop_idx = set(order[:k])
        keep = [j for j in range(X.shape[1]) if j not in drop_idx]
        rf2 = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=seed, n_jobs=-1
        )
        rf2.fit(Xtr[:, keep], ytr)
        auc2 = roc_auc_score(yte, rf2.predict_proba(Xte[:, keep])[:, 1])
        if k == 1:
            tag = "top-1"
        else:
            tag = "top-3"
        print(f"\n[Ablation] AUC base={base_auc:.3f} | drop {tag} → AUC={auc2:.3f}")

    # CV su TRAIN (AUC) per contesto
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(rf, Xtr, ytr, cv=cv, scoring="roc_auc")

    # Appunti di fine-run
    train_metrics = pack_metrics(ytr, yhat_tr, proba_tr)
    test_metrics  = pack_metrics(yte, yhat_te, proba_te)
    print_run_summary(
        model_label=f"RandomForest (n={n_estimators}, depth={max_depth})",
        threshold=threshold,
        seed=seed,
        train=train_metrics,
        test=test_metrics,
        cv_auc=(cv_auc.mean(), cv_auc.std())
    )

    if print_cheat:
        print_cheatsheet()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Feature importance demo: impurity vs permutation, correlazioni e ablation (RF)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=lambda s: None if s.lower()=="none" else int(s), default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=12, help="Quante feature mostrare nelle liste top")
    ap.add_argument("--print-cheatsheet", action="store_true")
    args = ap.parse_args()

    main(seed=args.seed, n_estimators=args.n_estimators, max_depth=args.max_depth,
         threshold=args.threshold, top=args.top, print_cheat=args.print_cheatsheet)
