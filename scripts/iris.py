#!/usr/bin/env python3
# ML-Lab — Decision Tree su Iris (parla umano) + tuning onesto + appunti di fine-run

import argparse
import numpy as np
from typing import Optional, Dict, Tuple

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# --------------------- Appunti in console (mini-modulo inline, multiclass-aware) ---------------------

def _fmt(x, nd=3):
    return "nan" if x is None else f"{x:.{nd}f}"

def _print_line():
    print("-" * 72)

def _print_block(title: str):
    _print_line()
    print(title)
    _print_line()

def _print_cm_table(cm, labels):
    # stampa confusion matrix con header leggibili
    w = max(7, max(len(l) for l in labels) + 2)
    hdr = " " * (w) + "".join(f"{lab:>{w}s}" for lab in labels)
    print(hdr)
    for i, lab in enumerate(labels):
        row = "".join(f"{cm[i, j]:>{w}d}" for j in range(len(labels)))
        print(f"{lab:>{w}s}{row}")

def print_run_summary_multiclass(
    model_label: str,
    seed: Optional[int] = None,
    train_acc: Optional[float] = None,
    train_f1: Optional[float] = None,
    test_acc: Optional[float] = None,
    test_f1: Optional[float] = None,
    cm_test=None,
    labels=None,
    cv_acc: Optional[Tuple[float, float]] = None,
    cv_f1: Optional[Tuple[float, float]] = None,
    best_params: Optional[Dict] = None,
):
    _print_block(f"RUN SUMMARY — {model_label}")

    print("» Setup")
    if seed is not None:
        print(f"  seed: {seed}")
    if best_params:
        print(f"  best params: {best_params}")

    if (train_acc is not None) or (train_f1 is not None):
        print("\n» Train metrics")
        print(f"  Acc={_fmt(train_acc)} | F1-macro={_fmt(train_f1)}")

    if (test_acc is not None) or (test_f1 is not None):
        print("\n» Test metrics")
        print(f"  Acc={_fmt(test_acc)} | F1-macro={_fmt(test_f1)}")
        if cm_test is not None and labels is not None:
            print("  Confusion matrix (righe=vero, colonne=predetto):")
            _print_cm_table(cm_test, labels)

    if cv_acc or cv_f1:
        print("\n» Cross-Validation (train-only)")
        if cv_acc:
            mu, sd = cv_acc
            print(f"  Accuracy (mean±std): {_fmt(mu)} ± {_fmt(sd)}")
        if cv_f1:
            mu, sd = cv_f1
            print(f"  F1-macro (mean±std): {_fmt(mu)} ± {_fmt(sd)}")

    _print_block("REGOLE FLASH")
    print("1) Tuning solo su TRAIN (test è sacro).")
    print("2) Controlla over/underfitting con Acc/F1 su train vs test.")
    print("3) Regola la complessità: max_depth / min_samples_leaf / max_leaf_nodes.")
    print("4) Albero singolo = alta varianza; se serve stabilità, pensa a Random Forest.")

def print_cheatsheet():
    _print_block("ML — CHEAT-SHEET (compatto)")
    print("• ML = imparare f(input→output) dai dati. Test separato e SACRO.")
    print("• Overfit: altissimo su train, più basso su test → semplifica o più dati.")
    print("• Underfit: scarso ovunque → modello più ricco / nuove feature.")
    print("• Per alberi: controlla max_depth e leaf size; regole leggibili.")
    _print_line()

# --------------------- Utility “parla umano” ---------------------

def print_tree_rules(clf, feature_names, class_names):
    print("\nRegole dell’albero:")
    txt = export_text(clf, feature_names=list(feature_names))
    print(txt)

def print_misclassifications(X_te, y_te, y_pred, feature_names, target_names, max_items=10):
    err_idx = np.where(y_te != y_pred)[0]
    if len(err_idx) == 0:
        print("\nErrori puntuali: nessuno 🎉")
        return
    print("\nErrori puntuali:")
    for k, i in enumerate(err_idx[:max_items]):
        feats = ", ".join([f"{feature_names[j].replace(' (cm)','').replace('(cm)','')[:12]}={X_te[i, j]:.2f}" for j in range(X_te.shape[1])])
        print(f"- vero={target_names[y_te[i]]}, predetto={target_names[y_pred[i]]}, features=[{feats}]")
    if len(err_idx) > max_items:
        print(f"... (altri {len(err_idx) - max_items} errori)")

def feature_importance_report(clf, feature_names, top=10):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\nImportanza feature:")
    for i in order[:top]:
        print(f"- {feature_names[i]}: {importances[i]:.3f}")

# --------------------- Main ---------------------

def main(seed=0, max_depth=None, tune=False, print_cheat=False):
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # Se richiesto, tuning onesto su TRAIN per scegliere max_depth
    chosen_depth = max_depth
    best_params = {"max_depth": chosen_depth}
    if tune:
        print("\n[Tuning] Valuto le profondità solo sul TRAIN (k-fold=5):")
        depths = [1, 2, 3, 4, 5, None]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, random_state=seed)
            accs = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="accuracy")
            print(f"  depth={str(d):>4}  cv-mean={accs.mean():.3f}  cv-std={accs.std():.3f}")
            scores.append((accs.mean(), d))
        # scegli la profondità con miglior media (nessuna 1-SE per semplicità qui)
        chosen_depth = max(scores, key=lambda t: t[0])[1]
        print(f"[Tuning] Scelgo depth={chosen_depth} (miglior media CV).")
        best_params = {"max_depth": chosen_depth}

    # Fit finale su TRAIN
    clf = DecisionTreeClassifier(max_depth=chosen_depth, random_state=seed)
    clf.fit(Xtr, ytr)

    # Metriche train/test
    yhat_tr = clf.predict(Xtr)
    yhat_te = clf.predict(Xte)

    acc_tr = accuracy_score(ytr, yhat_tr)
    f1_tr  = f1_score(ytr, yhat_tr, average="macro")
    acc_te = accuracy_score(yte, yhat_te)
    f1_te  = f1_score(yte, yhat_te, average="macro")

    print(f"\nAccuratezza train: {acc_tr:.3f}")
    print(f"Accuratezza test:  {acc_te:.3f}")

    print_tree_rules(clf, feature_names, target_names)

    cm = confusion_matrix(yte, yhat_te)
    print("\nConfusion matrix (righe=vero, colonne=predetto):")
    _print_cm_table(cm, list(target_names))

    print("\nReport classi (test):")
    print(classification_report(yte, yhat_te, target_names=target_names, digits=3))

    print_misclassifications(Xte, yte, yhat_te, feature_names, target_names, max_items=10)
    feature_importance_report(clf, feature_names, top=10)

    # Sweep profondità (solo info: train vs test)
    print("\nSweep profondità (train vs test):")
    for d in [1, 2, 3, 4, 5, None]:
        c = DecisionTreeClassifier(max_depth=d, random_state=seed).fit(Xtr, ytr)
        a_tr = accuracy_score(ytr, c.predict(Xtr))
        a_te = accuracy_score(yte, c.predict(Xte))
        tag  = f"{d}" if d is not None else "None"
        print(f"depth={tag:>5}  train={a_tr:.3f}  test={a_te:.3f}")

    # CV accuracy/F1 (sul TRAIN, per riassunto)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_acc = cross_val_score(DecisionTreeClassifier(max_depth=chosen_depth, random_state=seed),
                             Xtr, ytr, cv=cv, scoring="accuracy")
    cv_f1  = cross_val_score(DecisionTreeClassifier(max_depth=chosen_depth, random_state=seed),
                             Xtr, ytr, cv=cv, scoring="f1_macro")

    # Appunti di fine-run
    print_run_summary_multiclass(
        model_label=f"Decision Tree (max_depth={chosen_depth})",
        seed=seed,
        train_acc=acc_tr,
        train_f1=f1_tr,
        test_acc=acc_te,
        test_f1=f1_te,
        cm_test=cm,
        labels=list(target_names),
        cv_acc=(cv_acc.mean(), cv_acc.std()),
        cv_f1=(cv_f1.mean(),   cv_f1.std()),
        best_params=best_params
    )

    if print_cheat:
        print_cheatsheet()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision Tree su Iris — chiaro, spiegabile, con tuning onesto e appunti finali")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=lambda s: None if s.lower()=="none" else int(s), default=None)
    ap.add_argument("--tune", action="store_true", help="Seleziona max_depth con CV (solo TRAIN)")
    ap.add_argument("--print-cheatsheet", action="store_true", help="Stampa anche il mini cheat-sheet a fine run")
    args = ap.parse_args()

    main(seed=args.seed, max_depth=args.max_depth, tune=args.tune, print_cheat=args.print_cheatsheet)
