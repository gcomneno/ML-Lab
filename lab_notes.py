# lab_notes.py — Appunti in console per fine-run (binary classification)
# Usa solo standard library. Incolla accanto ai tuoi script e importa.

from typing import Optional, Dict, Tuple

def _fmt(x, nd=3):
    return "nan" if x is None else f"{x:.{nd}f}"

def _print_line():
    print("-" * 72)

def _print_block(title: str):
    _print_line()
    print(title)
    _print_line()

def _fmt_cm(cm: Tuple[int, int, int, int]):
    tn, fp, fn, tp = cm
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
    train: Optional[Dict] = None,   # es: dict(acc=..., prec=..., rec=..., f1=..., auc=..., cm=(tn,fp,fn,tp))
    test: Optional[Dict] = None,
    cv_auc: Optional[Tuple[float, float]] = None,  # (mean, std)
    cv_f1: Optional[Tuple[float, float]] = None,   # (mean, std)
    cost_fp: Optional[float] = None,
    cost_fn: Optional[float] = None,
    calibrated: Optional[str] = None,              # 'isotonic' | 'sigmoid' | None
    class_weight: Optional[str] = None             # 'balanced' | None
):
    """
    Stampa un riassunto “da banco” con i numeri chiave del run.
    Passa solo ciò che hai: i None vengono ignorati.
    """
    _print_block(f"RUN SUMMARY — {model_label}")

    # Setup
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

    # Train
    if train:
        print("\n» Train metrics")
        print(f"  Acc={_fmt(train.get('acc'))} | Prec={_fmt(train.get('prec'))} | "
              f"Rec={_fmt(train.get('rec'))} | F1={_fmt(train.get('f1'))} | AUC={_fmt(train.get('auc'))}")
        if train.get("cm"):
            print(_fmt_cm(train["cm"]))

    # Test
    if test:
        print("\n» Test metrics")
        print(f"  Acc={_fmt(test.get('acc'))} | Prec={_fmt(test.get('prec'))} | "
              f"Rec={_fmt(test.get('rec'))} | F1={_fmt(test.get('f1'))} | AUC={_fmt(test.get('auc'))}")
        if test.get("cm"):
            print(_fmt_cm(test["cm"]))

    # CV
    if cv_auc or cv_f1:
        print("\n» Cross-Validation (train-only)")
        if cv_auc:
            mu, sd = cv_auc
            print(f"  ROC-AUC (mean±std): {_fmt(mu)} ± {_fmt(sd)}")
        if cv_f1:
            mu, sd = cv_f1
            print(f"  F1      (mean±std): {_fmt(mu)} ± {_fmt(sd)}")

    # Costi e suggerimenti di soglia
    if cost_fp is not None and cost_fn is not None and test and test.get("cm"):
        tn, fp, fn, tp = test["cm"]
        costo = cost_fp * fp + cost_fn * fn
        print("\n» Costo decisionale")
        print(f"  cost_fp={cost_fp}  cost_fn={cost_fn}  -> costo(test)={costo:.3f}")
        print("  Nota: abbassare la soglia ↓ riduce FN (recall↑) ma aumenta FP; alzarla ↑ fa il contrario.")

    # Regole flash (cheat operativo breve)
    _print_block("REGOLE FLASH")
    print("1) Mai usare il TEST per scegliere iperparametri o soglia (test è sacro).")
    print("2) Con sbilanciamento valuta F1/Recall e la confusion matrix; accuracy da sola inganna.")
    print("3) La soglia 0.5 non è legge: valuta auto-threshold (F1/Youden) o soglia a costo.")
    print("4) Scaling per Logit/SVM; non serve per Alberi/Random Forest.")
    print("5) RF: scegli bene la soglia; calibra (isotonic/sigmoid) se vuoi probabilità affidabili.")
    print("6) Importanze: guarda impurity + permutation; se togli top-1 e l’AUC non scende, c’è ridondanza.")
    print("7) Pipeline sempre: imputazione/scaling/encoding dentro la CV (no leakage).")

def print_cheatsheet():
    """Stampa un mini-cheat-sheet statico (Lessons Learned compatto)."""
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
