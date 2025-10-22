# Script inclusi

Questa pagina elenca gli script del laboratorio con **cosa fanno**, **opzioni principali** e **comandi tipici**.  
Gli script stampano report “parlanti” (metriche, confusion, analisi soglie ecc.). Alcuni includono anche gli **appunti finali** con `--print-cheatsheet`.

---

## `scripts/iris.py`
**Cosa:** Decision Tree sull’IRIS (3 classi).  
**Perché:** capire over/underfitting con regole **leggibili** (SE… ALLORA…).

**Opzioni principali**
- `--max-depth INT` … limita la profondità dell’albero.
- `--tune` … sceglie in modo onesto la profondità via k-fold (solo sul train).
- `--seed INT` … fissa lo split.
- `--print-cheatsheet` … stampa appunti finali (lesson learned essenziali).

**Esempi**
```bash
python scripts/iris.py
python scripts/iris.py --max-depth 3
python scripts/iris.py --tune --seed 13 --print-cheatsheet
````

**Cosa vedrai**

* Accuratezza **train/test**
* **Regole** dell’albero in testo
* **Confusion matrix** multiclasse
* **Feature importance**
* Mini **sweep** di profondità con train vs test
* (se `--tune`) scelta della profondità via CV

---

## `scripts/imbalance.py`

**Cosa:** Classificazione **sbilanciata** (Breast Cancer) con **Logistic Regression** + **scaling**.
**Perché:** mostrare che l’**accuracy** può ingannare; serve guardare **Precision/Recall/F1** e **scegliere la soglia**.

**Opzioni principali**

* `--C FLOAT` … forza del modello logit (C↑ = meno regolarizzazione).
* `--threshold FLOAT` … soglia decisione (default 0.5).
* `--auto-threshold` … sceglie la soglia da **validation** (solo train).
* `--metric {f1,youden}` … criterio per l’auto-soglia.
* `--seed INT`
* `--print-cheatsheet` … appunti finali.

**Esempi**

```bash
python scripts/imbalance.py
python scripts/imbalance.py --auto-threshold --metric f1 --print-cheatsheet
python scripts/imbalance.py --C 0.5 --seed 13
```

**Cosa vedrai**

* **Baseline** (classe maggioritaria)
* **Distribuzione** degli score (quantili per negativi/positivi)
* **Sweep soglia** (tabella soglia → FP/FN/Precision/Recall/F1)
* **Soglia a costo** (es. FN 5× FP)
* Report completo: Accuracy, Precision, Recall, **F1**, **ROC-AUC**, **Confusion matrix**
* **CV ROC-AUC** sul train

---

## `scripts/forest_vs_logit.py`

**Cosa:** Confronto **Random Forest** vs **Logistic** su Breast Cancer.
Include **calibrazione** delle probabilità RF e **scelta soglia** separata per i due modelli.

**Opzioni principali**

* Soglia:

  * `--threshold FLOAT`, `--auto-threshold`, `--metric {f1,youden}`
* Random Forest:

  * `--rf-n INT`, `--rf-depth {INT|None}`, `--rf-mf {sqrt,log2}`
  * `--rf-class-weight {balanced|None}`
  * `--rf-tune` … grid semplice via CV (sul train)
  * `--rf-calibrate {isotonic,sigmoid}` … **CalibratedClassifierCV** su train
* Logistica:

  * `--C FLOAT`
* Generali:

  * `--seed INT`, `--print-cheatsheet`

**Esempi**

```bash
python scripts/forest_vs_logit.py --auto-threshold
python scripts/forest_vs_logit.py --rf-calibrate isotonic --auto-threshold --print-cheatsheet
python scripts/forest_vs_logit.py --rf-tune --seed 13
```

**Cosa vedrai**

* **Baseline** maggioritaria
* Analisi **punteggi RF** (quantili + **sweep soglia**)
* Report logit e RF (Accuracy, **F1**, **ROC-AUC**, Confusion)
* **Importanze RF** (da un clone non calibrato, se necessario)
* **CV ROC-AUC** sul train per entrambi

---

## `scripts/importance_demo.py`

**Cosa:** Importanza delle feature con Random Forest: **impurity vs permutation**, **correlazioni** e **ablation** (drop top-k).
**Perché:** non farsi ingannare dalle sole `feature_importances_`.

**Opzioni principali**

* `--seed INT`
* `--print-cheatsheet`

**Esempi**

```bash
python scripts/importance_demo.py
python scripts/importance_demo.py --seed 13 --print-cheatsheet
```

**Cosa vedrai**

* **AUC** e **Accuracy** base della RF
* **Impurity importance** (top-k)
* **Permutation importance** su **TEST** (metrica ROC-AUC)
* **Coppie molto correlate** (|corr| ≥ 0.9) — warning interpretativo
* **Ablation**: AUC base vs AUC senza top-1 / top-3

---

## `scripts/gridsearch_mixed.py`

**Cosa:** Dati **misti** (numeriche + categoriche, con missing) con **Pipeline/ColumnTransformer** e **GridSearchCV** (no leakage).
Sceglie la **soglia onesta** da **OOF** (predizioni out-of-fold). Supporta **Logit** e **RF**.

**Opzioni principali** (le più usate)

* Modello: `--model {logit,rf}`
* Soglia: `--auto-threshold`, `--thr-mode {f1,youden,cost}`, `--cost-fp FLOAT`, `--cost-fn FLOAT`
* Dati: `--missing FLOAT` (quota di valori mancanti generati), `--seed INT`
* (Altre opzioni sono mostrate da `--help`)

**Esempi**

```bash
# Logit con soglia OOF (F1)
python scripts/gridsearch_mixed.py --model logit --auto-threshold

# RF con soglia a costo (FN 10× FP)
python scripts/gridsearch_mixed.py --model rf --auto-threshold --thr-mode cost --cost-fn 10
```

**Cosa vedrai**

* **Top configurazioni** da GridSearch (per F1 CV) + AUC
* **Soglia OOF** scelta (criterio F1/Youden/costo)
* Report **TRAIN** e **TEST** alla soglia scelta
* Per **Logit**: **Top coefficienti** (segnano +/− rischio)
  Per **RF**: **Top importanze**

---

## `scripts/pipeline_leakage.py`

**Cosa:** Dimostrazione **leakage**: confronto “sbagliato” (trasformazioni prima dello split) vs “corretto” (Pipeline solo sul train).

**Esempi**

```bash
python scripts/pipeline_leakage.py
```

**Cosa vedrai**

* Confronto affiancato: **Accuracy** e **F1**
* **CV (train)** con media ± std per la versione corretta

---

## Convenzioni di output (rapido promemoria)

* **Metriche chiave:** Accuracy, **Precision**, **Recall**, **F1**, **ROC-AUC**
* **Confusion matrix** con conteggi **TN/FP/FN/TP** “in chiaro”
* **Soglia:** tabella “sweep soglia” e/o scelta **auto** (validation/OOF) o **a costo**
* **CV:** riportata come **media ± deviazione standard** (k-fold sul **train**)

> **Tip:** se perdi **pochi positivi** è grave, abbassa la soglia; se odi i **falsi allarmi**, alzala. Con RF valuta **calibrazione** (`isotonic`/`sigmoid`) quando servono probabilità affidabili.
