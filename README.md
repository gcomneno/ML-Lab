# ML-Lab — Laboratorio pratico per principianti al Machine Learning (Python + scikit-learn)

## Documentazione
Il sito della documentazione è generato con MkDocs e pubblicato su GitHub Pages.
- Build locale: `pip install mkdocs mkdocs-material && mkdocs serve`
- Deploy automatico via GitHub Actions (workflow `docs.yml`).

[![CI](https://github.com/gcomneno/ML-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/gcomneno/ML-Lab/actions/workflows/ci.yml)
[![docs](https://github.com/gcomneno/ML-Lab/actions/workflows/docs.yml/badge.svg)](https://github.com/gcomneno/ML-Lab/actions/workflows/docs.yml)

# ML-Lab — Allenamento pratico al Machine Learning (Python + scikit-learn)

**Obiettivo:** capire davvero **cosa** sono e **come** funzionano i modelli.

Qui trovi script “parlanti” che stampano metriche in chiaro, analisi delle soglie, importanze delle feature e **appunti di fine-run** (mini-riassunti automatici).

---

## Contenuti

- **`iris.py`** — Decision Tree sull’IRIS (multiclasse)
  - Regole leggibili, tuning onesto (`--tune`), report completo, appunti finali.
- **`imbalance.py`** — Classificazione sbilanciata (Breast Cancer) con **Logistic Regression**
  - Scaling, scelta soglia (fissa/automatica/a costo), sweep soglie, appunti finali.
- **`forest_vs_logit.py`** — **Random Forest vs Logistic** su Breast Cancer
  - Soglia automatica, **calibrazione** RF (`--rf-calibrate isotonic|sigmoid`), importanze, CV.
- **`importance_demo.py`** — Feature importance con RF
  - **Impurity vs Permutation**, coppie molto correlate, **ablation** (drop top-k), appunti finali.
- **`gridsearch_mixed.py`** — **Pipeline + ColumnTransformer + GridSearchCV** su dati misti (num + cat, sintetici)
  - Nessun leakage, soglia **OOF onesta** (F1/Youden/costo), importanze/coeff, appunti finali.
- **`pipeline_leakage.py`** — Mini dimostrazione del **data leakage** (prima/dopo pipeline).

Ogni script include il flag `--print-cheatsheet` per stampare un mini promemoria a fine esecuzione.

---

## Installazione

Prerequisiti: Python **3.8+** (testato con 3.8).

```bash
# (solo se serve) assicurati di avere il modulo venv
sudo apt-get update && sudo apt-get install -y python3-venv

# 1) crea un ambiente isolato nella cartella
python3 -m venv .venv

# 2) attivalo
source .venv/bin/activate

# 3) aggiorna pip e installa le librerie
pip install --upgrade pip
pip install -r requirements.txt

# non hai il file requirements?
pip install scikit-learn numpy
````

---

## Esecuzione rapida (snippet)

```bash
# Albero su IRIS con tuning + appunti
python iris.py --tune --print-cheatsheet

# Sbilanciamento: logistica con soglia auto (F1) + appunti
python imbalance.py --auto-threshold --metric f1 --print-cheatsheet

# RF vs Logit, calibrazione + soglia auto
python forest_vs_logit.py --rf-calibrate isotonic --auto-threshold --print-cheatsheet

# Importanze RF: impurity vs permutation + correlazioni + ablation
python importance_demo.py --print-cheatsheet

# Dati misti con GridSearch (no leakage), soglia a costo (FN 10x FP)
python gridsearch_mixed.py --model rf --auto-threshold --thr-mode cost --cost-fn 10 --print-cheatsheet
```

---

## Come leggere gli output “parlanti”

* **Confusion matrix (righe=vero, colonne=predetto)** + **TN/FP/FN/TP** in chiaro.
* **Metriche chiave:**

  * **Precision**: su quanti allarmi avevi ragione.
  * **Recall**: quanti positivi hai preso.
  * **F1**: sale solo se **precision e recall** sono **entrambi** alti.
  * **ROC-AUC**: qualità del **ranking** su **tutte** le soglie (0.5 non c’entra).
* **Soglia (`thr`)**: spostarla scambia **FP ↔ FN**.

  * `--auto-threshold`: scelta su validation/OOF (solo train) max F1 o **Youden** (TPR−FPR).
  * **Soglia a costo**: minimizza `c_fp·FP + c_fn·FN`.
* **Calibrazione (RF)**: `--rf-calibrate isotonic|sigmoid` per rendere sensate le probabilità (utile quando 0.5 non è una buona soglia).
* **Importanze**:

  * `feature_importances_` (**impurity**): veloce ma distorta con variabili molto granulari e **forte correlazione**.
  * **Permutation importance** (su **TEST**, a metrica scelta): misura la perdita reale → più onesta.
  * **Ablation**: togli top-k e vedi se l’AUC crolla → capisci ridondanze.

---

## Datasets

* **IRIS** (multiclasse 3 classi) — per la spiegabilità delle regole dell’albero.
* **Breast Cancer** (binario, leggermente sbilanciato) — per soglie, calibrazione, importanze.
* **Dataset sintetico misto** (in `gridsearch_mixed.py`) — numeriche + categoriche, con missing, per mostrare **Pipeline/ColumnTransformer** e **GridSearch** senza leakage.

*(Tutti i dataset “toy” arrivano da `sklearn.datasets` o sono generati al volo.)*

---

## Metodo: cosa ci siamo imposti (e perché)

1. **Train/Test**: il **test è sacro** (mai usarlo per scegliere iperparametri/soglie).
2. **Pipeline ovunque**: imputazione, scaling, encoding **dentro** la pipeline → niente **leakage**.
3. **Tuning onesto**: `StratifiedKFold` sul **train**; se dataset piccolo, **CV ripetuta** consigliata.
4. **Scelta soglia**: su validation/OOF o per **costo**; 0.5 non è una legge divina.
5. **Metriche giuste**: con sbilanciamento, guarda **F1/Recall** (e **ROC-AUC** per confronti).
6. **Spiegabilità**: regole (albero), coef (logit), impurity+permutation (RF), **ablation**.
7. **Stabilità**: tieni d’occhio **media ± dev.std** in CV; applica se vuoi la **regola 1-SE** (scegli la più semplice entro l’incertezza).

---

## Gli script, in breve

### `iris.py`

* **Perché**: capire over/underfitting con un modello leggibile.
* **Cose da notare**: `max_depth` controlla la complessità; tuning CV; **report multiclasse**.

### `imbalance.py`

* **Perché**: far vedere che **accuracy** può mentire con classi sbilanciate.
* **Cose da notare**: scaling (serve alla logistica), **sweep soglie**, **soglia a costo**, **auto-threshold**.

### `forest_vs_logit.py`

* **Perché**: confrontare un **modello lineare** ben calibrato con una **RF** non lineare ma con probabilità spesso storte.
* **Cose da notare**: **calibrazione** RF, scelta soglie separata, importanze RF da clone **fit** anche quando calibrata, **CV AUC**.

### `importance_demo.py`

* **Perché**: imparare a **non farsi fregare** dalle importanze.
* **Cose da notare**: impurity vs permutation, **coppie correlate** (|corr|≥0.9), **ablation**.

### `gridsearch_mixed.py`

* **Perché**: workflow reale con **numeriche + categoriche** (missing compresi), **GridSearch** senza leakage.
* **Cose da notare**: soglia “onesta” da **OOF**, modalità **F1/Youden/costo**, estrazione coef/importanze post-preprocessing.

---

## Lessons Learned (indice)

* **Cos’è il ML**: imparare f(input→output) dai dati; **test separato** e sacro.
* **Over/Underfitting**: segnali, rimedi.
* **Alberi**: regole, `max_depth`, alta varianza → RF.
* **Iperparametri**: **CV su train**, regola **1-SE** a parità di resa.
* **Classi sbilanciate**: usa **Precision/Recall/F1**, confusion matrix.
* **F1 / ROC-AUC**: cosa misurano, quando usarle.
* **Soglia**: 0.5 ≠ legge; auto-threshold/Youden/costo.
* **Scaling**: serve per logit/SVM; non per RF/Tree.
* **Random Forest**: robusta; **soglia ≠ 0.5**; **calibrazione** utile.
* **Importanze**: impurity vs permutation; occhio alle **correlazioni**; **ablation**.
* **Leakage**: tutto **fit solo sul train** con **Pipeline/ColumnTransformer**.
* **Workflow**: split → pipeline+CV → soglia → fit finale → test (una volta) → spiegazioni.

> Gli script stampano a fine run un **RUN SUMMARY** e, se richiesto, un **CHEAT-SHEET** con questi punti.

---

## Riproducibilità

* Usa `--seed` per fissare gli split.
* Dataset piccoli ⇒ la varianza tra run può essere visibile: confronta sempre **media ± std** in CV.

---

## Estensioni possibili

* **Curva Precision-Recall** e **PR-AUC** (interessante con forte sbilanciamento).
* **Brier score** e **calibration curve** (per probabilità).
* **Model card**: decisioni di soglia, costi, popolazione, limiti e rischi.
* Nuovi modelli: **SVM**, **Gradient Boosting/XGBoost**, **Logistic con penalità elastic-net**.

---

## Struttura del repo (suggerita)

```
ml-lab/
├── iris.py
├── imbalance.py
├── forest_vs_logit.py
├── importance_demo.py
├── gridsearch_mixed.py
├── pipeline_leakage.py
└── README.md
```

---

## FAQ veloci

* **Perché a volte RF fa meno AUC della logistica?**
  Se il confine è quasi lineare, la logistica con scaling e soglia ben scelta è durissima da battere.

* **Quando calibrare?**
  Quando ti servono **probabilità affidabili** (decisioni a costo, risk scoring). RF di solito beneficia di **isotonic/sigmoid**.

* **Perché scegliere la soglia su OOF/validation e non su test?**
  Per non “barare”: il **test** serve a misurare **una sola volta** ciò che hai deciso usando **solo il train**.

---

## Licenza / Autori
Uso libero a scopo didattico.
Autore del laboratorio: **Giancarlo** (VM Ubuntu + VSCode) — io ho fatto da sparring/tutor 😄
