# ML-Lab — Laboratorio pratico per principianti al Machine Learning (Python + scikit-learn)

**Obiettivo:** capire davvero **cosa** sono e **come** funzionano i modelli.  
Gli script sono “parlanti”: stampano metriche in chiaro, analisi delle soglie, importanze e **appunti di fine-run** (mini-riassunti automatici).

In più, il repo contiene **Turbo-K**, un laboratorio didattico per valutare funzioni di bucketizzazione su IP/chiavi 32-bit con test statistici (χ²/DoF), ricerca di parametri (“oli”) e confronti riproducibili.

---

## Documentazione

Il sito della documentazione è generato con **MkDocs** (tema *Ivory*) e pubblicato su GitHub Pages.

- Build locale:  
  ```bash
  pip install mkdocs mkdocs-ivory && mkdocs serve
````

* Deploy automatico via **GitHub Actions** (workflow `docs.yml`).

---

## Contenuti

### Machine Learning (scikit-learn)

* **`scripts/iris.py`** — Decision Tree sull’IRIS (multiclasse)
  Regole leggibili, tuning onesto (`--tune`), report completo, appunti finali.

* **`scripts/imbalance.py`** — Classificazione sbilanciata (Breast Cancer) con **Logistic Regression**
  Scaling, scelta soglia (fissa/automatica/a costo), sweep soglie, appunti finali.

* **`scripts/forest_vs_logit.py`** — **Random Forest vs Logistic** su Breast Cancer
  Soglia automatica, **calibrazione** RF (`--rf-calibrate isotonic|sigmoid`), importanze, CV.

* **`scripts/importance_demo.py`** — Feature importance con RF
  **Impurity vs Permutation**, coppie molto correlate, **ablation** (drop top-k), appunti finali.

* **`scripts/gridsearch_mixed.py`** — **Pipeline + ColumnTransformer + GridSearchCV** su dati misti (num + cat, sintetici)
  Nessun leakage, soglia **OOF onesta** (F1/Youden/costo), importanze/coeff, appunti finali.

> Ogni script include il flag `--print-cheatsheet` per stampare un mini promemoria a fine esecuzione.

### Turbo-K (hash/bucketization su 32-bit)

* **`scripts/turbo_k_eval.py`** — Genera chiavi (uniformi/CIDR/file), applica `y=(a*x+b) mod 2^32`, assegna a bucket (MSB o MOD) e valuta:

  * **χ²/DoF** (verde ~ 0.9–1.2),
  * **ricerca parametri**: `--search-a N`, `--search-b N` (con filtri tipo `--min-popcount-b`),
  * **confronto oli**: `--compare a,b` *N* volte o `--presets presets/oils.yaml`,
  * **file-mode normalizzato** per sorgenti con **duplicati** (stima dell’effetto “s=N/U”),
  * report leggibile + consigli automatici.

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
```

---

## Esecuzione rapida (snippet)

```bash
# Albero su IRIS con tuning + appunti
python scripts/iris.py --tune --print-cheatsheet

# Sbilanciamento: logistica con soglia auto (F1) + appunti
python scripts/imbalance.py --auto-threshold --metric f1 --print-cheatsheet

# RF vs Logit, calibrazione + soglia auto
python scripts/forest_vs_logit.py --rf-calibrate isotonic --auto-threshold --print-cheatsheet

# Importanze RF: impurity vs permutation + correlazioni + ablation
python scripts/importance_demo.py --print-cheatsheet

# Dati misti con GridSearch (no leakage), soglia a costo (FN 10x FP)
python scripts/gridsearch_mixed.py --model rf --auto-threshold --thr-mode cost --cost-fn 10 --print-cheatsheet
```

### Turbo-K: comandi tipici

```bash
# Uniforme, MSB K=12 (baseline)
python scripts/turbo_k_eval.py --source uniform --mode msb --K 12 --N 200000

# Cerca un 'a' migliore (b fisso)
python scripts/turbo_k_eval.py --source uniform --mode msb --K 12 --N 200000 --search-a 256

# CIDR mix + cerca 'b' (a fisso), con filtro di popcount su b
python scripts/turbo_k_eval.py --source cidr --cidr 10.0.0.0/8 --cidr 192.168.0.0/16 \
  --mode msb --K 12 --N 300000 --a 0x7649D8CF --search-b 256 --min-popcount-b 6

# Confronto oli da presets
python scripts/turbo_k_eval.py --presets presets/oils.yaml \
  --source uniform --mode msb --K 12 --N 200000

# File di IP (una chiave per riga: IPv4 puntato o intero dec/hex)
python scripts/turbo_k_eval.py --source file --ip-file ./data/ips.txt --mode msb --K 12
```

> Scelta pratica di **K**: mantieni **E = N/B ≥ ~50** (stabilità statistica).

---

## Come leggere gli output “parlanti”

**Confusion matrix (righe=vero, colonne=predetto)** + **TN/FP/FN/TP** in chiaro.

**Metriche chiave (ML):**

* **Precision**: su quanti allarmi avevi ragione.
* **Recall**: quanti positivi hai preso.
* **F1**: sale solo se **precision e recall** sono **entrambi** alti.
* **ROC-AUC**: qualità del **ranking** su **tutte** le soglie (0.5 non c’entra).

**Soglia (`thr`)**: spostarla scambia **FP ↔ FN**.

* `--auto-threshold`: scelta su validation/OOF (solo train) max **F1** o **Youden** (TPR−FPR).
* **Soglia a costo**: minimizza `c_fp·FP + c_fn·FN`.

**Calibrazione (RF)**: `--rf-calibrate isotonic|sigmoid` per rendere sensate le probabilità (utile quando 0.5 non è una buona soglia).

**Importanze:**

* `feature_importances_` (**impurity**): veloce ma distorta con variabili molto granulari e **forte correlazione**.
* **Permutation importance** (su **TEST**, a metrica scelta): perdita reale → più onesta.
* **Ablation**: togli top-k e guarda l’AUC → capisci ridondanze.

**Turbo-K (distribuzioni su bucket):**

* **χ²/DoF ≈ 1** → in media uniforme. Fascia verde ~ **0.9–1.2**.
* Report include **std/E**, **z_max** (massimo scostamento in σ), bucket top/bottom, consigli su **K**.
* Con **file** ricchi di **duplicati**, guarda il blocco **“File-mode normalizzato”**: separa l’effetto del mixing dall’effetto del rapporto `s=N/U`.

---

## Datasets

* **IRIS** (multiclasse 3 classi) — per la spiegabilità delle regole dell’albero.
* **Breast Cancer** (binario, leggermente sbilanciato) — per soglie, calibrazione, importanze.
* **Dataset sintetico misto** (in `gridsearch_mixed.py`) — numeriche + categoriche, con missing, per **Pipeline/ColumnTransformer** e **GridSearch** senza leakage.
* **Turbo-K** genera chiavi sintetiche uniformi o da **CIDR**; puoi anche usare **file** tuoi (`./data/*.txt`).

*(Tutti i dataset “toy” arrivano da `sklearn.datasets` o sono generati al volo.)*

---

## Metodo: cosa ci siamo imposti (e perché)

1. **Train/Test**: il **test è sacro** (mai usarlo per scegliere iperparametri/soglie).
2. **Pipeline ovunque**: imputazione, scaling, encoding **dentro** la pipeline → niente **leakage**.
3. **Tuning onesto**: `StratifiedKFold` sul **train**; su dataset piccoli, **CV ripetuta**.
4. **Scelta soglia**: su validation/OOF o per **costo**; 0.5 non è una legge divina.
5. **Metriche giuste**: con sbilanciamento, guarda **F1/Recall** (e **ROC-AUC** per confronti).
6. **Spiegabilità**: regole (albero), coef (logit), impurity+permutation (RF), **ablation**.
7. **Stabilità**: tieni d’occhio **media ± dev.std** in CV; usa la **regola 1-SE** a parità di resa.

---

## Gli script, in breve

### `scripts/iris.py`

**Perché**: capire over/underfitting con un modello leggibile.
**Note**: `max_depth` controlla la complessità; tuning CV; **report multiclasse**.

### `scripts/imbalance.py`

**Perché**: far vedere che **accuracy** può mentire con classi sbilanciate.
**Note**: scaling (serve alla logistica), **sweep soglie**, **soglia a costo**, **auto-threshold**.

### `scripts/forest_vs_logit.py`

**Perché**: confrontare un **modello lineare** ben calibrato con una **RF** non lineare.
**Note**: **calibrazione** RF, soglie separate per modello, importanze RF da clone **fit** quando calibrata, **CV AUC**.

### `scripts/importance_demo.py`

**Perché**: imparare a **non farsi fregare** dalle importanze.
**Note**: impurity vs permutation, **coppie correlate** (|corr|≥0.9), **ablation**.

### `scripts/gridsearch_mixed.py`

**Perché**: workflow reale con **numeriche + categoriche** (missing compresi), **GridSearch** senza leakage.
**Note**: soglia “onesta” da **OOF**, modalità **F1/Youden/costo**, estrazione coef/importanze post-preprocessing.

### `scripts/turbo_k_eval.py`

**Perché**: validare una funzione di partizionamento su chiavi 32-bit (es. IPv4) in scenari uniformi/CIDR/reali.
**Note**:

* `--mode msb --K K` (B=2^K) o `--mode mod --M M` (B=M),
* ricerca `--search-a N`, `--search-b N` (+ filtri),
* confronto `--compare a,b` e `--presets presets/oils.yaml`,
* diagnostica completa + normalizzazione per duplicati.

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

## Makefile: scorciatoie utili

```bash
# setup ambiente
make setup

# script ML
make run-iris
make run-imbalance
make run-forest
make run-importance
make run-grid

# Turbo-K (suite e confronto oli)
make run-turbok
make run-turbok-oils

# docs
make docs-serve
make docs-build
make docs-deploy
```

I log della suite Turbo-K vengono salvati in `reports/turbo_k/<timestamp>/` con un **riepilogo finale** (χ²/DoF e best da search/compare).

---

## Riproducibilità

* Usa `--seed` per fissare gli split e la generazione.
* Dataset piccoli ⇒ la varianza tra run può essere visibile: confronta sempre **media ± std** in CV.
* Per Turbo-K, scegli **K** in modo che **E=N/B** non sia troppo basso (regola empirica **≥ 50**).

---

## Estensioni possibili

* **Curva Precision-Recall** e **PR-AUC** (forte sbilanciamento).
* **Brier score** e **calibration curve** (probabilità).
* **Model card**: decisioni di soglia, costi, popolazione, limiti e rischi.
* Nuovi modelli: **SVM**, **Gradient Boosting/XGBoost**, **Logistic** con penalità **elastic-net**.
* Turbo-K: salvataggio best-oil in YAML e check automatici in CI (soglie su χ²/DoF).

---

## Licenza / Autori

Uso libero a scopo didattico.
Autore del laboratorio: **Giancarlo** (VM Ubuntu + VSCode)

[![CI](https://github.com/gcomneno/ML-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/gcomneno/ML-Lab/actions/workflows/ci.yml)
[![docs](https://github.com/gcomneno/ML-Lab/actions/workflows/docs.yml/badge.svg)](https://github.com/gcomneno/ML-Lab/actions/workflows/docs.yml)

```

::contentReference[oaicite:0]{index=0}
