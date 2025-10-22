# ML — Lessons Learned (versione “terra-terra”, aggiornata)
Di seguito il riassunto che useremmo davvero in lab: spiegato semplice, con esempi concreti e zero fuffa.

---

## 0) Mini-glossario (3 parole che tornano sempre)

* **Parametri**: numeri che il modello *impara* (es. pesi della logistica).
* **Iperparametri**: manopole che *scegli tu* (es. `max_depth` di un albero, `C` della logistica).
* **Fit**: l’atto di far apprendere il modello dal **train**.

---

## 1) Cosa stiamo facendo

* **ML = imparare una funzione `input → output` dai dati.**
  Non scrivi la formula a mano: la stima il modello minimizzando un errore.
* **Metodo onesto:** separa sempre i dati in **train** (si impara) e **test** (si misura).
  **Il test è sacro**: non lo usi per scegliere nulla.

---

## 2) Fit buono vs fit cattivo

* **Overfitting:** vola su **train**, crolla su **test** → ha imparato anche il rumore.
  *Sintomi:* modello troppo complesso, regole iper-specifiche.
* **Underfitting:** scarso sia su train sia su test → modello troppo semplice o features povere.
* **Che fare:** confronta **train vs test**, regola la complessità (es. `max_depth`, `C`), aggiungi features/dati migliori.

---

## 3) Decision Tree (albero)

* **Pro:** regole leggibili (*SE … ALLORA …*), spiegazioni facili.
* **Manopole principali:** `max_depth`, `min_samples_leaf`, `max_leaf_nodes`.
* **Nota:** un singolo albero è **instabile** (alta varianza). La **Random Forest** media molti alberi e stabilizza.

---

## 4) Come scegliere gli iperparametri senza barare

* **k-fold cross-validation** sul **solo train** (es. 5-fold).
  Media e deviazione ti dicono qualità e incertezza.
* **Regola “1-SE”:** se più settaggi sono quasi pari, scegli **il più semplice**.
* **Dataset piccolo?** Fai **CV ripetuta** (es. 5×5-fold) per avere una media più stabile.

---

## 5) Classi sbilanciate: l’accuracy mente

* Se il 90% è classe 0, dire sempre “0” dà **accuracy 0.90**… ma non serve a nulla.
* Guarda **Precision** (pochi falsi allarmi), **Recall** (pochi positivi persi), **F1** (equilibrio tra i due).
* **Confusion matrix** = tavolo della verità: TN, FP, FN, TP.

**Esempio lampo (100 casi):**
Predici 40 positivi → 30 giusti (TP) e 10 falsi allarmi (FP).
Ci sono 35 positivi reali: ne hai persi 5 (FN).

* Precision = 30/(30+10)=0.75
* Recall = 30/(30+5)=0.86
* F1 ≈ 0.80

---

## 6) F1 e ROC-AUC in due righe

* **F1:** numero unico che sale solo se **precision** e **recall** sono **entrambi** alti.
  `F1 = 2 · (Prec·Rec) / (Prec + Rec)`
* **ROC-AUC:** qualità del **ranking** delle probabilità su **tutte** le soglie.
  0.5 = a caso, 1.0 = perfetto. *Non dipende* dalla soglia 0.5.

> Nota: con sbilanciamenti **molto** forti, guarda anche la **PR-AUC** (area Precision-Recall).

---

## 7) La soglia: 0.5 non è legge

Cambiare soglia scambia **FP ↔ FN**. Tre modi pratici per sceglierla:

1. **Auto-threshold su validation** (dal **train**): massimizza F1 (o Youden `TPR−FPR`).
2. **A costo:** scegli `soglia` che minimizza `c_fp·FP + c_fn·FN` (decidi tu quanto pesa un FN vs FP).
3. **Sweep tabellare:** stampa soglia → (FP, FN, Precision, Recall, F1) e guardala.

**Regola pratica:**

* Se **perdere un positivo fa male** → soglia **più bassa** (recall su, più FP).
* Se **odi i falsi allarmi** → soglia **più alta** (precision su, più FN).

---

## 8) Scaling: quando sì e quando no

* **Serve** a modelli basati su distanze/geometria: **Logistica**, **SVM**, **k-NN**, **PCA**.
* **Non serve in genere** per **Alberi/Random Forest** (si basano su soglie, non su distanze).

---

## 9) Random Forest: perché funziona e dove punge

* **Pro:** cattura **non-linearità** e **interazioni**, robusta al rumore, buona “out-of-the-box”.
* **Contro tipico:** le **probabilità** possono essere **poco calibrate** → 0.5 spesso non è la soglia giusta.
* **Rimedi:**

  * scegli bene la **soglia** (auto/costo/sweep),
  * valuta `class_weight='balanced'` se la positiva è rara,
  * **calibra** le probabilità (**sigmoid/Platt** o **isotonic**) se ti servono score affidabili.

---

## 10) “Importanza” delle feature: evitare abbagli

* **Impurity importance** (`feature_importances_`): rapida, ma **favorisce** variabili con molte soglie e soffre con **feature molto correlate** (le “sorelle” si dividono il merito).
* **Permutation importance** (su **TEST** e con una metrica, es. ROC-AUC): misuri *quanto peggiora* il modello quando rompi una feature → spesso più onesta.
* **Ablation test:** rimuovi la top-1/top-k e rimisura. Se l’AUC non scende, c’è **ridondanza** (più feature dicono la stessa cosa).

---

## 11) Leakage: barare senza volerlo

* **Cos’è:** usare info del **test** (o del fold di validazione) per calcolare imputazioni, scaling, encoding, selezione feature…
  Risultato: metriche gonfiate.
* **Regola d’oro:** tutte le trasformazioni si **fittano solo sul train**.
* **Strumenti che ti proteggono:** **Pipeline** (e **ColumnTransformer** per numeriche+categoriche) + **GridSearchCV** *sulla pipeline*.
* **Spia rossa:** CV troppo bella dopo aver preprocessato *prima* della CV.

---

## 12) Workflow che non ti tradisce

1. **Split** train/test (il test in cassaforte).
2. Sul **train**: costruisci **Pipeline** (preprocess + modello).
3. **CV** (anche ripetuta) per **tuning** iperparametri (regola 1-SE se pari).
4. **Soglia**: scegli su validation/OOF o per **costo**.
5. **Fit finale** sul train completo con pipeline + iperparametri + soglia scelti.
6. **Test una sola volta**: F1, ROC-AUC, confusion; se serve, **calibrazione** (Brier, reliability bins).
7. **Spiega**: importanze (impurity + permutation) e, se serve, ablation.

---

## 13) Regole veloci da campo

* **Mai** usare il test per scegliere iperparametri o soglia.
* Dataset piccolo → **CV ripetuta** + regola **1-SE**.
* **Sbilanciamento:** valuta **F1** e/o **Recall** (se i FN costano), usa **ROC-AUC** per confrontare modelli e **PR-AUC** se lo sbilanciamento è estremo.
* **RF:** non fissarti su 0.5; **scegli soglia** e valuta **calibrazione**.
* **Logistica:** con scaling, spesso imbattibile su confini quasi lineari; **probabilità molto affidabili**.

---

## 14) Debug veloce (checklist)

1. **Baseline**: quanto fa un dummy che predice la classe più frequente?
2. **Leakage**: sto facendo imputazione/scaling *fuori* pipeline?
3. **Metriche giuste**: sto guardando F1/Recall oltre all’accuracy?
4. **Soglia**: ho provato auto-threshold o a costo?
5. **Varianza**: risultati stabili cambiando **seed**? (se no, CV ripetuta)
6. **Ridondanza**: feature molto correlate? Usa permutation + ablation.

---
