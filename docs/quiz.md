# ML-Lab — Quiz Q&A (versione chiara)
_Studio autonomo — domande & risposte semplici ma dettagliate._

> Consiglio: prova prima a rispondere da solo, poi confronta con la soluzione.
> Ogni risposta include il “perché” e spesso un mini-esempio pratico.

---

## 1) Che cos’è, in una frase, il Machine Learning?
**Risposta (umana):** Costruire un programma/funzione che **impara una determinata regola dai dati**: gli mostri esempi di **input** e il **risultato giusto**, e lui **regola da solo** le sue manopole per **sbagliare il meno possibile** sugli esempi, puntando ad andare bene anche su **dati nuovi**.
- **Parametri = manopole** del modello (numeri).  
- **Algoritmo di apprendimento** = come giriamo le manopole per ridurre l’**errore**.  
- **Generalizzazione** = andare bene su **test** nuovi, su dati mai visti.

---

## 2) Perché il “test è sacro”?
**Risposta:** Serve a misurare la **generalizzazione vera**. Se lo usi per scegliere iperparametri (o le soglie), **inquini** la misura (stai “imparando” anche dal test ed è scorretto).  
**Regola:** tutte le decisioni (tuning, soglia, trasformazioni) vanno prese **solo** col **train** (CV/validation). Il test si usa **una volta alla fine**.

---

## 3) Overfitting vs Underfitting: differenza pratica e rimedi
**Overfit:** ottimo su **train**, peggio su **test** → troppo complesso / ha “pappagallato”.  
**Underfit:** scarso su **train** e **test** → troppo semplice / info insufficienti.  
**Rimedi:** riduci o aumenti la complessità (es. `max_depth`, `C`), aggiungi dati/feature, regolarizzazione, semplifica il modello.

---

## 4) Decision Tree: perché è spiegabile e come ne controlli la complessità?
**Risposta:** L’albero produce regole **SE … ALLORA …** leggibili (threshold sulle feature).  
**Controllo:** `max_depth`, `min_samples_leaf`, `max_leaf_nodes`.  
**Nota:** un albero singolo ha **alta varianza** → piccole variazioni nei dati cambiano molto la struttura.

---

## 5) Perché una Random Forest riduce la varianza dell’albero singolo?
**Risposta:** Fa la **media** di molti alberi addestrati su **campioni diversi** e **sottoinsiemi di feature** (bagging + casualità). La media **stabilizza** e riduce errori “capricciosi”.

---

## 6) Cos’è un iperparametro? Esempi
**Risposta:** È un **controllo del modello** che **non** viene imparato dai dati ma **scelto** dall’utente con CV.  
**Esempi:** `max_depth` (Tree/RF), `n_estimators` (RF), `C` (Logit/SVM), `max_features` (RF), `learning_rate` (boosting).

---

## 7) k-fold Cross-Validation: cos’è e perché sul solo train?
**Risposta:** Dividi il **train** in k parti; ripeti k volte: alleni su k-1 parti, valuti sulla parte rimasta. Media e deviazione standard ti danno una **stima stabile**. Si fa solo sul **train** per non “guardare” il **test**.

---

## 8) Regola 1-SE: quando e perché?
**Risposta:** Se più modelli sono **statisticamente equivalenti** (entro 1 deviazione standard dalla migliore metrica), scegli la **versione più semplice**. È più **robusta** su dati nuovi.

---

## 9) Perché l’accuracy può mentire con classi sbilanciate?
**Risposta:** Se i positivi sono rari, dire “sempre negativo” può dare alta accuracy ma **recall=0**.  
**Soluzione:** guarda **Precision**, **Recall**, **F1** e la **Confusion matrix**.

---

## 10) Precision, Recall, F1: definizioni pratiche
**Precision** = tra gli **allarmi** che hai dato, quanti erano **giusti**? = `TP/(TP+FP)`  
**Recall** = tra i **positivi veri**, quanti **non hai perso**? = `TP/(TP+FN)`  
**F1** = media armonica di Precision e Recall → sale **solo** se **entrambi** sono alti.  
**Quando usarle:** problemi sbilanciati o quando i costi di FP/FN sono diversi.

---

## 11) ROC-AUC: cosa misura e perché non dipende da 0.5
**Risposta:** Valuta la qualità del **ranking** delle probabilità su **tutte le soglie**.  
**Interpretazione:** probabilità che un positivo abbia score > di un negativo. **0.5** ≈ a caso, **1.0** perfetto. Non usa la soglia fissa 0.5.

---

## 12) Come si sceglie la soglia? Tre strade
1) **Auto-threshold** su validation/OOF per massimizzare **F1** o **Youden = TPR−FPR**.  
2) **A costo**: minimizza `c_fp·FP + c_fn·FN` (decidi tu quanto pesa perdere un positivo).  
3) **Sweep**: stampi la tabella soglia → (FP, FN, Precision, Recall, F1) e scegli consapevolmente.  
**Regola pratica:** se **FN è gravissimo**, abbassa la soglia.

---

## 13) Cosa sono Youden e OOF?
**Youden J:** `TPR − FPR` → cerca la soglia più distante dalla diagonale della ROC.  
**OOF (Out-Of-Fold):** predizioni di CV sul **train**, fatte da modelli che **non** hanno visto quel campione. Servono per scelte **oneste** (soglia, calibrazione) senza toccare il test.

---

## 14) Quando serve lo scaling?
**Risposta:** Serve per modelli che usano **distanze o geometria** (Logit, SVM, k-NN, PCA).  
In genere **non serve** per **Tree/Random Forest** (lavorano per soglie, non distanze).

---

## 15) Logistic Regression: pro/contro e iper chiave
**Pro:** semplice, veloce, **probabilità ben calibrate**, ottima su confini **quasi lineari**.  
**Contro:** fatica con **forte non-linearità** se non aggiungi feature/interazioni.  
**Iper:** `C` (meno regolarizzazione se alto) → occhio all’overfit; **scalare** le feature è essenziale.

---

## 16) Random Forest: pro/contro e pomelli
**Pro:** coglie **non-linearità** e **interazioni**, robusta, pochi settaggi critici.  
**Contro:** **probabilità** spesso **non calibrate**; spiegabilità globale limitata.  
**Pomelli:** `n_estimators`, `max_depth`, `max_features`, `class_weight` per sbilanciamento.

---

## 17) Perché calibrare le probabilità di una RF? Come?
**Risposta:** Perché gli score possono essere **troppo ottimisti/piatti**. Con `CalibratedClassifierCV`:
- **sigmoid (Platt):** semplice, pochi dati.  
- **isotonic:** più flessibile, richiede più dati.  
**Trucco pratico:** per leggere `feature_importances_` quando usi la calibrazione, rifitta un **clone** della RF **non calibrata** sul train.

---

## 18) Importanza delle feature: “impurity” vs “permutation”
**Impurity (feature_importances_)**: veloce, ma **favorisce** feature con tante soglie; con feature **correlate** si **divide il merito**.  
**Permutation (su TEST)**: misura il **calo reale** di metrica quando “rompi” la feature → spesso **più onesta**.  
**Uso consigliato:** guarda **entrambe** e conferma con **ablation**.

---

## 19) Effetto delle feature molto correlate
**Risposta:** Se due feature dicono **la stessa cosa**, l’importanza si **spalma** su entrambe: nessuna sembra “dominare” da sola.  
**Cosa fare:** controlla **correlazioni** (|ρ|≥0.9), prova **ablation**, valuta **grouping**/RIDUZIONE (PCA o scelta di una sola).

---

## 20) Cos’è l’ablation test?
**Risposta:** Togli le top-1/top-k feature e **misuri** quanto **scende** la metrica (es. AUC).  
**Lettura:** se non scende, il segnale era **ridondante**; se crolla, quelle feature sono **cruciali**.

---

## 21) Cos’è il data leakage? Esempio e prevenzione
**Leakage:** usi info del **futuro/test** nell’addestramento.  
**Esempio:** scalare o imputare sull’**intero** dataset prima dello split/CV.  
**Prevenzione:** tutte le trasformazioni in **Pipeline/ColumnTransformer**, così ogni fold fitta **solo sul proprio train**.

---

## 22) Perché Pipeline + GridSearchCV previene il leakage?
**Risposta:** Perché imputazione/scaling/encoding e modello sono **insieme** nella pipeline e vengono **fittati dentro la CV** solo sui dati di **train** del fold. Il fold di validazione (e il test finale) restano **puliti**.

---

## 23) Predizioni OOF: perché usarle per la soglia?
**Risposta:** Sono predizioni su campioni mai visti dal rispettivo modello di fold → sono **oneste** come un mini-test interno. Permettono di scegliere soglie/calibrazioni **senza toccare** il test.

---

## 24) “Refit su F1”: cosa vuol dire e come valuti il test?
**Risposta:** In `GridSearchCV(refit="f1")`, dopo la CV si ri-addestra il **miglior modello** (per F1) su **tutto il train**. Poi si valuta **una sola volta** sul **test** (niente ulteriori scelte sul test).

---

## 25) Confusion matrix: lettura e collegamento ai costi
**TN:** negativi corretti; **FP:** falsi allarmi; **FN:** positivi persi; **TP:** positivi presi.  
**Costo totale:** `c_fp·FP + c_fn·FN`. **Soglia a costo:** scegli la soglia che **minimizza** questo valore, in base al tuo scenario.

---

## 26) Logit o Random Forest? Quando scegliere cosa
**Logit:** confini **lineari**, bisogno di **probabilità affidabili**, spiegabilità dei **coefficienti**.  
**RF:** dati **non-lineari**, interazioni, “out-of-the-box” forte; poi regola **soglia** e, se servono probabilità affidabili, **calibra**.

---

## 27) `class_weight='balanced'`: cosa fa e quando usarlo?
**Risposta:** Pesa gli errori della classe **minoritaria** di più (peso ≈ 1/frequenza).  
**Quando:** sbilanciamento marcato e interesse a **recall** migliore senza riscrivere la loss.

---

## 28) Come controlli la stabilità dei risultati?
**Risposta:** Guarda **media ± std** in **CV** (anche ripetuta), prova più **seed**, verifica coerenza tra **OOF** e **test**. Se la varianza è alta, preferisci modelli/parametri più **semplici** (regola 1-SE).

---

## 29) Perché 0.5 non è una legge per la soglia?
**Risposta:** 0.5 presuppone **costi simmetrici** e probabilità perfettamente calibrate. Nel mondo reale **FN ≠ FP**: scegli la soglia in base a **F1/Youden/costo** (e dopo aver **calibrato** se servono probabilità affidabili).

---

## 30) Workflow “che non ti frega” (passo-passo)
1) Split train/test (test **in cassaforte**).  
2) **Pipeline** con preprocess → **CV** → **tuning** (regola 1-SE se serve).  
3) **Soglia** su validation/OOF (o per **costo**).  
4) **Fit finale** su tutto il train (stessa pipeline/parametri/soglia).  
5) **Valutazione** sul test una sola volta.  
6) **Spiega**: importanze/coef, curve ROC/PR, ablation, calibrazione.

---

## Bonus) Come leggere un RUN SUMMARY stampato dagli script
**Risposta:** È un estratto **a prova d’occhio**: seed, soglia, metriche **train/test**, **confusion** e **CV (media±std)**, eventuale **costo**. Ti dice **come** hai scelto e **quanto** puoi fidarti (stabilità, onestà metodologica).

---

### Mini-glossario
- **Parametro:** numero interno al modello regolato dall’algoritmo (le “manopole”).  
- **Iperparametro:** numero scelto dall’utente con CV (profondità, C, learning rate…).  
- **Loss/Errore:** numero che misura “quanto sto sbagliando” durante l’addestramento.  
- **Generalizzazione:** andare bene su dati nuovi (test).  
- **Calibrazione:** rendere le **probabilità** coerenti con le frequenze reali.  
- **OOF:** predizioni su train ottenute senza vedere quel campione (da altri fold).  
- **Youden:** TPR−FPR; criterio di scelta soglia dalla ROC.  
- **Ablation:** test di robustezza rimuovendo feature importanti.
