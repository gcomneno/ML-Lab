# 🧪 ML-Lab — Machine Learning Laboratory

[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://gcomneno.github.io/ML-Lab/)
[![CI](https://img.shields.io/github/actions/workflow/status/gcomneno/ML-Lab/python-ci.yaml?branch=main)](https://github.com/gcomneno/ML-Lab/actions)
[![PHP CI](https://img.shields.io/github/actions/workflow/status/gcomneno/ML-Lab/php-ci.yaml?branch=main)](https://github.com/gcomneno/ML-Lab/actions)
![Python](https://img.shields.io/badge/python-3.8%2B-informational)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

**ML-Lab** è un laboratorio aperto di *Machine Learning* e strumenti predittivi.  
Nasce per studiare in modo trasparente modelli, dataset e decisioni, combinando codice leggibile e analisi guidate.

### 🎯 Obiettivi
- 🧠 capirne la logica, non solo usarla;  
- 📊 rendere i risultati ripetibili e commentati;  
- 🔬 sperimentare algoritmi e integrazioni “fuori standard”.

---

## ⚙️ Setup rapido
```bash
git clone https://github.com/gcomneno/ML-Lab.git
cd ML-Lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

### Prova rapida

```bash
python scripts/iris.py --tune --print-cheatsheet
```

---

## 🧩 PHP-MCP — micro adapter MCP-like
> Esperimento interno di *ML-Lab* per collegare strumenti PHP a modelli LLM tramite **Model Context Protocol-like**.

**Caratteristiche**

* ✅ 100 % JSON via STDIN / STDOUT, nessuna dipendenza esterna
* 🧱 Tool integrati: `ping`, `sum`, `fs_list` (whitelist del filesystem)
* 🧩 Estendibile con nuovi tool in poche righe
* 🧮 Compatibile PHP 7.4 + (demo stabile v0.1-MCP)

📘 [Documentazione completa → `docs/tools/php-mcp.md`](docs/tools/php-mcp.md)

Esempio:

```bash
echo '{"type":"call_tool","name":"ping","args":{}}' | tools/php-mcp/bin/run.sh
```

Output:

```json
{"type":"tool_result","name":"ping","result":{"message":"pong"}}
```

---

## 📚 Documentazione
La documentazione completa (MkDocs) è disponibile qui:
👉 **[https://gcomneno.github.io/ML-Lab/](https://gcomneno.github.io/ML-Lab/)**

### Sezioni principali

* **Start in 10 minuti** — setup, esempi e primi script.
* **Tools** — moduli Python e PHP-MCP.
* **Reports** — esempi di output e analisi guidate.

---

## 🧮 Struttura del progetto
```
ML-Lab/
 ├── scripts/          # esperimenti e mini-prove ML
 ├── tools/
 │    └── php-mcp/     # micro adapter MCP-like in PHP
 ├── docs/             # documentazione MkDocs
 ├── requirements.txt
 ├── Makefile
 └── ...
```

---

## 🧭 Prossimi Passi
Consulta la roadmap completa su GitHub:  
👉 [https://github.com/gcomneno/ML-Lab/issues](https://github.com/gcomneno/ML-Lab/issues)

Contribuzioni e feedback sono benvenuti!

---

## 🤝 Contributing
1. Forka il progetto
2. Crea un branch (`feat/qualcosa`)
3. Commit + PR chiaro
4. CI verde prima del merge

Consulta anche `CONTRIBUTING.md` (in arrivo) e la [security checklist](docs/security-checklist.md) non appena disponibile.

---

## 📜 License
[MIT License](LICENSE) © Giadaware / Giancarlo Comneno

---

> *ML-Lab è parte dell’ecosistema Giadaware: un luogo dove convivono codice, statistica e curiosità!*
