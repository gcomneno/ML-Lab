# PHP MCP Tool (minimal)

Un micro–tool **MCP** scritto in **PHP 7.4+** per test e demo. Espone due comandi:
- `ping()` → `{"message":"pong"}`
- `sum({a,b})` → `{"result": a+b}`

> MCP (*Model Context Protocol*) è il “contratto” che permette a un LLM di parlare con strumenti esterni tramite **STDIN/STDOUT** usando JSON. Qui lo implementiamo in forma essenziale.

## Requisiti

- PHP **7.4+** (ok sulla tua VM Ubuntu)  
- Nessuna dipendenza esterna (niente Composer obbligatorio per la versione minimale)

> Nota: se in CI userai una versione più nuova (8.x) non ci sono problemi: il codice resta compatibile.

## Struttura

```
tools/php-mcp/
├─ server.php
├─ bin/
│  └─ run
└─ examples/
├─ initialize.json
└─ call_ping.json

````

- `server.php`: legge una richiesta JSON da **STDIN** e scrive una risposta JSON su **STDOUT**.
- `bin/run.sh`: launcher comodo (ricorda `chmod +x tools/php-mcp/bin/run`).
- `examples/`: richieste d’esempio.

## Diagramma (ASCII)
```md
## Come fluisce tutto (schemino)

LLM (client MCP)
      │
      │  JSON su STDIN
      ▼
┌─────────────────────┐
│  server.php (PHP)   │
│  - parse richiesta  │
│  - valida schema    │
│  - chiama tool      │
└─────────┬───────────┘
          │
          │  es. ping() / sum(a,b)
          ▼
   Funzioni interne
          │
          │  JSON su STDOUT
          ▼
    Risposta al LLM
```

## Uso rapido (manuale)
Handshake MCP:
```bash
cat tools/php-mcp/examples/initialize.json | tools/php-mcp/bin/run
````

Chiamata `ping`:
```bash
cat tools/php-mcp/examples/call_ping.json | tools/php-mcp/bin/run
```

Somma:
```bash
echo '{"type":"call_tool","name":"sum","args":{"a":10,"b":32}}' | tools/php-mcp/bin/run
```

Output atteso (esempio):
```json
{"type":"tool_result","name":"sum","result":{"result":42}}
```

## API “tipo”
- **Initialize**
  ```json
  { "type": "initialize" }
  ```

  Risposta:
  ```json
  {
    "type": "initialize_result",
    "tools": [
      { "name": "ping", "description": "Risponde \"pong\".", "input_schema": {...} },
      { "name": "sum",  "description": "Somma due numeri.",  "input_schema": {...} }
    ]
  }
  ```

- **Call tool**
  ```json
  { "type": "call_tool", "name": "<toolName>", "args": { ... } }
  ```

  Risposta (OK):
  ```json
  { "type": "tool_result", "name": "<toolName>", "result": { ... } }
  ```

  Risposta (errore):
  ```json
  { "type": "error", "error": "Messaggio di errore" }
  ```

## Estensioni suggerite (Issue)
- `fs_list` con **whitelist** (solo directory sicure).
- `mysql_query` **readonly** (DSN via `.env`, niente credenziali nel repo).
- Logging su **STDERR** in NDJSON per non sporcare l’output del protocollo.

## Troubleshooting
- **Non stampa nulla?** Verifica che stai inviando JSON valido su `run.sh` e che `server.php` sia eseguibile via PHP
- **JSON su più righe?** Il server legge tutto lo STDIN; usa `echo '...' | run.sh` o `cat file.json | run.sh`.
- **CI fallisce?** Se hai un workflow Python rinominato, nessun problema. Il tool PHP non richiede CI propria per funzionare.
