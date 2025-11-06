# PHP MCP Tool (MCP-like Adapter)

**PHP-MCP** è un micro-adapter scritto in **PHP 7.4+** che emula il comportamento di un *Model Context Protocol* server minimale.

Permette a un LLM o a uno script esterno di comunicare via **STDIN/STDOUT in JSON**, eseguendo piccoli “tool” (funzioni locali) in modo sicuro e prevedibile.

---

## 🔍 Introduzione

L’obiettivo di questo esperimento è dimostrare come anche un linguaggio classico come PHP possa:

- rispondere a richieste strutturate provenienti da un modello LLM;
- validare input e restituire risultati in formato JSON coerente;
- mantenere isolamento e sicurezza tramite **whitelist** del filesystem;
- fungere da **adapter didattico** per integrazioni MCP future.

Attualmente il server è **MCP-like**, cioè non è conforme 100% al protocollo completo:  
gestisce `initialize`, `call_tool` e pochi tool dimostrativi (`ping`, `sum`, `fs_list`).

## 🧠 Architettura
LLM / Client MCP
      │
      │  JSON via STDIN
      ▼
┌────────────────────────┐
│  server.php (PHP-MCP)  │
│  - valida richiesta    │
│  - chiama il tool      │
│  - risponde JSON       │
└──────────┬─────────────┘
           │
           ▼
    Funzioni locali (ping, sum, fs_list)

## ⚙️ Tool disponibili
| Nome      | Descrizione                       | Input                      | Output                     |                  |
| --------- | --------------------------------- | -------------------------- | -------------------------- | ---------------- |
| `ping`    | Test di connessione               | `{}`                       | `{"message": "pong"}`      |                  |
| `sum`     | Somma due numeri                  | `{"a": <num>, "b": <num>}` | `{"result": <num>}`        |                  |
| `fs_list` | Elenca file/dir sotto root sicura | `{"path": ""}`             | `[{"name":"…","type":"file | dir","size":0}]` |

## 🧱 Sicurezza: SAFE_ROOT
- Tutte le operazioni di `fs_list` avvengono sotto una root *whitelisted*.
- Percorsi come `../` o file fuori da SAFE_ROOT vengono **bloccati**.
- Variabile d’ambiente:

  export SAFE_ROOT=tools/php-mcp/safe
 
  Se non impostata, il tool usa di default `tools/php-mcp/safe/`.

## 🚀 Come provarlo

### 1️⃣ Esegui handshake iniziale
cat tools/php-mcp/examples/initialize.json | tools/php-mcp/bin/run.sh

Output (estratto):
{"type":"initialize_result","tools":[{"name":"ping"},{"name":"sum"},{"name":"fs_list"}]}

### 2️⃣ Prova i tool base

# Ping
echo '{"type":"call_tool","name":"ping","args":{}}' | tools/php-mcp/bin/run.sh

# Somma
echo '{"type":"call_tool","name":"sum","args":{"a":10,"b":32}}' | tools/php-mcp/bin/run.sh

### 3️⃣ Testa la whitelist

# Elenco root
cat tools/php-mcp/examples/call_fs_list_root.json | tools/php-mcp/bin/run.sh

# Sottocartella
cat tools/php-mcp/examples/call_fs_list_sub.json | tools/php-mcp/bin/run.sh

# Traversal bloccato
cat tools/php-mcp/examples/call_fs_list_escape.json | tools/php-mcp/bin/run.sh

## 🧾 Output d’esempio (OK)
{"type":"tool_result","name":"fs_list","result":[{"name":"subdir","type":"dir","size":0},{"name":"hello.txt","type":"file","size":6}]}

## ⚠️ Output d’esempio (errore)
{"type":"error","code":"BAD_ARGS","error":"Percorso non ammesso (traversal o fuori whitelist)"}

## 🧩 Estendere il tool
Aggiungere nuovi comandi è semplice:

1. Apri `tools/php-mcp/server.php`

2. Inserisci nel registro `$tools` una nuova voce:

   'version' => [
       'desc' => 'Restituisce versione PHP e tool',
       'schema' => ['type' => 'object', 'properties' => []],
       'fn' => function(array $args): array {
           return ['php' => phpversion(), 'tool' => 'PHP-MCP v0.1'];
       }
   ],

3. Testa con:

   echo '{"type":"call_tool","name":"version","args":{}}' | tools/php-mcp/bin/run.sh

## 🧪 Test rapido automatico

Per controllo base:

bash -c '
echo "Test ping"
echo "{\"type\":\"call_tool\",\"name\":\"ping\",\"args\":{}}" | tools/php-mcp/bin/run.sh | grep pong
echo "Test fs_list root"
echo "{\"type\":\"call_tool\",\"name\":\"fs_list\",\"args\":{\"path\":\"\"}}" | tools/php-mcp/bin/run.sh | grep hello.txt
'

## 🧮 Versione e stato
| Campo          | Valore                                                             |
| -------------- | ------------------------------------------------------------------ |
| Versione tool  | **v0.1-MCP**                                                       |
| Compatibilità  | PHP 7.4 +                                                          |
| Stato          | *Esperimento stabile / demo didattica*                             |
| Prossimi passi | logging NDJSON · codici errore uniformi · integrazione DB readonly |

## 📚 Risorse utili
* [Specifica Model Context Protocol (MCP) – GitHub](https://github.com/modelcontextprotocol)
* [Documentazione ML-Lab](../index.md)
* [Repository principale](https://github.com/gcomneno/ML-Lab)

---

> 💡 *PHP-MCP* fa parte del laboratorio **ML-Lab** ed è pensato per esplorare
> l’interazione tra strumenti legacy e modelli generativi moderni.
> È liberamente estendibile e mantenuto da **Giadaware / Giancarlo Cicellyn Comneno**.
