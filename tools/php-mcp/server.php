#!/usr/bin/env php
<?php
declare(strict_types=1);

/**
 * Minimal MCP-like tool server (STDIN/STDOUT JSON) — PHP 7.4+
 * Tools:
 *  - ping() -> { message: "pong" }
 *  - sum({a,b}) -> { result: number }
 *  - fs_list({path}) -> [ {name, type, size} ]  (whitelist root)
 */

function readStdin(): array {
    $input = stream_get_contents(STDIN);
    if ($input === false || $input === '') return [];
    $json = json_decode($input, true);
    return is_array($json) ? $json : [];
}
function writeJson(array $payload): void {
    // singola riga per compatibilità pipe
    fwrite(STDOUT, json_encode($payload, JSON_UNESCAPED_SLASHES) . PHP_EOL);
}
function errorJson(string $msg, string $code = 'INVALID_REQUEST'): void {
    writeJson(['type' => 'error', 'code' => $code, 'error' => $msg]);
}

/** --- FS whitelist helpers --- **/
function getSafeRoot(): string {
    // Configurabile via env SAFE_ROOT, default: tools/php-mcp/safe (relative a CWD)
    $root = getenv('SAFE_ROOT');
    if ($root === false || $root === '') {
        $root = __DIR__ . DIRECTORY_SEPARATOR . 'safe';
    } elseif ($root[0] !== DIRECTORY_SEPARATOR) {
        // se relativo, rendilo assoluto relativo a CWD corrente
        $root = realpath(getcwd() . DIRECTORY_SEPARATOR . $root) ?: (getcwd() . DIRECTORY_SEPARATOR . $root);
    }
    // normalizza/crea se non esiste
    if (!is_dir($root)) {
        @mkdir($root, 0775, true);
    }
    $real = realpath($root);
    return $real !== false ? $real : $root;
}

/**
 * Restituisce percorso assoluto sicuro dentro SAFE_ROOT o false se traversal.
 * - Non permette ".."
 * - Normalizza separatori
 */
function resolveSafePath(string $relative): string {
    $root = getSafeRoot(); // assoluto
    // normalizza separatori e rimuovi caratteri pericolosi
    $relative = trim($relative);
    if ($relative === '.' || $relative === './') $relative = '';
    // vieta sequenze di traversal
    if (strpos($relative, '..') !== false) {
        return '';
    }
    // costruisci path
    $full = rtrim($root, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . ltrim($relative, DIRECTORY_SEPARATOR);
    $real = realpath($full);
    if ($real === false) {
        // Se non esiste, prova comunque a verificare containment sul path previsto
        $real = $full;
    }
    // Verifica containment: $real deve iniziare con $root
    $rootPrefix = rtrim($root, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR;
    $realPrefix = rtrim((string)$real, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR;
    if (strpos($realPrefix, $rootPrefix) !== 0) {
        return '';
    }
    return rtrim((string)$real, DIRECTORY_SEPARATOR);
}

/**
 * Lista contenuti (non ricorsiva) con name/type/size
 */
function listDirEntries(string $absDir): array {
    $out = [];
    if (!is_dir($absDir)) {
        return $out;
    }
    $dh = opendir($absDir);
    if ($dh === false) return $out;
    while (($e = readdir($dh)) !== false) {
        if ($e === '.' || $e === '..') continue;
        $p = $absDir . DIRECTORY_SEPARATOR . $e;
        $type = is_dir($p) ? 'dir' : (is_file($p) ? 'file' : 'other');
        $size = is_file($p) ? filesize($p) : 0;
        $out[] = ['name' => $e, 'type' => $type, 'size' => $size];
    }
    closedir($dh);
    // ordina per type (dir prima), poi name
    usort($out, function ($a, $b) {
        if ($a['type'] === $b['type']) return strcmp($a['name'], $b['name']);
        return $a['type'] === 'dir' ? -1 : 1;
    });
    return $out;
}

/** --- registry strumenti --- **/
$tools = [
    'ping' => [
        'desc' => 'Risponde "pong".',
        'schema' => ['type' => 'object', 'properties' => []],
        'fn' => function(array $args): array {
            return ['message' => 'pong'];
        }
    ],
    'sum' => [
        'desc' => 'Somma due numeri: a + b.',
        'schema' => [
            'type' => 'object',
            'properties' => [
                'a' => ['type' => 'number'],
                'b' => ['type' => 'number']
            ],
            'required' => ['a', 'b']
        ],
        'fn' => function(array $args): array {
            $a = $args['a'] ?? null; $b = $args['b'] ?? null;
            if (!is_numeric($a) || !is_numeric($b)) {
                return ['error' => 'Argomenti non numerici'];
            }
            return ['result' => (float)$a + (float)$b];
        }
    ],
    'fs_list' => [
        'desc' => 'Elenca i contenuti (non ricorsivi) sotto una root whitelisted.',
        'schema' => [
            'type' => 'object',
            'properties' => [
                'path' => ['type' => 'string', 'description' => 'Percorso relativo alla SAFE_ROOT']
            ],
            'required' => []
        ],
        'fn' => function(array $args): array {
            $rel = isset($args['path']) ? (string)$args['path'] : '';
            $abs = resolveSafePath($rel);
            if ($abs === '') {
                return ['error' => 'Percorso non ammesso (traversal o fuori whitelist)'];
            }
            if (!file_exists($abs)) {
                return ['error' => 'Percorso inesistente nella whitelist'];
            }
            if (is_file($abs)) {
                // Se è file singolo, restituisci solo quel file
                return [[
                    'name' => basename($abs),
                    'type' => 'file',
                    'size' => filesize($abs)
                ]];
            }
            // Directory
            return listDirEntries($abs);
        }
    ],
];

/** --- routing richieste --- **/
$request = readStdin();
$type = $request['type'] ?? '';

switch ($type) {
    case 'initialize':
        $toolList = [];
        foreach ($tools as $name => $meta) {
            $toolList[] = [
                'name' => $name,
                'description' => $meta['desc'],
                'input_schema' => $meta['schema']
            ];
        }
        writeJson(['type' => 'initialize_result', 'tools' => $toolList]);
        break;

    case 'call_tool':
        $name = $request['name'] ?? '';
        $args = is_array($request['args'] ?? null) ? $request['args'] : [];
        if (!isset($tools[$name])) {
            errorJson("Tool '$name' non trovato", 'TOOL_NOT_FOUND');
            break;
        }
        try {
            $result = $tools[$name]['fn']($args);
            if (is_array($result) && isset($result['error'])) {
                // normalizza errori "gentili"
                errorJson($result['error'], 'BAD_ARGS');
                break;
            }
            writeJson(['type' => 'tool_result', 'name' => $name, 'result' => $result]);
        } catch (Throwable $e) {
            errorJson('Errore interno', 'INTERNAL');
        }
        break;

    default:
        errorJson('Richiesta non valida o type mancante', 'INVALID_REQUEST');
        break;
}
