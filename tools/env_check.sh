#!/usr/bin/env bash
set -euo pipefail

RED()  { printf "\033[31m%s\033[0m\n" "$*"; }
GRN()  { printf "\033[32m%s\033[0m\n" "$*"; }
YLW()  { printf "\033[33m%s\033[0m\n" "$*"; }
SEC()  { printf "\n\033[36m# %s\033[0m\n" "$*"; }

fail() { RED "✗ $*"; exit 1; }
ok()   { GRN "✓ $*"; }

# Trova l'interprete Python
if command -v python3 >/dev/null; then PY=python3
elif command -v python >/dev/null; then PY=python
else fail "python3 (o python) mancante"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEC "Riepilogo ambiente"
echo "PWD         : $PWD"
echo "User        : $(whoami)"
echo "Branch      : $(git rev-parse --abbrev-ref HEAD || echo 'n/a')"
echo "Python      : $($PY -V || true)"
echo "Pip         : $($PY -m pip --version || true)"
echo "PHP         : $(php -v | head -n1 || echo 'missing')"
echo "MkDocs      : $(mkdocs --version 2>/dev/null || echo 'missing')"
echo "jq          : $(jq --version 2>/dev/null || echo 'missing')"
echo "gh          : $(gh --version 2>/dev/null | head -n1 || echo 'missing')"

SEC "Prerequisiti minimi"
command -v $PY >/dev/null   || fail "Python mancante"
command -v php  >/dev/null  || fail "PHP mancante (7.4+)"
command -v jq   >/dev/null  || YLW "jq non trovato (consigliato per parse JSON)"
command -v mkdocs >/dev/null || YLW "mkdocs non trovato (build docs opzionale)"

# Version checks
PY_MAJ=$($PY - <<'EOF'
import sys
print(sys.version_info.major, sys.version_info.minor, sep='.')
EOF
)
if [[ "${PY_MAJ%%.*}" -lt 3 ]]; then fail "Python < 3"; fi

PHP_VER=$(php -r 'echo PHP_MAJOR_VERSION.".".PHP_MINOR_VERSION;')
if [[ "${PHP_VER%%.*}" -lt 7 ]]; then fail "PHP < 7"; fi
if [[ "${PHP_VER%%.*}" -eq 7 && "${PHP_VER#*.}" -lt 4 ]]; then YLW "PHP $PHP_VER (ok, ma target 7.4+)"; fi

SEC "Python venv + deps"
if [[ ! -d ".venv" ]]; then
  YLW "venv non trovata → la creo"
  $PY -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python3 -m pip install -U pip >/dev/null
pip install -r requirements.txt >/dev/null || fail "pip install -r requirements.txt"
ok "Deps Python installate"

SEC "Smoke test script Python"
python3 -m compileall -q scripts || fail "Syntax error in scripts/"
python3 scripts/iris.py --help              >/dev/null || fail "iris.py --help"
python3 scripts/imbalance.py --help         >/dev/null || fail "imbalance.py --help"
python3 scripts/forest_vs_logit.py --help   >/dev/null || fail "forest_vs_logit.py --help"
python3 scripts/importance_demo.py --help   >/dev/null || fail "importance_demo.py --help"
python3 scripts/gridsearch_mixed.py --help  >/dev/null || fail "gridsearch_mixed.py --help"
ok "Script Python avviabili"

SEC "PHP-MCP: handshake + ping + fs_list"
pushd tools/php-mcp >/dev/null
chmod +x bin/run.sh || true
echo '{"type":"initialize"}' | bin/run.sh | jq -e '.type=="initialize_result"' >/dev/null || fail "MCP initialize"
echo '{"type":"call_tool","name":"ping","args":{}}' | bin/run.sh | jq -e '.result.message=="pong"' >/dev/null || fail "MCP ping"
mkdir -p safe; touch safe/.keep
echo '{"type":"call_tool","name":"fs_list","args":{"path":""}}' | bin/run.sh | jq -e '.type=="tool_result"' >/dev/null || fail "MCP fs_list"
popd >/dev/null
ok "PHP-MCP ok"

SEC "Docs (MkDocs) build"
if command -v mkdocs >/dev/null; then
  mkdocs build --strict >/dev/null || fail "mkdocs build"
  ok "Docs buildate in ./site"
else
  YLW "MkDocs non presente: salto build docs"
fi

SEC "Git sanity"
git status --porcelain
ok "Git pronto"

SEC "Conclusione"
ok "Ambiente OK 🚀"
