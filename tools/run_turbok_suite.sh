#!/usr/bin/env bash
# Turbo-K — Test suite end-to-end con logging e riepilogo
# Uso:
#   chmod +x tools/run_turbok_suite.sh
#   tools/run_turbok_suite.sh
#
# Requisiti:
#   - Python 3 + numpy
#   - scripts/turbo_k_eval.py presente
#   - (opzionale) .venv/ con Python

set -u -o pipefail

# ---- Utility ---------------------------------------------------------------

timestamp() { date +"%Y%m%d_%H%M%S"; }

slug() {
  tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
}

PY="python"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

if [[ ! -f "scripts/turbo_k_eval.py" ]]; then
  echo "ERRORE: scripts/turbo_k_eval.py non trovato. Esegui da root del repo." >&2
  exit 1
fi

OUTDIR="reports/turbo_k/$(timestamp)"
mkdir -p "$OUTDIR"

fail_count=0
run() {
  local title="$1"; shift
  local cmd="$*"

  local file_slug
  file_slug="$(echo "$title" | slug)"
  local logfile="$OUTDIR/$(printf "%02d" "$(( $(ls "$OUTDIR" | wc -l) + 1 ))"))_${file_slug}.log"

  echo
  echo "▶ ${title}"
  echo "   $cmd"
  echo "   -> $logfile"

  # Esegui, cattura exit code ma NON fermare la suite
  set +e
  eval "$cmd" | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "✗ FALLITO ($rc) — vedi $logfile"
    fail_count=$((fail_count+1))
  else
    echo "✓ OK — log: $logfile"
  fi
}

# ---- Dati di esempio (se mancano) ------------------------------------------

ensure_data_files() {
  mkdir -p data

  if [[ ! -f "data/ips.txt" ]]; then
    echo "Generazione data/ips.txt (10k IP, mix CIDR + random + duplicati)…"
    "$PY" - <<'PYCODE'
import random, ipaddress, sys
random.seed(42)
ips=set()
def add_cidr(cidr, n):
    net=ipaddress.IPv4Network(cidr, strict=False)
    for _ in range(n):
        ip=int(net.network_address)+random.randrange(0, net.num_addresses)
        ips.add(str(ipaddress.IPv4Address(ip)))
add_cidr("10.0.0.0/8", 4500)
add_cidr("192.168.0.0/16", 3000)
add_cidr("172.16.0.0/12", 1500)
while len(ips)<9000:
    ips.add(str(ipaddress.IPv4Address(random.getrandbits(32))))
ips=list(ips)
dup=[]
for x in ips[:1000]:
    dup.extend([x]*random.randint(2,5))
pool=ips+dup
random.shuffle(pool)
pool=pool[:10000]
open("data/ips.txt","w").write("\n".join(pool))
print("Scritto data/ips.txt con", len(pool), "righe")
PYCODE
  fi

  if [[ ! -f "data/ips_small.txt" ]]; then
    echo "Generazione data/ips_small.txt (1k IP, più duplicati)…"
    "$PY" - <<'PYCODE'
import random, ipaddress
random.seed(7)
base=[str(ipaddress.IPv4Address(int(ipaddress.IPv4Address("203.0.113.0"))+i)) for i in range(200)]
pool=[]
for x in base:
    pool.extend([x]*random.randint(1,8))
random.shuffle(pool)
pool=pool[:1000]
open("data/ips_small.txt","w").write("\n".join(pool))
print("Scritto data/ips_small.txt con", len(pool), "righe")
PYCODE
  fi
}

# ---- Lancio test -----------------------------------------------------------

echo "=== Turbo-K Test Suite ==="
echo "Python: $PY"
echo "Output: $OUTDIR"
echo

ensure_data_files

# 1) Uniforme — MSB K=12
run "uniform-msb-k12" \
  "$PY scripts/turbo_k_eval.py --source uniform --mode msb --K 12 --N 200000"

# 2) Uniforme — MOD M=4096
run "uniform-mod-m4096" \
  "$PY scripts/turbo_k_eval.py --source uniform --mode mod --M 4096 --N 200000"

# 3) CIDR mix
run "cidr-mix-msb-k12" \
  "$PY scripts/turbo_k_eval.py --source cidr --cidr 10.0.0.0/8 --cidr 192.168.0.0/16 --mode msb --K 12 --N 300000"

# 4) Search-a (uniform)
run "search-a-128-uniform-msb-k12" \
  "$PY scripts/turbo_k_eval.py --source uniform --mode msb --K 12 --N 200000 --search-a 128"

# 5) Search-a (cidr)
run "search-a-256-cidr-msb-k12" \
  "$PY scripts/turbo_k_eval.py --source cidr --cidr 10.0.0.0/8 --cidr 192.168.0.0/16 --mode msb --K 12 --N 300000 --search-a 256"

# 6) Compare oli
run "compare-oils" \
  "$PY scripts/turbo_k_eval.py --K 12 --compare 0xDEADBEEF,0xBADC0FFE --compare 0xA5A5A5A5,0x1"

# 7) File grande
run "file-ips-k12" \
  "$PY scripts/turbo_k_eval.py --source file --ip-file ./data/ips.txt --K 12"

# 8) File piccolo
run "file-ips-small-k10" \
  "$PY scripts/turbo_k_eval.py --source file --ip-file ./data/ips_small.txt --K 10"

# 9) Sweep K (uniforme)
for K in 10 11 12 13; do
  run "uniform-msb-sweep-k${K}" \
    "$PY scripts/turbo_k_eval.py --source uniform --mode msb --K ${K} --N 200000"
done

# 10) MOD con M non potenza di 2
run "uniform-mod-m1000" \
  "$PY scripts/turbo_k_eval.py --source uniform --mode mod --M 1000 --N 200000"

# 11) Variabilità seed
for s in 0 1 2 3 4; do
  run "uniform-msb-k12-seed-${s}" \
    "$PY scripts/turbo_k_eval.py --source uniform --mode msb --K 12 --N 200000 --seed ${s}"
done

# ---- Riepilogo -------------------------------------------------------------

echo
echo "=== RIEPILOGO ==="

# riassunto chi2/DoF per ciascun log
echo
echo "-- χ²/DoF per test --"
grep -R "χ²=.*χ²/DoF=" "$OUTDIR" | sed -E "s|$OUTDIR/||" | awk -F':' '{file=$1; line=$2; sub(/^ +/,"",line); print file"  |  "line}'

# best da compare
echo
echo "-- Migliori da [Compare] --"
grep -R " <= BEST" "$OUTDIR" | sed -E "s|$OUTDIR/||"

# suggerimenti da search-a
echo
echo "-- Suggerimenti da [Search-a] --"
grep -R "\[Search-a\] SUGGERITO" "$OUTDIR" | sed -E "s|$OUTDIR/||"

# esiti
echo
if [[ $fail_count -gt 0 ]]; then
  echo "Completato con $fail_count fallimenti. Controlla i log in: $OUTDIR"
  exit 1
else
  echo "Tutti i comandi completati. Log in: $OUTDIR"
fi
