#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turbo-K — valutatore di distribuzione per bucket (CIDR, χ²/DoF) + search/compare oli

Novità:
- --search-a N : prova N valori casuali (dispari) di 'a' e sceglie il migliore (χ²/DoF min)
- --search-b N : prova N valori casuali per 'b' (a fisso) e sceglie il migliore (χ²/DoF min)
- Blacklist pattern: scarta 'a' con byte ripetuti/pattern ovvi; scarta 'b' con Hamming weight troppo bassa
- --presets PATH : carica coppie (a,b) da YAML/TOML e le confronta sulla sorgente selezionata
- Report “file-mode normalizzato” (U, s=N/U, chi²/DoF normalizzato, solo-unici)
"""

import argparse
import ipaddress
import math
from typing import List, Tuple, Optional, Sequence
import numpy as np

# ----------------------
# Core Turbo-K utilities
# ----------------------

MASK32 = np.uint32(0xFFFFFFFF)

def parse_int_auto(s: str) -> int:
    """Parsa interi in dec/hex (0x...)."""
    return int(s, 0)

def turbo_perm_uint32(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """y = (a*x + b) mod 2^32 (vectorizzato)."""
    ax = (np.uint64(a) * np.uint64(x)) + np.uint64(b)
    return np.uint32(ax & np.uint64(0xFFFFFFFF))

def bucket_msb(y: np.ndarray, K: int) -> np.ndarray:
    return np.uint32(y >> np.uint32(32 - K))

def bucket_mod(y: np.ndarray, M: int) -> np.ndarray:
    return np.uint32(np.uint64(y) % np.uint64(M))

def inv_mod_pow2_32(a: int) -> int:
    """Inverso moltiplicativo di a modulo 2^32 (a dispari)."""
    if a % 2 == 0:
        raise ValueError("a deve essere dispari per avere inverso mod 2^32.")
    x = 1
    for _ in range(5):  # 2 -> 4 -> 8 -> 16 -> 32 bit
        x = (x * (2 - a * x)) & 0xFFFFFFFF
    return x

# ----------------------
# Pattern blacklist & helpers
# ----------------------

_BAD_BYTES = {0xA5, 0x5A, 0xAA, 0x55, 0xFF, 0x00}

def bytes_of_u32(v: int) -> Tuple[int,int,int,int]:
    return ((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)

def is_bad_a(a: int) -> bool:
    """Scarta 'a' con byte ripetuti o pattern ovvi (A5,5A,AA,55/FF/00).
       Nota: 'a' deve comunque essere DISPARI."""
    b3,b2,b1,b0 = bytes_of_u32(a)
    bs = (b3,b2,b1,b0)
    # Tutti uguali e "sospetti"
    if len(set(bs)) == 1 and b3 in _BAD_BYTES:
        return True
    # Pattern a due byte ripetuto tipo A5,5A,A5,5A o 55,AA,55,AA
    if bs[0] in _BAD_BYTES and bs[1] in _BAD_BYTES and bs == (bs[0], bs[1], bs[0], bs[1]):
        return True
    # Pattern 0xFFFFFFFF o 0x00000001 (banalotti)
    if a in (0xFFFFFFFF, 0x00000001, 0x00000000):
        return True
    return False

def popcount32(x: int) -> int:
    # compat con py3.8 senza int.bit_count()
    return bin(x & 0xFFFFFFFF).count("1")

def is_bad_b(b: int, min_pop: int = 5) -> bool:
    """Scarta 'b' con Hamming weight troppo bassa o 'banali'."""
    if b in (0x00000000, 0xFFFFFFFF, 0x00000001):
        return True
    return popcount32(b) < min_pop

# ----------------------
# Data generation
# ----------------------

def sample_uniform_uint32(N: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=np.uint64(1) << np.uint64(32), size=N, dtype=np.uint64).astype(np.uint32)

def parse_cidr(cidr: str) -> Tuple[int, int]:
    net = ipaddress.IPv4Network(cidr, strict=False)
    base = int(net.network_address)
    size = 1 << (32 - net.prefixlen)
    return base, size

def sample_from_cidrs(N: int, cidrs: List[str], rng: np.random.Generator, weights: Optional[List[float]] = None) -> np.ndarray:
    if not cidrs:
        raise ValueError("Con --source cidr serve almeno un --cidr A.B.C.D/p")
    K = len(cidrs)
    if weights is None:
        weights = [1.0 / K] * K
    if len(weights) != K:
        raise ValueError("--weights deve avere la stessa lunghezza del numero di --cidr")

    s = sum(weights)
    weights = [w / s for w in weights]
    counts = [int(round(N * w)) for w in weights]
    while sum(counts) < N:
        counts[np.argmax(weights)] += 1
    while sum(counts) > N:
        counts[np.argmax(counts)] -= 1

    out = np.empty(N, dtype=np.uint32)
    idx = 0
    for c, cnt in zip(cidrs, counts):
        base, size = parse_cidr(c)
        offs = rng.integers(low=0, high=size, size=cnt, dtype=np.uint64)
        xs = (np.uint64(base) + offs) & np.uint64(0xFFFFFFFF)
        out[idx:idx+cnt] = xs.astype(np.uint32)
        idx += cnt
    rng.shuffle(out)
    return out

def read_ips_file(path: str) -> np.ndarray:
    vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                if "." in s:
                    vals.append(int(ipaddress.IPv4Address(s)))
                else:
                    vals.append(int(s, 0))
            except Exception:
                continue
    if not vals:
        raise ValueError(f"Nessun IP valido in {path}")
    return np.array(vals, dtype=np.uint64).astype(np.uint32)

# ----------------------
# Metrics & reporting
# ----------------------

def chi2_stats(counts: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Ritorna: (chi2, dof, chi2_per_dof, std_rel, z_max)
    - std_rel = std(counts) / E
    - z_max   = (max(counts)-E) / sqrt(E)
    """
    N = int(np.sum(counts))
    B = int(len(counts))
    E = N / B if B > 0 else 0.0
    if E == 0:
        return 0.0, float(B - 1), 0.0, 0.0, 0.0
    dif = counts - E
    chi2 = float(np.sum((dif * dif) / E))
    dof = float(B - 1)
    chi2_per_dof = chi2 / dof if dof > 0 else float("nan")
    std_rel = float(np.std(counts) / E)
    z_max = float((np.max(counts) - E) / math.sqrt(E))
    return chi2, dof, chi2_per_dof, std_rel, z_max

def traffic_light(chi2_per_dof: float) -> str:
    if 0.9 <= chi2_per_dof <= 1.2:
        return "🟢"
    if 0.8 <= chi2_per_dof < 0.9 or 1.2 < chi2_per_dof <= 1.4:
        return "🟡"
    return "🔴"

def print_top_buckets(counts: np.ndarray, top: int = 10):
    N = int(np.sum(counts))
    B = len(counts)
    E = N / B if B > 0 else 0.0
    order = np.argsort(counts)[::-1]
    print(f"\n[Top {top} bucket più carichi]   (count, +% vs atteso)")
    for i in order[:top]:
        c = int(counts[i])
        over = ((c - E) / E * 100.0) if E > 0 else 0.0
        print(f"  #{i:>6d} : {c:>8d}   ({over:+6.2f}%)")
    order_rev = np.argsort(counts)
    print(f"\n[Top {top} bucket più leggeri]  (count, -% vs atteso)")
    for i in order_rev[:top]:
        c = int(counts[i])
        under = ((c - E) / E * 100.0) if E > 0 else 0.0
        print(f"  #{i:>6d} : {c:>8d}   ({under:+6.2f}%)")

def human_oil(a: int, b: int) -> str:
    return f"a={a} (0x{a:08X}, {'dispari' if a & 1 else 'pari ❌'}),  b={b} (0x{b:08X})"

def print_cheatsheet():
    print("\n--- Cheatsheet Turbo-K (appunti rapidi) ---")
    print("* y = (a*x + b) mod 2^32 con a DISPARI (invertibile).")
    print("* Bucket: MSB => B=2^K  |  MOD => B=M")
    print("* Valida con χ²/DoF ≈ 1 (verde 0.9–1.2). E = N/B >= ~50 per stabilità.")
    print("* Se χ²/DoF alto: riduci K, cambia (a,b) o usa MOD con M diverso.")
    print("* La mul a*x diffonde i bit bassi verso gli MSB (carry).")
    print("------------------------------------------")

# ----------------------
# Search & Compare helpers
# ----------------------

def evaluate_counts(X: np.ndarray, mode: str, K: int, M: int, a: int, b: int) -> Tuple[np.ndarray, Tuple[float,float,float,float,float]]:
    y = turbo_perm_uint32(X, a, b)
    if mode == "msb":
        buckets = bucket_msb(y, K)
        B = 1 << K
    else:
        buckets = bucket_mod(y, M)
        B = M
    counts = np.bincount(buckets.astype(np.int64), minlength=B)
    stats = chi2_stats(counts)
    return counts, stats

def search_best_a(X: np.ndarray, mode: str, K: int, M: int, b: int, trials: int, rng: np.random.Generator, top_k: int = 10):
    print(f"\n[Search-a] Provo {trials} valori casuali DISPARI per 'a' (b fisso = 0x{b:08X}) …")
    rows = []
    tried = 0
    while tried < trials:
        a = int(rng.integers(0, 1 << 32)) | 1  # forza a dispari
        if is_bad_a(a):
            continue
        _, stats = evaluate_counts(X, mode, K, M, a, b)
        chi2, dof, chi2_dof, std_rel, zmax = stats
        rows.append((chi2_dof, std_rel, zmax, a))
        tried += 1
    rows.sort(key=lambda t: t[0])
    print("\n[Search-a] Top risultati (min χ²/DoF):")
    print("    rank   a (hex)      χ²/DoF    std/E     z_max")
    for i, (chi2_dof, std_rel, zmax, a) in enumerate(rows[:top_k], 1):
        print(f"    {i:>4d}   0x{a:08X}   {chi2_dof:7.3f}   {std_rel:6.3f}   {zmax:6.2f}")
    best = rows[0]
    print(f"\n[Search-a] SUGGERITO: a=0x{best[3]:08X}  (χ²/DoF={best[0]:.3f})")
    return best[3]

def search_best_b(X: np.ndarray, mode: str, K: int, M: int, a: int, trials: int, rng: np.random.Generator, min_pop: int = 5, top_k: int = 10):
    print(f"\n[Search-b] Provo {trials} valori casuali per 'b' (a fisso = 0x{a:08X}, min popcount={min_pop}) …")
    rows = []
    tried = 0
    while tried < trials:
        b = int(rng.integers(0, 1 << 32))
        if is_bad_b(b, min_pop=min_pop):
            continue
        _, stats = evaluate_counts(X, mode, K, M, a, b)
        chi2, dof, chi2_dof, std_rel, zmax = stats
        rows.append((chi2_dof, std_rel, zmax, b))
        tried += 1
    rows.sort(key=lambda t: t[0])
    print("\n[Search-b] Top risultati (min χ²/DoF):")
    print("    rank   b (hex)      χ²/DoF    std/E     z_max")
    for i, (chi2_dof, std_rel, zmax, b) in enumerate(rows[:top_k], 1):
        print(f"    {i:>4d}   0x{b:08X}   {chi2_dof:7.3f}   {std_rel:6.3f}   {zmax:6.2f}")
    best = rows[0]
    print(f"\n[Search-b] SUGGERITO: b=0x{best[3]:08X}  (χ²/DoF={best[0]:.3f})")
    return best[3]

def compare_oils(X: np.ndarray, mode: str, K: int, M: int, pairs: List[Tuple[int,int]], names: Optional[Sequence[str]] = None):
    print("\n[Compare] Confronto oli (stessa sorgente)")
    header = "   #   a (hex)      b (hex)      χ²/DoF    std/E     z_max    note"
    if names is not None:
        header += "    name"
    print(header)
    best_idx = None
    best_val = float("inf")
    results = []
    for idx, (a, b) in enumerate(pairs, 1):
        note_parts = []
        if (a % 2) == 0:
            note_parts.append("a pari → forzato dispari")
            a |= 1
        if is_bad_a(a):
            note_parts.append("a(pattern)")
        if is_bad_b(b):
            note_parts.append("b(popcount)")
        _, stats = evaluate_counts(X, mode, K, M, a, b)
        chi2, dof, chi2_dof, std_rel, zmax = stats
        results.append((idx, a, b, chi2_dof, std_rel, zmax, " ".join(note_parts)))
        if chi2_dof < best_val:
            best_val, best_idx = chi2_dof, idx
    for j, (idx, a, b, chi2_dof, std_rel, zmax, note) in enumerate(results):
        star = "  <= BEST" if idx == best_idx else ""
        extra = f"    {names[j]}" if names is not None else ""
        print(f"{idx:4d}  0x{a:08X}  0x{b:08X}   {chi2_dof:7.3f}   {std_rel:6.3f}   {zmax:6.2f}  {note}{star}{extra}")
    if best_idx is not None:
        print(f"\n[Compare] Migliore: #{best_idx} (χ²/DoF {best_val:.3f})")

# ----------------------
# Preset loader (YAML/TOML)
# ----------------------

def load_presets(path: str) -> Tuple[List[Tuple[int,int]], List[str]]:
    """
    Ritorna (pairs, names) da un file YAML/TOML del tipo:
    oils:
      - name: default
        a: 0xDEADBEEF
        b: 0xBADC0FFE
      - name: stripes
        a: 0xA5A5A5A5
        b: 0x1
    """
    pairs: List[Tuple[int,int]] = []
    names: List[str] = []
    try:
        if path.lower().endswith((".yml", ".yaml")):
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif path.lower().endswith(".toml"):
            try:
                import tomllib as toml  # py3.11+
            except Exception:
                import toml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = toml.load(f)
        else:
            raise ValueError("Estensione non supportata (usa .yaml/.yml o .toml).")
    except ModuleNotFoundError as e:
        raise SystemExit(f"Per leggere '{path}' installa la dipendenza mancante (es. 'pip install pyyaml' o 'toml'): {e}")

    oils = data.get("oils", [])
    for item in oils:
        nm = str(item.get("name", f"oil_{len(names)+1}"))
        a = parse_int_auto(str(item["a"]))
        b = parse_int_auto(str(item["b"]))
        pairs.append((a, b))
        names.append(nm)
    if not pairs:
        raise SystemExit(f"Nessun oil trovato in {path}")
    return pairs, names

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Turbo-K eval — genera chiavi (uniformi/CIDR/file), applica Turbo-K e valuta la distribuzione per bucket (χ²/DoF)."
    )
    ap.add_argument("--mode", choices=["msb", "mod"], default="msb", help="Assegnazione bucket: msb (default) o mod.")
    ap.add_argument("--K", type=int, default=12, help="Usato con mode=msb: numero di bit alti (B=2^K).")
    ap.add_argument("--M", type=int, default=None, help="Usato con mode=mod: numero di bucket B=M.")
    ap.add_argument("--a", type=parse_int_auto, default=0xDEADBEEF, help="Coefficiente a (dec o 0xHEX). Deve essere dispari.")
    ap.add_argument("--b", type=parse_int_auto, default=0xBADC0FFE, help="Intercetta b (dec o 0xHEX).")
    ap.add_argument("--N", type=int, default=200_000, help="Numero di chiavi generate/lette.")
    ap.add_argument("--seed", type=int, default=0, help="Seed RNG (sampling e search).")

    # sorgente dati
    ap.add_argument("--source", choices=["uniform", "cidr", "file"], default="uniform", help="Sorgente chiavi x.")
    ap.add_argument("--cidr", action="append", default=[], help="CIDR A.B.C.D/p (ripetibile). Usato con --source cidr.")
    ap.add_argument("--weights", type=str, default=None, help="Pesi per i CIDR (comma separati), es. 2,1,1. Default = equi.")
    ap.add_argument("--ip-file", type=str, default=None, help="File con IP (una per riga), usato con --source file.")

    # funzioni nuove
    ap.add_argument("--search-a", type=int, default=0, help="Se >0, prova N valori casuali DISPARI per 'a' e suggerisce il migliore.")
    ap.add_argument("--search-b", type=int, default=0, help="Se >0, prova N valori casuali per 'b' (a fisso) e suggerisce il migliore.")
    ap.add_argument("--min-popcount-b", type=int, default=5, help="Soglia minima Hamming weight per i candidati 'b' (default 5).")
    ap.add_argument("--compare", action="append", default=[], help="Coppia 'a,b' (dec/hex) per confronto. Ripetibile.")
    ap.add_argument("--presets", type=str, default=None, help="Percorso YAML/TOML con oli predefiniti (usato per Compare).")

    ap.add_argument("--print-cheatsheet", action="store_true", help="Stampa appunti finali.")

    args = ap.parse_args()

    # Validazioni parametri principali
    if args.mode == "msb":
        if not (0 <= args.K <= 32):
            raise SystemExit("--K deve essere in [0,32]")
        B = 1 << args.K
    else:
        if args.M is None or args.M <= 0:
            raise SystemExit("--M deve essere specificato e > 0 quando mode=mod")
        B = args.M

    if args.a % 2 == 0:
        print("⚠️  a è pari: lo rendo dispari forzando l'ultimo bit (a|=1).")
        args.a |= 1

    rng = np.random.default_rng(args.seed)

    # Generazione/lettura dati
    src_desc = ""
    all_ips = None  # per normalizzazione file-mode
    if args.source == "uniform":
        X = sample_uniform_uint32(args.N, rng)
        src_desc = "UNIFORME su [0, 2^32)"
    elif args.source == "cidr":
        if not args.cidr:
            raise SystemExit("Usa --cidr A.B.C.D/p (ripetibile) con --source cidr")
        weights = None
        if args.weights:
            try:
                weights = [float(x) for x in args.weights.split(",")]
            except Exception:
                raise SystemExit("--weights deve essere comma-separated, es: 2,1,1")
        X = sample_from_cidrs(args.N, args.cidr, rng, weights)
        src_desc = "CIDR=" + ",".join(args.cidr)
    else:
        if not args.ip_file:
            raise SystemExit("Usa --ip-file PATH con --source file")
        all_ips = read_ips_file(args.ip_file)
        if len(all_ips) < args.N:
            idx = rng.integers(low=0, high=len(all_ips), size=args.N)
            X = all_ips[idx]
        else:
            X = all_ips.copy()
            rng.shuffle(X)
            X = X[:args.N]
        src_desc = f"FILE={args.ip_file} (n={len(all_ips)})"

    # Eval base (olio attuale)
    y = turbo_perm_uint32(X, args.a, args.b)
    if args.mode == "msb":
        buckets = bucket_msb(y, args.K)
        assign_desc = f"MSB, K={args.K} -> B={B}"
    else:
        buckets = bucket_mod(y, args.M)
        assign_desc = f"MOD, M={args.M} -> B={B}"
    counts = np.bincount(buckets.astype(np.int64), minlength=B)
    N = int(np.sum(counts))
    E = N / B
    chi2, dof, chi2_dof, std_rel, zmax = chi2_stats(counts)
    light = traffic_light(chi2_dof)

    print("\n=== Turbo-K Evaluation ===")
    print(f"Sorgente: {src_desc}")
    print(f"Assegnazione: {assign_desc}")
    print(f"Olio: {human_oil(args.a, args.b)}")
    try:
        inv_a = inv_mod_pow2_32(args.a)
        ok_inv = (args.a * inv_a) & 0xFFFFFFFF
        print(f"Inverso di a (mod 2^32): 0x{inv_a:08X}  [check a*a^-1 mod 2^32 = 0x{ok_inv:08X}]")
    except Exception as e:
        print(f"Inverso di a: n/a ({e})")

    # Warnings sui pattern
    warn = []
    if is_bad_a(args.a):
        warn.append("a(pattern sospetto)")
    if is_bad_b(args.b, min_pop=args.min_popcount_b):
        warn.append("b(popcount basso)")
    if warn:
        print(f"⚠️  Nota olio: {'; '.join(warn)}")

    print("\n[Distribuzione bucket]")
    print(f"  N={N:,d}   B={B:,d}   atteso per bucket E=N/B ≈ {E:,.2f}")
    print(f"  χ²={chi2:,.2f}  DoF={dof:,.0f}  χ²/DoF={chi2_dof:.3f}  {light}")
    print(f"  std/E={std_rel:.3f}   z_max≈{zmax:.2f}σ   max={int(np.max(counts)):,d}   min={int(np.min(counts)):,d}")
    if E < 50:
        print("  ⚠️  E < 50 → varianza alta per definizione. Riduci K (o aumenta N) per una misura più stabile.")
    print_top_buckets(counts, top=min(10, B))

    # File-mode normalized (se sorgente=file)
    if all_ips is not None:
        U = int(np.unique(all_ips).size)
        s = N / U if U > 0 else float("nan")

        # solo-unici
        ips_unique = np.unique(all_ips)
        y_u = turbo_perm_uint32(ips_unique, args.a, args.b)
        buckets_u = bucket_msb(y_u, args.K) if args.mode == "msb" else bucket_mod(y_u, args.M)
        counts_u = np.bincount(buckets_u.astype(np.int64), minlength=B)
        chi2_u, dof_u, chi2_dof_u, std_rel_u, zmax_u = chi2_stats(counts_u)

        # normalizzato dividendo i conteggi per s (E diventa U/B)
        counts_norm = counts.astype(float) / s
        E_u = U / B
        dif = counts_norm - E_u
        chi2_norm = float(np.sum((dif * dif) / E_u)) if E_u > 0 else 0.0
        chi2_dof_norm = chi2_norm / (B - 1) if B > 1 else float("nan")

        # stima bucket vuoti attesi con U/B = λ
        lam = U / B
        exp_empty = B * math.exp(-lam)

        print("\n[File-mode normalizzato]")
        print(f"  U (IP unici) = {U:,d}   N = {N:,d}   s=N/U = {s:.3f}")
        print(f"  χ²/DoF (pieno)       = {chi2_dof:.3f}")
        print(f"  χ²/DoF normalizzato  = {chi2_dof_norm:.3f}   (≈1 atteso se mixing ok)")
        print(f"  χ²/DoF (solo unici)  = {chi2_dof_u:.3f}")
        print(f"  Bucket vuoti (unici) : obs={(counts_u == 0).sum():d}  exp≈{int(round(exp_empty))}  (con U/B={lam:.2f})")
        if 0.9 <= chi2_dof_norm <= 1.2:
            print("  ✅ Mixing buono: il rosso del χ²/DoF ‘pieno’ dipende da s=N/U (ripetizioni), non dall’olio.")
        else:
            print("  ⚠️  Normalizzato ancora fuori fascia: prova a cambiare olio o riduci K.")

    # Compare (da --compare e/o --presets)
    pairs_cli: List[Tuple[int,int]] = []
    names_cli: List[str] = []
    if args.compare:
        for sarg in args.compare:
            try:
                a_str, b_str = sarg.split(",")
                a_val = parse_int_auto(a_str.strip())
                b_val = parse_int_auto(b_str.strip())
                pairs_cli.append((a_val, b_val))
                names_cli.append(f"cli_{len(names_cli)+1}")
            except Exception:
                raise SystemExit(f"--compare richiede 'a,b' (es. 0xDEAD,0xBEEF) — errato: {sarg}")

    if args.presets:
        pairs_p, names_p = load_presets(args.presets)
        pairs_cli.extend(pairs_p)
        names_cli.extend(names_p)

    if pairs_cli:
        M_eff = args.M if args.mode == "mod" else 0
        compare_oils(X, args.mode, args.K, M_eff, pairs_cli, names_cli)

    # Search-a (se richiesto)
    if args.search_a and args.search_a > 0:
        best_a = search_best_a(X, args.mode, args.K, args.M if args.mode=="mod" else 0, args.b, args.search_a, rng)
        print(f"\nSuggerimento finale: usa a=0x{best_a:08X} (mantieni b=0x{args.b:08X}) e riesegui per verifica.")

    # Search-b (se richiesto)
    if args.search_b and args.search_b > 0:
        best_b = search_best_b(X, args.mode, args.K, args.M if args.mode=="mod" else 0, args.a, args.search_b, rng, min_pop=args.min_popcount_b)
        print(f"\nSuggerimento finale: usa b=0x{best_b:08X} (mantieni a=0x{args.a:08X}) e riesegui per verifica.")

    # Cheatsheet (opzionale)
    if args.print_cheatsheet:
        print_cheatsheet()

if __name__ == "__main__":
    main()
