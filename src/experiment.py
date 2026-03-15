#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np


ROOT = Path('/home/clawuser/.openclaw/workspace/holo-repro')
RUNS = ROOT / 'runs'
REPORTS = ROOT / 'reports'


@dataclass
class Metric:
    mode: str
    n: int
    pairs: int
    corr: float
    slope: float
    mae: float


def build_disk_graph(n: int, rng: np.random.Generator) -> Tuple[nx.Graph, List[int], List[int]]:
    # Ring boundary + random interior with geometric preference
    nb = n // 2
    ni = n - nb
    G = nx.Graph()
    boundary = list(range(nb))
    interior = list(range(nb, n))
    G.add_nodes_from(range(n))
    # boundary ring
    for i in range(nb):
        G.add_edge(boundary[i], boundary[(i + 1) % nb], w=1.0)
    # interior random geometric-like links
    for u in interior:
        # connect to 2-4 random boundary nodes + 2 interior nodes
        b_deg = rng.integers(2, 5)
        for v in rng.choice(boundary, size=b_deg, replace=False):
            G.add_edge(u, int(v), w=float(rng.uniform(0.8, 1.2)))
        i_targets = [x for x in interior if x != u]
        for v in rng.choice(i_targets, size=min(2, len(i_targets)), replace=False):
            G.add_edge(u, int(v), w=float(rng.uniform(0.5, 1.0)))
    return G, boundary, interior


def build_wormhole_graph(n: int, rng: np.random.Generator) -> Tuple[nx.Graph, List[int], List[int]]:
    # Two boundary rings connected by sparse bridge interior
    nb = n // 2
    half = nb // 2
    ni = n - nb
    G = nx.Graph()
    boundary = list(range(nb))
    interior = list(range(nb, n))
    G.add_nodes_from(range(n))
    ringA = boundary[:half]
    ringB = boundary[half:]
    for ring in (ringA, ringB):
        m = len(ring)
        for i in range(m):
            G.add_edge(ring[i], ring[(i + 1) % m], w=1.0)
    # bridge interior chain
    if interior:
        chain = interior
        for i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i + 1], w=float(rng.uniform(0.7, 1.1)))
        if ringA:
            G.add_edge(chain[0], ringA[len(ringA)//2], w=1.0)
        if ringB:
            G.add_edge(chain[-1], ringB[len(ringB)//2], w=1.0)
    # extra sparse links
    for u in interior:
        for v in rng.choice(boundary, size=1, replace=False):
            G.add_edge(u, int(v), w=float(rng.uniform(0.3, 0.8)))
    return G, boundary, interior


def entropy_proxy_from_laplacian(G: nx.Graph, boundary: List[int], subsets: List[List[int]]) -> np.ndarray:
    # Gaussian proxy: covariance ~ (L + eps I)^-1. Entropy proxy for subset A: logdet(C_A)
    L = nx.laplacian_matrix(G, nodelist=range(len(G))).astype(float).todense()
    L = np.asarray(L)
    eps = 1e-2
    C = np.linalg.pinv(L + eps * np.eye(L.shape[0]))
    vals = []
    for A in subsets:
        idx = np.array(A, dtype=int)
        CA = C[np.ix_(idx, idx)]
        sign, ld = np.linalg.slogdet(CA + 1e-8 * np.eye(len(idx)))
        vals.append(float(ld))
    return np.array(vals)


def mincut_proxy(G: nx.Graph, boundary: List[int], subsets: List[List[int]]) -> np.ndarray:
    # RT proxy: min cut between A and boundary\A, with capacities from edge weights.
    vals = []
    for A in subsets:
        Aset = set(A)
        Bset = set(boundary) - Aset
        H = nx.DiGraph()
        for u, v, d in G.edges(data=True):
            w = float(d.get('w', 1.0))
            H.add_edge(u, v, capacity=w)
            H.add_edge(v, u, capacity=w)
        s = len(G) + 10
        t = len(G) + 11
        BIG = 1e6
        for a in Aset:
            H.add_edge(s, a, capacity=BIG)
        for b in Bset:
            H.add_edge(b, t, capacity=BIG)
        cut, _ = nx.minimum_cut(H, s, t)
        vals.append(float(cut))
    return np.array(vals)


def sample_boundary_subsets(boundary: List[int], rng: np.random.Generator, k: int = 20) -> List[List[int]]:
    subsets = []
    m = len(boundary)
    for _ in range(k):
        size = int(rng.integers(max(2, m // 8), max(3, m // 2)))
        A = sorted(rng.choice(boundary, size=size, replace=False).tolist())
        subsets.append(A)
    return subsets


def fit_metrics(x: np.ndarray, y: np.ndarray, mode: str, n: int) -> Metric:
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
    yhat = slope * x + float(np.mean(y) - slope * np.mean(x))
    mae = float(np.mean(np.abs(y - yhat)))
    return Metric(mode=mode, n=n, pairs=len(x), corr=corr, slope=slope, mae=mae)


def run_mode(mode: str, n: int, seed: int) -> Metric:
    rng = np.random.default_rng(seed)
    if mode == 'disk':
        G, boundary, _ = build_disk_graph(n, rng)
    elif mode == 'wormhole':
        G, boundary, _ = build_wormhole_graph(n, rng)
    else:
        raise ValueError(mode)
    subsets = sample_boundary_subsets(boundary, rng, k=24)
    s_ent = entropy_proxy_from_laplacian(G, boundary, subsets)
    s_rt = mincut_proxy(G, boundary, subsets)
    return fit_metrics(s_rt, s_ent, mode, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['disk', 'wormhole', 'both'], default='both')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    modes = ['disk', 'wormhole'] if args.mode == 'both' else [args.mode]
    out = []
    for i, m in enumerate(modes):
        out.append(run_mode(m, n=args.n, seed=args.seed + i))

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date = datetime.now().strftime('%Y-%m-%d')

    data = {
        'timestamp': ts,
        'n': args.n,
        'seed': args.seed,
        'results': [m.__dict__ for m in out],
    }
    jpath = RUNS / f'{date}.json'
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        f'# Holo Repro Daily Report ({date})',
        '',
        f'- timestamp: {ts}',
        f'- graph size n: {args.n}',
        '',
        '| mode | corr(RT, EntropyProxy) | slope | MAE |',
        '|---|---:|---:|---:|',
    ]
    for m in out:
        lines.append(f'| {m.mode} | {m.corr:.4f} | {m.slope:.4f} | {m.mae:.4f} |')

    lines += [
        '',
        '## 今天做了什么',
        '- 运行了 disk + wormhole 两种几何图上的 CPU-only 复现实验。',
        '- 计算了 min-cut (RT proxy) 与 entropy proxy 的相关性、线性斜率与 MAE。',
        '',
        '## 和昨天相比（自动对比）',
    ]

    prev_files = sorted(RUNS.glob('*.json'))
    prev = None
    if len(prev_files) >= 2:
        prev = json.loads(prev_files[-2].read_text(encoding='utf-8'))

    if prev:
        prev_map = {r['mode']: r for r in prev.get('results', [])}
        for cur in out:
            p = prev_map.get(cur.mode)
            if not p:
                continue
            dc = cur.corr - float(p.get('corr', 0.0))
            dm = cur.mae - float(p.get('mae', 0.0))
            trend = '变好' if (dc >= 0 and dm <= 0) else '变差/混合'
            lines.append(f"- {cur.mode}: corr {dc:+.4f}, MAE {dm:+.4f} → {trend}")
    else:
        lines.append('- 首日或缺少前一日数据，暂无法自动对比。')

    lines += [
        '',
        '## 明天计划',
        '- 做 n=32/48/64 参数扫，观察相关性与误差对图规模的敏感性。',
        '- 增加随机种子重复，报告均值±方差，降低偶然性。',
    ]

    rpath = REPORTS / f'{date}.md'
    rpath.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
