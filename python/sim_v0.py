#!/usr/bin/env python3
"""
sim_v0.py -- v0 granularity/utilization simulator (Python, event-ordered, transaction level)

!! ALL HARDWARE NUMBERS BELOW ARE PLACEHOLDERS, NOT THE REAL CHIP. Replace HW0 with the
   real parameter sheet. Results show TRENDS of the model, not predictions for any product. !!

What it models (one Llama 3.1 8B transformer layer, fp16):
  compute      : per-core tensor unit (A x A MACs/cycle, weight-stationary) + vector unit
  memory       : per-core local memory (L1) + DRAM behind memory controllers on the west edge
  interconnect : X x Y 2D mesh, XY routing, link serialization + hop latency, link contention
  scheduling   : weights split into (tk x tn) tiles, activations into tm-row blocks; each
                 (row-block, n-tile) group is owned by one core; prefetch depth = win.
Granularity knobs: tk, tn (weight tile), tm (token-row block), win (lookahead), fuse (vector ops).

Simplifications (v0): ops run one after another with a barrier; writebacks are timed but do not
reserve shared links/controllers; attention uses full SxS (no causal mask); no KV-cache writes.
"""
import heapq, math, sys, time

HW0 = dict(X=4, Y=4,            # mesh size -> 16 cores
           A=32,                # tensor unit is A x A MACs/cycle (1024)
           L1=1 << 20,          # local memory per core (bytes)
           mc_bw=128.0,         # bytes/cycle per memory controller (4 controllers -> 512 B/cycle)
           link_bw=128.0,       # bytes/cycle per mesh link
           dram_lat=200,        # cycles
           hop_lat=2,           # cycles per hop
           issue=20,            # instruction/issue overhead per tile job (cycles)
           lanes=32,            # vector lanes per core (elements/cycle)
           win=4)               # max tiles in flight per core (lookahead / prefetch depth)

ceil = lambda a, b: -(-a // b)
BIN_KEYS = ["useful", "tensor_underfill_fill_issue", "wait_dram_latency_bw", "wait_dram_queue",
            "wait_noc_contention", "wait_buffer_window", "tail_imbalance_writeback", "vector_ops",
            "launch_overhead"]


def run_matmul(m, K, N, rep, tk, tn, tm, hw):
    """One GEMM (m x K) @ (K x N), repeated `rep` times. Returns dict or None if tiles don't fit."""
    X, Y, A = hw["X"], hw["Y"], hw["A"]
    nc = X * Y
    tk, tn, tm = min(tk, K), min(tn, N), min(tm, m)
    nk, nn, nm = ceil(K, tk), ceil(N, tn), ceil(m, tm)
    wbytes = (tk * tn + tm * tk) * 2            # one in-flight tile: weights + activation slice
    acc = tm * tn * 4                           # fp32 accumulators stay in local memory
    nbuf = min(hw["win"], (hw["L1"] - acc) // wbytes)
    if nbuf < 1:
        return None                             # tile does not fit in local memory
    jobs = [[] for _ in range(nc)]
    g = 0
    for r in range(rep):
        for mb in range(nm):
            mbs = min(tm, m - mb * tm)
            for j in range(nn):
                tnj = min(tn, N - j * tn)
                c = g % nc
                g += 1
                for i in range(nk):
                    jobs[c].append((mbs, min(tk, K - i * tk), tnj, i == nk - 1))
    mc_free = [0.0] * Y
    link_free = [[0.0] * max(X - 1, 0) for _ in range(Y)]
    idx = [0] * nc
    last_start = [0.0] * nc
    unit_free = [0.0] * nc
    comp_end = [[] for _ in range(nc)]
    end = [0.0] * nc
    busy = [0.0] * nc                           # sum of compute cycles + stall gaps per core
    bins = {k: 0.0 for k in BIN_KEYS}
    macs = 0
    heap = [(0.0, c) for c in range(nc) if jobs[c]]
    heapq.heapify(heap)
    mc_bw, link_bw, dram_lat, hop_lat, issue = hw["mc_bw"], hw["link_bw"], hw["dram_lat"], hw["hop_lat"], hw["issue"]
    while heap:
        t_ready, c = heapq.heappop(heap)
        i = idx[c]
        mbs, tki, tnj, last = jobs[c][i]
        x, y = c % X, c // X
        b = (tki * tnj + mbs * tki) * 2
        # ---- DMA from memory controller (row y) across x links to core (x,y) ----
        mc_ready = max(t_ready, mc_free[y])
        mc_wait = mc_ready - t_ready
        lf = link_free[y]
        lmax = max(lf[:x]) if x > 0 else 0.0
        start = max(mc_ready, lmax)
        link_wait = start - mc_ready
        dur_mc = b / mc_bw
        dur_link = b / link_bw if x > 0 else 0.0
        dur = max(dur_mc, dur_link)
        mc_free[y] = start + dur_mc
        for l in range(x):
            lf[l] = start + dur_link
        base = dram_lat + (x + 1) * hop_lat + dur
        arrive = start + base
        buf_wait = t_ready - last_start[c]
        # ---- tensor unit ----
        cs = max(arrive, unit_free[c])
        gap = cs - unit_free[c]
        sub = ceil(tki, A) * ceil(tnj, A)
        cyc = sub * max(mbs, A) + A + issue     # weight-stationary: max(stream rows, weight load)
        useful = mbs * tki * tnj / (A * A)
        ce = cs + cyc
        unit_free[c] = ce
        comp_end[c].append(ce)
        last_start[c] = start
        busy[c] += cyc + gap
        bins["useful"] += useful
        bins["tensor_underfill_fill_issue"] += cyc - useful
        tot = buf_wait + mc_wait + link_wait + base
        if gap > 0 and tot > 0:                 # heuristic: split stall in proportion to its causes
            bins["wait_buffer_window"] += gap * buf_wait / tot
            bins["wait_dram_queue"] += gap * mc_wait / tot
            bins["wait_noc_contention"] += gap * link_wait / tot
            bins["wait_dram_latency_bw"] += gap * base / tot
        if last:
            wb = (x + 1) * hop_lat + mbs * tnj * 2 / min(mc_bw, link_bw if x > 0 else mc_bw)
            end[c] = max(end[c], ce + wb)
        else:
            end[c] = max(end[c], ce)
        macs += mbs * tki * tnj
        idx[c] += 1
        i2 = idx[c]
        if i2 < len(jobs[c]):
            tr = last_start[c]
            if i2 - nbuf >= 0:
                tr = max(tr, comp_end[c][i2 - nbuf])
            heapq.heappush(heap, (tr, c))
    T = max(end) if end else 0.0
    bins["tail_imbalance_writeback"] = nc * T - sum(busy)
    return dict(T=T, macs=macs, bins=bins)


def vec_time(items, hw, fuse):
    nc = hw["X"] * hw["Y"]
    cyc = sum(e * c for e, c in items) / (nc * hw["lanes"])
    mem = 0.0 if fuse else sum(e for e, c in items) * 4 / (hw["mc_bw"] * hw["Y"])  # read+write fp16
    return max(cyc, mem)


def workload(name):
    if name.startswith("prefill"):
        S = int(name.split("=")[1])
        M, am, rep, L = S, 4 * S, 8, S
        soft = 32 * S * S
    else:
        B = int(name.split("B=")[1].split(",")[0]); L = 2048
        M, am, rep = B, 4, 8 * B
        soft = B * 32 * L
    dense = [(M, 4096, 6144), (M, 4096, 4096), (M, 4096, 28672), (M, 14336, 4096)]
    vec = [(2 * M * 4096, 3), (M * 5120, 3), (soft, 5), (M * 14336, 4), (2 * M * 4096, 1)]
    return dict(M=M, am=am, rep=rep, L=L, dense=dense, vec=vec)


def run_layer(wl_name, T, hw, tm=128, fuse=False, launch=0):
    wl = workload(wl_name)
    ops = [(m, K, N, 1, T, T) for (m, K, N) in wl["dense"]]
    ops.append((wl["am"], 128, wl["L"], wl["rep"], 128, T))      # scores: Q @ K^T
    ops.append((wl["am"], wl["L"], 128, wl["rep"], T, 128))      # probs @ V
    nc = hw["X"] * hw["Y"]
    total_T, macs = 0.0, 0
    bins = {k: 0.0 for k in BIN_KEYS}
    for (m, K, N, rep, tk, tn) in ops:
        r = run_matmul(m, K, N, rep, tk, tn, tm, hw)
        if r is None:
            return None
        total_T += r["T"]; macs += r["macs"]
        for k, v in r["bins"].items():
            bins[k] += v
    tv = vec_time(wl["vec"], hw, fuse)
    total_T += tv
    bins["vector_ops"] += nc * tv
    tl = launch * (len(ops) + 1)          # per-kernel launch/dispatch cost (GPU-style; 0 for v0 default)
    total_T += tl
    bins["launch_overhead"] += nc * tl
    peak = nc * hw["A"] ** 2
    return dict(util=macs / (peak * total_T), cycles=total_T, macs=macs, bins=bins,
                share={k: v / (nc * total_T) for k, v in bins.items()})


def roofline_bound(wl_name, hw):
    wl = workload(wl_name)
    nc = hw["X"] * hw["Y"]
    macs = sum(m * K * N for m, K, N in wl["dense"]) + wl["rep"] * wl["am"] * 128 * wl["L"] * 2
    wbytes = sum(K * N for m, K, N in wl["dense"]) * 2 + wl["rep"] * 128 * wl["L"] * 2 * 2
    cyc = max(macs / (nc * hw["A"] ** 2), wbytes / (hw["mc_bw"] * hw["Y"]))
    return macs / (nc * hw["A"] ** 2 * cyc)


def fmt_bins(share):
    names = {"useful": "useful", "tensor_underfill_fill_issue": "underfill", "wait_dram_latency_bw": "dram",
             "wait_dram_queue": "dramQ", "wait_noc_contention": "noc", "wait_buffer_window": "buf",
             "tail_imbalance_writeback": "tail", "vector_ops": "vec", "launch_overhead": "launch"}
    return " ".join(f"{names[k]}={100*share[k]:4.1f}" for k in BIN_KEYS)


if __name__ == "__main__":
    t0 = time.time()
    hw = dict(HW0)
    print("PLACEHOLDER hardware:", hw, "\n")
    WL = ["decode,B=1", "decode,B=32", "prefill,S=512"]
    # per-workload hardware peak MAC/cycle = 16*32*32
    print("== E0 sanity ==")
    for w in WL:
        r = run_layer(w, 128, hw)
        s = sum(r["share"].values())
        print(f"{w:16s} util={100*r['util']:5.2f}%  roofline_bound={100*roofline_bound(w, hw):5.2f}%  "
              f"bins_sum={100*s:6.2f}%  ok={r['util'] <= roofline_bound(w, hw) + 1e-9 and abs(s-1) < 1e-6}")
    fast = dict(hw, mc_bw=hw["mc_bw"] * 2, link_bw=hw["link_bw"] * 2)
    a, b = run_layer("decode,B=1", 128, hw)["util"], run_layer("decode,B=1", 128, fast)["util"]
    print(f"decode B=1: 2x DRAM+link bandwidth -> util x{b/a:.2f} (memory-bound => ~2x expected)\n")

    print("== E1 utilization vs tile size T (tk=tn=T, tm=128, win=4, unfused vector ops) ==")
    sizes = [32, 64, 128, 256, 512]
    best = {}
    for w in WL:
        print(w)
        for T in sizes:
            r = run_layer(w, T, hw)
            if r is None:
                print(f"  T={T:4d}  does not fit in local memory"); continue
            print(f"  T={T:4d} util={100*r['util']:5.2f}%  cycles={r['cycles']:.3g}  [{fmt_bins(r['share'])}]")
            if w not in best or r["util"] > best[w][1]:
                best[w] = (T, r["util"])
        sys.stdout.flush()

    print("\n== E2 lookahead window (win) at each workload's best T ==")
    for w in WL:
        T = best[w][0]
        row = []
        for win in [1, 2, 4, 8]:
            r = run_layer(w, T, dict(hw, win=win))
            row.append(f"win={win}: {'n/a' if r is None else f'{100*r['util']:.2f}%'}")
        print(f"{w:16s} T={T:4d}  " + "  ".join(row))

    print("\n== E3 token-row block tm (prefill only, best T) ==")
    T = best["prefill,S=512"][0]
    for tm in [32, 64, 128, 256]:
        r = run_layer("prefill,S=512", T, hw, tm=tm)
        print(f"  tm={tm:4d}  " + ("n/a (does not fit)" if r is None else f"util={100*r['util']:.2f}%"))

    print("\n== E4 fused vs unfused vector ops (best T) ==")
    for w in WL:
        T = best[w][0]
        u, f = run_layer(w, T, hw, fuse=False), run_layer(w, T, hw, fuse=True)
        print(f"{w:16s} T={T:4d} unfused={100*u['util']:.2f}%  fused={100*f['util']:.2f}%")
    print(f"\nruntime {time.time()-t0:.0f}s")
