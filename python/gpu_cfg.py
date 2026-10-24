from python.sim_v0 import *
# A100-like PLACEHOLDER mapping (public spec figures from memory; verify): 108 SMs, ~1.41 GHz,
# 312 TFLOP/s fp16 tensor = 1024 MAC/cycle/SM, HBM ~1.55 TB/s = ~1100 B/cycle, ~160 KB smem/SM.
gpu = dict(X=9, Y=12, A=32, L1=160*1024, mc_bw=1100/12, link_bw=256.0, dram_lat=500, hop_lat=1,
           issue=20, lanes=64, win=4)
LAUNCH = 7000   # ~5 us per kernel at 1.41 GHz (placeholder; CUDA graphs would cut this)
peak_tflops = 108*1024*2*1.41e9/1e12
print(f"A100-like peak = {peak_tflops:.0f} TFLOP/s; ridge = {108*1024/ (1100):.0f} MAC/B = {2*108*1024/1100/2:.0f} FLOP/B (fp16: {108*1024*2/1100:.0f} FLOP/B)\n")
print("== decode, context 2048: best tile per batch; util = tensor MACs / peak ==")
for B in [1,2,4,8,14,16,32,64]:
    w=f"decode,B={B}"
    best=None
    for T in [32,64,128,256]:
        r=run_layer(w,T,gpu,launch=LAUNCH)
        if r and (best is None or r["util"]>best[1]["util"]): best=(T,r)
    T,r=best; rb=roofline_bound(w,gpu)
    r0=run_layer(w,T,gpu,launch=0)
    print(f"B={B:3d} T={T:3d} util={100*r['util']:5.2f}%  (no-launch {100*r0['util']:5.2f}%)  roofline_bound={100*rb:5.2f}%  [{fmt_bins(r['share'])}]")
