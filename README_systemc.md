# SystemC port of sim_v0.py

**Status: not compiled or run against a real SystemC library.** The plain C++ passes `g++ -fsyntax-only`
against a stub header only. Expect small fixes when you first build with real SystemC.
All hardware numbers are placeholders (see `struct Hw`).

## Build
    export SYSTEMC_HOME=/path/to/systemc      # contains include/ and lib-linux64/ (or lib/)
    make
SystemC must be built with the same C++ standard as `CXXFLAGS` (-std=c++17 by default; adjust if needed).

## Run (options override the placeholder hardware)
    ./sim_v0 --wl decode  --n 1  --T 128
    ./sim_v0 --wl decode  --n 32 --T 256
    ./sim_v0 --wl prefill --n 512 --T 256
    ./sim_v0 --wl decode  --n 32 --T 256 --win 8 --link_bw 32

## Validate against the Python model (same defaults, fuse=0, launch=0)
| Command | Python util | Python cycles |
|---|---|---|
| --wl decode --n 1 --T 128 | 1.61% | 8.895e5 |
| --wl decode --n 32 --T 256 | 29.19% | 1.571e6 |
| --wl decode --n 32 --T 64 | 24.71% | 1.857e6 |
| --wl prefill --n 512 --T 256 | 95.89% | 7.244e6 |
| --wl prefill --n 512 --T 128 | 94.51% | 7.350e6 |
Target: within ~1-2 percentage points. Also check "bins sum to 100%".
Small differences are expected: when two cores request the same controller at the same simulated time,
the Python version orders them by core id while SystemC orders them by process scheduling.

## How it maps to the three building blocks
- Compute: `Core::compute_proc` (tensor unit) -- replace with your out-of-order vector+tensor model
- Memory: `Core::credits` (local buffers) and `Fabric` memory controllers
- Interconnect: `Fabric::read` link reservations -- replace with a router-level 2D mesh model
- Granularity knobs: `--T` (tile), `--tm` (token-row block), `--win` (lookahead), `--fuse`

## Known v0 simplifications (same as the Python model)
Ops run with a barrier between them; writebacks are timed but don't reserve links/controllers; no causal mask;
no KV-cache writes; static round-robin tile ownership; stall-cause split is a proportional heuristic.
