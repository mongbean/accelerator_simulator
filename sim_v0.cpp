// sim_v0.cpp -- SystemC port of sim_v0.py (transaction-level, cycle-approximate)
//
// STATUS: NOT COMPILED OR RUN against a real SystemC library (none was available when this was
// written). Only the plain C++ was syntax-checked against a stub header. Expect to fix small
// SystemC API issues. Validate by matching sim_v0.py (see README_systemc.md).
//
// !! ALL HARDWARE NUMBERS ARE PLACEHOLDERS (defaults of struct Hw), NOT THE REAL CHIP. !!
//
// Structure (mirrors the three building blocks):
//   Fabric    : memory controllers (west edge, one per mesh row) + mesh links (XY routing)
//   Core      : SC_MODULE with two SC_THREADs
//                 fetch_proc   - DMA engine: issues tile fetches, limited by `credits` (= local buffers
//                                free = lookahead window)
//                 compute_proc - tensor unit: consumes arrived tiles, frees a credit when done
//   Sequencer : SC_MODULE that tiles each op, hands job lists to the cores, waits for a barrier
//               (all cores done + writeback), then adds vector-op and launch time.
// 1 cycle == 1 ns of simulated time (1 GHz); resolution is set to 1 ps so fractional cycles survive.
#include <systemc>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <utility>
#include <vector>
using namespace sc_core;

static inline double  cyc(const sc_time& t) { return t.to_seconds() * 1e9; }
static inline sc_time cyc2t(double c)       { return sc_time(c, SC_NS); }
static inline long    ceil_div(long a, long b) { return (a + b - 1) / b; }

struct Hw {                       // PLACEHOLDER parameters (same defaults as HW0 in sim_v0.py)
  int    X = 4, Y = 4;            // mesh size (cores = X*Y)
  int    A = 32;                  // tensor unit is A x A MACs/cycle
  long   L1 = 1L << 20;           // local memory per core (bytes)
  double mc_bw = 128.0;           // bytes/cycle per memory controller (one per mesh row)
  double link_bw = 128.0;         // bytes/cycle per mesh link
  int    dram_lat = 200;          // cycles
  int    hop_lat = 2;             // cycles per hop
  int    issue = 20;              // issue overhead per tile job (cycles)
  int    lanes = 32;              // vector lanes per core
  int    win = 4;                 // max tiles in flight per core (lookahead)
};

enum Bin { USEFUL, UNDERFILL, WAIT_DRAM, WAIT_DRAMQ, WAIT_NOC, WAIT_BUF, TAIL, VEC, LAUNCH, NBINS };
static const char* BIN_NAME[NBINS] = {"useful", "underfill", "dram", "dramQ", "noc", "buf", "tail", "vec", "launch"};
struct Stats { double v[NBINS] = {}; };

struct Job { int m, tk, tn; bool last; };

// ---------------------------------------------------------------- interconnect + DRAM -------
struct Fabric {
  const Hw& hw;
  std::vector<double> mc_free;                    // per row: time (cycles) controller is free
  std::vector<std::vector<double>> link_free;     // per row, per eastbound link
  explicit Fabric(const Hw& h)
      : hw(h), mc_free(h.Y, 0.0), link_free(h.Y, std::vector<double>((size_t)std::max(h.X - 1, 0), 0.0)) {}
  void reset(double t) {
    for (auto& v : mc_free) v = t;
    for (auto& r : link_free) for (auto& v : r) v = t;
  }
  struct Xfer { double start, arrive, mc_wait, link_wait, base; };
  // Reserve controller + links for a read of `bytes` to core (x,y). Called at sim time == t_ready,
  // so reservations happen in simulation-time order (this replaces the Python heap).
  Xfer read(int x, int y, double bytes, double t_ready) {
    Xfer r;
    double mc_ready = std::max(t_ready, mc_free[y]);
    r.mc_wait = mc_ready - t_ready;
    double lmax = 0.0;
    for (int l = 0; l < x; ++l) lmax = std::max(lmax, link_free[y][l]);
    r.start = std::max(mc_ready, lmax);
    r.link_wait = r.start - mc_ready;
    double dur_mc = bytes / hw.mc_bw;
    double dur_link = (x > 0) ? bytes / hw.link_bw : 0.0;
    double dur = std::max(dur_mc, dur_link);
    mc_free[y] = r.start + dur_mc;
    for (int l = 0; l < x; ++l) link_free[y][l] = r.start + dur_link;
    r.base = hw.dram_lat + (x + 1) * hw.hop_lat + dur;
    r.arrive = r.start + r.base;
    return r;
  }
};

// ---------------------------------------------------------------- core ----------------------
struct Pkt { double arrive, buf_wait, mc_wait, link_wait, base; };

SC_MODULE(Core) {
  int x, y;
  const Hw* hw;
  Fabric* fab;
  int* remaining;
  sc_event* all_done;

  std::vector<Job> jobs;          // set by Sequencer before start_ev
  int credits = 0;                // free local buffers (lookahead window)
  sc_event start_ev, credit_ev, arr_ev;
  std::deque<Pkt> arrivals;       // fetch -> compute hand-off
  double busy = 0.0, end_t = 0.0; // per-op results read by Sequencer
  Stats bins;                     // accumulated over the whole run

  SC_HAS_PROCESS(Core);
  Core(sc_module_name n, int x_, int y_, const Hw* h, Fabric* f, int* rem, sc_event* done)
      : sc_module(n), x(x_), y(y_), hw(h), fab(f), remaining(rem), all_done(done) {
    SC_THREAD(fetch_proc);
    SC_THREAD(compute_proc);
  }

  void fetch_proc() {             // DMA engine
    while (true) {
      wait(start_ev);
      double last_start = cyc(sc_time_stamp());
      for (size_t i = 0; i < jobs.size(); ++i) {
        while (credits == 0) wait(credit_ev);          // no free buffer -> stall (window limit)
        --credits;
        double t_ready = cyc(sc_time_stamp());
        const Job& j = jobs[i];
        double bytes = ((double)j.tk * j.tn + (double)j.m * j.tk) * 2.0;   // weights + activation slice
        Fabric::Xfer xf = fab->read(x, y, bytes, t_ready);
        Pkt p{xf.arrive, t_ready - last_start, xf.mc_wait, xf.link_wait, xf.base};
        arrivals.push_back(p);
        arr_ev.notify(SC_ZERO_TIME);
        double dt = xf.start - t_ready;                 // DMA engine busy until the transfer starts
        if (dt > 0) wait(cyc2t(dt));
        last_start = xf.start;
      }
    }
  }

  void compute_proc() {           // tensor unit (weight-stationary A x A array)
    const int A = hw->A;
    while (true) {
      wait(start_ev);
      double t0 = cyc(sc_time_stamp());
      double unit_free = t0;
      busy = 0.0;
      end_t = t0;
      for (size_t i = 0; i < jobs.size(); ++i) {
        while (arrivals.empty()) wait(arr_ev);
        Pkt p = arrivals.front();
        arrivals.pop_front();
        const Job& j = jobs[i];
        double now = cyc(sc_time_stamp());
        double cs = std::max(p.arrive, now);
        if (cs > now) wait(cyc2t(cs - now));            // stall until the tile has arrived
        double gap = cs - unit_free;
        long sub = ceil_div(j.tk, A) * ceil_div(j.tn, A);
        double cycles = (double)sub * std::max(j.m, A) + A + hw->issue;
        double useful = (double)j.m * j.tk * j.tn / ((double)A * A);
        wait(cyc2t(cycles));
        double ce = cyc(sc_time_stamp());
        unit_free = ce;
        busy += cycles + gap;
        bins.v[USEFUL] += useful;
        bins.v[UNDERFILL] += cycles - useful;
        double tot = p.buf_wait + p.mc_wait + p.link_wait + p.base;   // heuristic split of the stall
        if (gap > 0 && tot > 0) {
          bins.v[WAIT_BUF]   += gap * p.buf_wait / tot;
          bins.v[WAIT_DRAMQ] += gap * p.mc_wait / tot;
          bins.v[WAIT_NOC]   += gap * p.link_wait / tot;
          bins.v[WAIT_DRAM]  += gap * p.base / tot;
        }
        double e = ce;
        if (j.last) {                                    // output writeback (timed, not contended in v0)
          double bw = std::min(hw->mc_bw, x > 0 ? hw->link_bw : hw->mc_bw);
          e += (x + 1) * hw->hop_lat + (double)j.m * j.tn * 2.0 / bw;
        }
        end_t = std::max(end_t, e);
        ++credits;                                       // buffer freed
        credit_ev.notify(SC_ZERO_TIME);
      }
      if (--(*remaining) == 0) all_done->notify(SC_ZERO_TIME);
    }
  }
};

// ---------------------------------------------------------------- workload + tiler ----------
struct OpSpec { long m, K, N; int rep; int tk, tn; };
struct Workload {
  long M, am, L; int rep;
  std::vector<std::array<long, 3>> dense;
  std::vector<std::pair<double, double>> vec;            // (elements, cycles per element per lane)
};

static Workload make_workload(bool prefill, int n) {     // n = tokens (prefill) or batch (decode)
  Workload w;
  double soft;
  if (prefill) { w.M = n; w.am = 4L * n; w.rep = 8; w.L = n; soft = 32.0 * n * n; }
  else         { w.M = n; w.am = 4;      w.rep = 8 * n; w.L = 2048; soft = (double)n * 32 * w.L; }
  long M = w.M;
  w.dense = {{M, 4096, 6144}, {M, 4096, 4096}, {M, 4096, 28672}, {M, 14336, 4096}};
  w.vec = {{2.0 * M * 4096, 3}, {(double)M * 5120, 3}, {soft, 5}, {(double)M * 14336, 4}, {2.0 * M * 4096, 1}};
  return w;
}

static double vec_time(const Workload& w, const Hw& hw, bool fuse) {
  double nc = hw.X * hw.Y, cycv = 0, elems = 0;
  for (auto& it : w.vec) { cycv += it.first * it.second; elems += it.first; }
  cycv /= (nc * hw.lanes);
  double mem = fuse ? 0.0 : elems * 4.0 / (hw.mc_bw * hw.Y);
  return std::max(cycv, mem);
}

// Port of run_matmul's tiling: groups (rep, row-block, n-tile) are owned round-robin by cores.
static bool build_jobs(const OpSpec& op, int tm, const Hw& hw, std::vector<std::vector<Job>>& jobs, int& nbuf) {
  int nc = hw.X * hw.Y;
  long tk = std::min<long>(op.tk, op.K), tn = std::min<long>(op.tn, op.N), tmm = std::min<long>(tm, op.m);
  long nk = ceil_div(op.K, tk), nn = ceil_div(op.N, tn), nm = ceil_div(op.m, tmm);
  long wbytes = (tk * tn + tmm * tk) * 2;
  long acc = tmm * tn * 4;
  long room = (hw.L1 - acc) / wbytes;
  if (hw.L1 - acc < wbytes) return false;                // tile does not fit in local memory
  nbuf = (int)std::min<long>(hw.win, room);
  if (nbuf < 1) return false;
  jobs.assign(nc, std::vector<Job>());
  long g = 0;
  for (int r = 0; r < op.rep; ++r)
    for (long mb = 0; mb < nm; ++mb) {
      int mbs = (int)std::min(tmm, op.m - mb * tmm);
      for (long j = 0; j < nn; ++j) {
        int tnj = (int)std::min(tn, op.N - j * tn);
        int c = (int)(g % nc);
        ++g;
        for (long i = 0; i < nk; ++i)
          jobs[c].push_back(Job{mbs, (int)std::min(tk, op.K - i * tk), tnj, i == nk - 1});
      }
    }
  return true;
}

// ---------------------------------------------------------------- sequencer -----------------
struct RunCfg { bool prefill = false; int n = 32; int T = 128; int tm = 128; bool fuse = false; double launch = 0; };

SC_MODULE(Sequencer) {
  const Hw* hw; Fabric* fab; std::vector<Core*>* cores; int* remaining; sc_event* all_done; RunCfg cfg;
  bool infeasible = false; double total_T = 0, macs = 0; Stats bins;

  SC_HAS_PROCESS(Sequencer);
  Sequencer(sc_module_name n, const Hw* h, Fabric* f, std::vector<Core*>* c, int* rem, sc_event* d, RunCfg cf)
      : sc_module(n), hw(h), fab(f), cores(c), remaining(rem), all_done(d), cfg(cf) { SC_THREAD(run); }

  void run() {
    wait(SC_ZERO_TIME);                                   // let core threads reach wait(start_ev)
    int nc = hw->X * hw->Y;
    Workload w = make_workload(cfg.prefill, cfg.n);
    std::vector<OpSpec> ops;
    for (auto& d : w.dense) ops.push_back(OpSpec{d[0], d[1], d[2], 1, cfg.T, cfg.T});
    ops.push_back(OpSpec{w.am, 128, w.L, w.rep, 128, cfg.T});      // scores = Q @ K^T
    ops.push_back(OpSpec{w.am, w.L, 128, w.rep, cfg.T, 128});      // probs @ V
    for (auto& op : ops) {
      std::vector<std::vector<Job>> jobs; int nbuf = 0;
      if (!build_jobs(op, cfg.tm, *hw, jobs, nbuf)) { infeasible = true; break; }
      for (auto& cj : jobs) for (auto& j : cj) macs += (double)j.m * j.tk * j.tn;
      double t0 = cyc(sc_time_stamp());
      fab->reset(t0);
      *remaining = nc;
      for (int c = 0; c < nc; ++c) {
        (*cores)[c]->jobs = std::move(jobs[c]);
        (*cores)[c]->credits = nbuf;
        (*cores)[c]->arrivals.clear();
      }
      for (int c = 0; c < nc; ++c) (*cores)[c]->start_ev.notify();     // immediate: threads are waiting
      wait(*all_done);
      double tend = t0, busy_sum = 0;
      for (int c = 0; c < nc; ++c) { tend = std::max(tend, (*cores)[c]->end_t); busy_sum += (*cores)[c]->busy; }
      double T = tend - t0;
      bins.v[TAIL] += nc * T - busy_sum;                                // imbalance + writeback + idle cores
      double now = cyc(sc_time_stamp());
      if (tend > now) wait(cyc2t(tend - now));                          // barrier includes writeback
      total_T += T;
    }
    if (!infeasible) {
      double tv = vec_time(w, *hw, cfg.fuse);
      wait(cyc2t(tv)); total_T += tv; bins.v[VEC] += nc * tv;
      double tl = cfg.launch * (ops.size() + 1);
      if (tl > 0) wait(cyc2t(tl));
      total_T += tl; bins.v[LAUNCH] += nc * tl;
    }
    sc_stop();
  }
};

// ---------------------------------------------------------------- main ----------------------
int sc_main(int argc, char* argv[]) {
  sc_set_time_resolution(1, SC_PS);                       // must precede any sc_time use
  Hw hw; RunCfg cfg;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i]; const char* v = argv[i + 1];
    if (k == "--wl") cfg.prefill = (std::strcmp(v, "prefill") == 0);
    else if (k == "--n") cfg.n = std::atoi(v);
    else if (k == "--T") cfg.T = std::atoi(v);
    else if (k == "--tm") cfg.tm = std::atoi(v);
    else if (k == "--fuse") cfg.fuse = std::atoi(v) != 0;
    else if (k == "--launch") cfg.launch = std::atof(v);
    else if (k == "--X") hw.X = std::atoi(v);
    else if (k == "--Y") hw.Y = std::atoi(v);
    else if (k == "--A") hw.A = std::atoi(v);
    else if (k == "--L1") hw.L1 = std::atol(v);
    else if (k == "--mc_bw") hw.mc_bw = std::atof(v);
    else if (k == "--link_bw") hw.link_bw = std::atof(v);
    else if (k == "--dram_lat") hw.dram_lat = std::atoi(v);
    else if (k == "--hop_lat") hw.hop_lat = std::atoi(v);
    else if (k == "--issue") hw.issue = std::atoi(v);
    else if (k == "--lanes") hw.lanes = std::atoi(v);
    else if (k == "--win") hw.win = std::atoi(v);
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  Fabric fab(hw);
  int remaining = 0; sc_event all_done;
  std::vector<Core*> cores;
  for (int y = 0; y < hw.Y; ++y)
    for (int x = 0; x < hw.X; ++x) {
      char nm[32]; std::snprintf(nm, sizeof nm, "core_%d_%d", x, y);
      cores.push_back(new Core(nm, x, y, &hw, &fab, &remaining, &all_done));   // index = y*X + x
    }
  Sequencer seq("seq", &hw, &fab, &cores, &remaining, &all_done, cfg);
  sc_start();

  if (seq.infeasible) { std::printf("tile does not fit in local memory\n"); return 1; }
  int nc = hw.X * hw.Y;
  for (auto* c : cores) for (int b = 0; b < NBINS; ++b) seq.bins.v[b] += c->bins.v[b];
  double peak = (double)nc * hw.A * hw.A;
  std::printf("%s n=%d T=%d tm=%d win=%d  util=%.2f%%  cycles=%.4g\n", cfg.prefill ? "prefill" : "decode",
              cfg.n, cfg.T, cfg.tm, hw.win, 100.0 * seq.macs / (peak * seq.total_T), seq.total_T);
  double sum = 0;
  for (int b = 0; b < NBINS; ++b) {
    double share = seq.bins.v[b] / (nc * seq.total_T);
    sum += share;
    std::printf("  %-9s %5.1f%%\n", BIN_NAME[b], 100 * share);
  }
  std::printf("  (bins sum to %.2f%%; should be 100%%)\n", 100 * sum);
  return 0;
}
