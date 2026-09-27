/* The toy Landscape and both optimizers, ported to run in the browser.
 *
 * Source of truth is the Analysis project:
 *   experiments/joint_optimization/landscape.py    the surface and the climb
 *   experiments/joint_optimization/optimizers.py   climb, designer_propose, controller_act
 *   experiments/joint_optimization/sweep.py        the step loop
 *
 * Only the SHAPE of those formulae is duplicated here. Every number comes from
 * landscape-constants.json, generated from the Python modules by scripts/emit-constants.py, so
 * retuning the landscape upstream cannot leave this page showing a surface nobody tuned.
 *
 * The page is always N=1: one design coordinate and one action coordinate, a surface. The Python
 * is batched over a run dimension and over N; here there is exactly one Pair in 2D, so both are
 * dropped and the sample clouds are plain Float64Arrays.
 *
 * A Landscape is one draw, fixed by its seed. The page opens on the draw the constants file
 * carries -- torch's generator cannot be replayed here, so that one is emitted value for value --
 * and Reset draws a fresh one from the same distributions with a seeded generator of its own.
 */

(function (global) {
  "use strict";

  // --- random numbers -----------------------------------------------------------------------

  /** mulberry32: a seeded uniform in [0, 1). Only draws need to be reproducible from a seed; the
   *  per-step sampling noise stays on Math.random. */
  function seeded(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Box-Muller normals off a uniform source. Each draw makes two; the second is kept for the next
   *  call rather than thrown away, halving the log/sqrt/random cost. */
  function normals(uniform) {
    let spare = 0, haveSpare = false;
    return function () {
      if (haveSpare) {
        haveSpare = false;
        return spare;
      }
      let u = 0;
      while (u === 0) u = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      const th = 2 * Math.PI * uniform();
      spare = r * Math.sin(th);
      haveSpare = true;
      return r * Math.cos(th);
    };
  }
  const randn = normals(Math.random);

  // --- the landscape ------------------------------------------------------------------------

  /** A fresh draw, as landscape.Landscape.__init__ makes one at N=1: per-coordinate constants
   *  `C` (16 rows x 2 columns, design then action, in landscape.ROWS order, amplitudes already
   *  normalized) and a Haar-uniform orthogonal `Q` (2x2). */
  function draw(k, seed) {
    const u = seeded(seed), g = normals(u);
    const logn = ([base, scale]) => base * Math.exp(scale * g());
    const pair = (fn) => { const v0 = fn(); return [v0, fn()]; };
    const fb = pair(() => Math.min(k.F_B_MAX, logn(k.F_B)));
    const fm = pair(() => logn(k.F_M));
    const rw = pair(() => logn(k.R_W));
    const fw = [fm[0] * rw[0], fm[1] * rw[1]];
    const fs = pair(() => logn(k.F_S));
    const ns = pair(() => Math.max(2, Math.round(logn(k.N_S))));
    const fp = pair(() => logn(k.F_P));
    const phases = Array.from({ length: 5 }, () => pair(() => 2 * Math.PI * u()));
    // product amplitudes one per dimension, separable one per coordinate
    const [ab, am, aw, as] = [k.B, k.M, k.W, k.S].map((base) => logn([base, k.A_SCALE]));
    const ap = pair(() => logn([k.P, k.A_SCALE]));
    // theoretical range from this draw's own amplitudes, mapped to [-1, 1]
    const { lo: lo0, hi: hi0 } = range(k.MOD, ab, am, aw, as);
    const hi = hi0 + ap[0] + ap[1];
    const lo = lo0 - ap[0] - ap[1];
    const kk = 2 / (hi - lo);
    const both = (v) => { const r = Math.sqrt(kk * v); return [r, r]; };
    const C = [fb, fm, fw, fs, ns, fp, ...phases,
               both(ab), both(am), both(aw), both(as), [kk * ap[0], kk * ap[1]]];
    // QR of a Gaussian with R's diagonal positive is Gram-Schmidt on its columns.
    const g00 = g(), g10 = g(), g01 = g(), g11 = g();
    const n0 = Math.hypot(g00, g10);
    const q0 = [g00 / n0, g10 / n0];
    const dot = q0[0] * g01 + q0[1] * g11;
    const r0 = g01 - dot * q0[0], r1 = g11 - dot * q0[1];
    const n1 = Math.hypot(r0, r1);
    const Q = [[q0[0], r0 / n1], [q0[1], r1 / n1]];
    return { seed, C, Q };
  }

  /** The product terms' range at one dimension. It is bilinear in the basin factor b in [-1, 1] and
   *  the texture t in [-(M+W), M+W+S] (the spike is >= 0) -- b*B + (1 + MOD b) t -- so the extremes
   *  sit at the corners. Works in raw or normalized amplitudes alike. */
  function range(mod, B, M, W, S) {
    let lo = Infinity, hi = -Infinity;
    for (const b of [-1, 1]) {
      for (const t of [-(M + W), M + W + S]) {
        const v = b * B + (1 + mod * b) * t;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    return { lo, hi };
  }

  /** Integer powers by squaring: Math.pow with an integer exponent was a third of a step. */
  function powInt(x, n) {
    let r = 1;
    for (let e = n; e > 0; e >>= 1) {
      if (e & 1) r *= x;
      x *= x;
    }
    return r;
  }

  /**
   * One draw made callable: `f(d, a)` and `climb(x, other, side)`, both exactly landscape.py's at
   * N=1. `side` 0 climbs a design `x` with the action `other` held fixed; `side` 1 an action.
   *
   * `C` holds amplitudes already normalized -- sqrt(k * amplitude) on both coordinates of a product
   * term, k * amplitude for the separable one -- so the offset and the Lipschitz bound both come
   * back out of it without the raw draw.
   */
  function make(k, spec) {
    const [fb, fm, fw, fs, ns, fp, pb, pm, pw, ps, pp, cb, cm, cw, cs, ap] = spec.C;
    const Q = spec.Q;
    // the product term's normalized amplitude, and its sqrt share on each factor
    const Ab = cb[0] * cb[1], Am = cm[0] * cm[1], Aw = cw[0] * cw[1], As = cs[0] * cs[1];
    const MOD = k.MOD;
    const offset = -(range(MOD, Ab, Am, Aw, As).lo - ap[0] - ap[1]) - 1; // -lo_normalized - 1

    // Lipschitz bound of the gradient (see landscape.py), in normalized units. A dimension is
    // b (B + MOD t) + t, so by the product rule its Hessian is
    // <= (B + MOD |t|) H_b + 2 MOD G_b G_t + (1 + MOD) H_t.
    const block = (Bd, Ba, Ad, Aa) => Math.max(Bd, Ba) + Ad * Aa;
    const sine = (f) => block(f[0] ** 2, f[1] ** 2, f[0], f[1]);
    const sd = fs[0] * Math.sqrt(ns[0]), sa = fs[1] * Math.sqrt(ns[1]);
    const H_t = Am * sine(fm) + Aw * sine(fw) +
                As * block(ns[0] * fs[0] ** 2, ns[1] * fs[1] ** 2, sd, sa);
    const G_t = Am * Math.hypot(fm[0], fm[1]) + Aw * Math.hypot(fw[0], fw[1]) + As * Math.hypot(sd, sa);
    const per = (Ab + MOD * (Am + Aw + As)) * sine(fb) + 2 * MOD * Math.hypot(fb[0], fb[1]) * G_t +
                (1 + MOD) * H_t;
    const L = per + Math.max(ap[0] * fp[0] ** 2, ap[1] * fp[1] ** 2);
    const eta = k.step / L; // / N, and N = 1
    const steps = k.climbSteps;
    const n0 = ns[0], n1 = ns[1];

    function f(d, a) {
      const z0 = Q[0][0] * d + Q[0][1] * a, z1 = Q[1][0] * d + Q[1][1] * a;
      const basin = Math.sin(fb[0] * z0 + pb[0]) * Math.sin(fb[1] * z1 + pb[1]);
      const texture = Am * Math.sin(fm[0] * z0 + pm[0]) * Math.sin(fm[1] * z1 + pm[1]) +
                      Aw * Math.sin(fw[0] * z0 + pw[0]) * Math.sin(fw[1] * z1 + pw[1]) +
                      As * powInt(Math.abs(Math.sin(fs[0] * z0 + ps[0])), n0) *
                           powInt(Math.abs(Math.sin(fs[1] * z1 + ps[1])), n1);
      // the basin scales the local texture: taller hills on its crests, shallower in its valleys
      const perT = Ab * basin + (1 + MOD * basin) * texture;
      const sep = ap[0] * Math.sin(fp[0] * d + pp[0]) + ap[1] * Math.sin(fp[1] * a + pp[1]);
      return perT + sep + offset;
    }

    // The product amplitudes multiply out to A per term whichever factor carries them, so the
    // gradient uses the whole A on the factor being differentiated and none on its partner.
    function climb(x, other, side) {
      const q0 = Q[0][side], q1 = Q[1][side];
      const o = 1 - side;
      const zo0 = Q[0][o] * other, zo1 = Q[1][o] * other;
      const fpx = fp[side], ppx = pp[side], apx = ap[side];
      for (let t = 0; t < steps; t++) {
        const z0 = zo0 + x * q0, z1 = zo1 + x * q1;
        const ub0 = fb[0] * z0 + pb[0], ub1 = fb[1] * z1 + pb[1];
        const um0 = fm[0] * z0 + pm[0], um1 = fm[1] * z1 + pm[1];
        const uw0 = fw[0] * z0 + pw[0], uw1 = fw[1] * z1 + pw[1];
        const us0 = fs[0] * z0 + ps[0], us1 = fs[1] * z1 + ps[1];
        const sb0 = Math.sin(ub0), sb1 = Math.sin(ub1);
        const sm0 = Math.sin(um0), sm1 = Math.sin(um1);
        const sw0 = Math.sin(uw0), sw1 = Math.sin(uw1);
        const s0 = Math.sin(us0), s1 = Math.sin(us1);
        const b0 = powInt(Math.abs(s0), n0 - 2), b1 = powInt(Math.abs(s1), n1 - 2);
        const spk0 = b0 * s0 * s0, spk1 = b1 * s1 * s1;
        const ds0 = fs[0] * n0 * b0 * s0 * Math.cos(us0);
        const ds1 = fs[1] * n1 * b1 * s1 * Math.cos(us1);
        const texture = Am * sm0 * sm1 + Aw * sw0 * sw1 + As * spk0 * spk1;
        const envelope = 1 + MOD * sb0 * sb1;
        // product rule: basin term, envelope's slope times texture, envelope times texture's slope
        const gz0 = fb[0] * Math.cos(ub0) * sb1 * (Ab + MOD * texture) +
                    envelope * (Am * fm[0] * Math.cos(um0) * sm1 + Aw * fw[0] * Math.cos(uw0) * sw1 +
                                As * ds0 * spk1);
        const gz1 = fb[1] * Math.cos(ub1) * sb0 * (Ab + MOD * texture) +
                    envelope * (Am * fm[1] * Math.cos(um1) * sm0 + Aw * fw[1] * Math.cos(uw1) * sw0 +
                                As * ds1 * spk0);
        x += eta * (gz0 * q0 + gz1 * q1 + apx * fpx * Math.cos(fpx * x + ppx));
      }
      return x;
    }

    return { f, climb, L, offset, spec };
  }

  /** The draw's value range over [lo, hi]^2, on an n x n grid -- what the colourscale and the
   *  reward axis are pinned to, so neither re-normalises as the window slides. */
  function valueRange(land, lo, hi, n = 256) {
    let zLo = Infinity, zHi = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = lo + ((hi - lo) * i) / (n - 1);
      for (let j = 0; j < n; j++) {
        const v = land.f(d, lo + ((hi - lo) * j) / (n - 1));
        if (v < zLo) zLo = v;
        if (v > zHi) zHi = v;
      }
    }
    return { zLo, zHi };
  }

  /* The controller's climb, tabulated. It is a pure function of (design, action), and the
   * controller runs N*N of them a step at several gradient steps each -- too many to run exactly at
   * any playback speed -- so it is looked up instead: the displacement a climb makes, on a grid of
   * spacing `h`, bilinearly interpolated and added to the sample. Displacement rather than the
   * endpoint, so the lookup is exact wherever the grid is, and smooth inside a hill.
   *
   * The grid is unbounded -- the basins are wide enough that a Pair walks far outside the starting
   * box -- and filled in TILE x TILE tiles on first use, so a fresh draw costs no stall: the first
   * steps pay only for the ground the walk actually visits. Past MAX_TILES it starts over rather
   * than growing without end. The designer's climb, a mere P_d a step, always runs exactly. */
  const TILE = 64;
  const MAX_TILES = 2048;
  function ClimbTable(land, h = 1 / 64) {
    const tiles = new Map();
    let lastKey = NaN, lastTile = null;
    const tile = (ti, tj) => {
      const key = ti * 1048576 + tj;
      if (key === lastKey) return lastTile;
      let t = tiles.get(key);
      if (!t) {
        if (tiles.size >= MAX_TILES) tiles.clear();
        t = new Float32Array(TILE * TILE);
        for (let u = 0; u < TILE; u++) {
          const d = (ti * TILE + u) * h;
          for (let v = 0; v < TILE; v++) {
            const a = (tj * TILE + v) * h;
            t[u * TILE + v] = land.climb(a, d, 1) - a;
          }
        }
        tiles.set(key, t);
      }
      lastKey = key;
      lastTile = t;
      return t;
    };
    const at = (i, j) => tile(i >> 6, j >> 6)[(i & 63) * TILE + (j & 63)]; // TILE = 64
    this.action = function (a, d) {
      const x = d / h, y = a / h;
      const i = Math.floor(x), j = Math.floor(y), u = x - i, v = y - j;
      let p00, p01, p10, p11;
      if ((i & 63) !== 63 && (j & 63) !== 63) {
        const t = tile(i >> 6, j >> 6), q = (i & 63) * TILE + (j & 63);
        p00 = t[q]; p01 = t[q + 1]; p10 = t[q + TILE]; p11 = t[q + TILE + 1];
      } else {
        p00 = at(i, j); p01 = at(i, j + 1); p10 = at(i + 1, j); p11 = at(i + 1, j + 1);
      }
      const top = p00 + v * (p01 - p00);
      return a + top + u * (p10 + v * (p11 - p10) - top);
    };
  }

  // --- the pair -----------------------------------------------------------------------------

  /* CMA-style log weights over `m` elites, descending, summing to 1. Cached: `m` only changes
   * when a population size does, which is never during a run. */
  const _weightCache = new Map();
  function rankWeights(m) {
    let w = _weightCache.get(m);
    if (w) return w;
    w = new Float64Array(m);
    let sum = 0;
    const lead = Math.log(m + 0.5);
    for (let i = 0; i < m; i++) {
      w[i] = lead - Math.log(i + 1);
      sum += w[i];
    }
    for (let i = 0; i < m; i++) w[i] /= sum;
    _weightCache.set(m, w);
    return w;
  }

  /* Step `mu` toward the rank-weighted mean of the best `topPercent` of `x`. The result stays
   * inside the convex hull of the samples, so the update cannot diverge however wide the spread.
   *
   * `P` is passed rather than read off `x.length` because the sample buffers are allocated once
   * at the largest population the widget offers and used as a prefix, so their length is the
   * maximum rather than the count in play. */
  const _eliteCache = new Map();
  function recombine(mu, x, v, P, alpha, topPercent) {
    const m = Math.max(Math.floor(P * topPercent), 1);
    // Partial selection of the top m rather than a full sort: P is up to 16384 every step and the
    // elite fraction is 1%, so one insertion pass over a length-m buffer beats sorting comfortably.
    // The buffers are reused per m, like the weights: allocating them twice a step was garbage.
    let buf = _eliteCache.get(m);
    if (!buf) _eliteCache.set(m, (buf = { idx: new Int32Array(m), v: new Float64Array(m) }));
    const elite = buf.idx;
    const eliteV = buf.v.fill(-Infinity);
    for (let i = 0; i < P; i++) {
      const vi = v[i];
      if (vi <= eliteV[m - 1]) continue;
      let k = m - 1;
      while (k > 0 && eliteV[k - 1] < vi) {
        eliteV[k] = eliteV[k - 1];
        elite[k] = elite[k - 1];
        k--;
      }
      eliteV[k] = vi;
      elite[k] = i;
    }
    const w = rankWeights(m);
    let target = 0;
    for (let i = 0; i < m; i++) target += x[elite[i]] * w[i];
    return mu + alpha * (target - mu);
  }

  /**
   * One Designer/Controller Pair walking one Landscape.
   *
   * `params` is read fresh at the top of every step, so a slider moved mid-walk takes effect on
   * the very next samples rather than at the next reset. That includes the population `p.P`,
   * which sets both P_d and P_a: the buffers below are allocated once at `pMax` and used as a
   * prefix, so changing it costs no allocation and cannot stall the walk.
   */
  function Pair(consts, spec, pMax) {
    let land = null, table = null;
    const P_MAX = pMax || Math.max(consts.run.P_d, consts.run.P_a);
    let P_d = consts.run.P_d;
    let P_a = consts.run.P_a;

    const D = new Float64Array(P_MAX); // designer proposals
    const A = new Float64Array(P_MAX * P_MAX); // actions played on them
    const V = new Float64Array(P_MAX * P_MAX); // their values
    const vd = new Float64Array(P_MAX); // mean value per design, this step

    this.mu_d = 0;
    this.mu_a = 0;
    this.t = 0;
    // the centre the *previous* batch was proposed from: what the controller has recently played
    let mu_d_prev = 0, proposedFrom = 0;

    /** Walk a different draw from here on. */
    this.setLandscape = function (s) {
      land = make(consts.landscape, s);
      table = new ClimbTable(land);
    };
    this.setLandscape(spec);

    this.reset = function (d0, a0) {
      this.mu_d = d0;
      this.mu_a = a0;
      mu_d_prev = proposedFrom = d0;
      this.t = 0;
    };

    /* Partially climb a sample toward the end of its uphill climb: `g` is the generalization
     * radius. g -> inf lands on the target, g -> 0 never moves. Exploration, the probability of
     * leaving a sample raw, is checked by the caller first, so e = 1 disables the climb entirely --
     * which is why exploration gates generalization rather than sitting beside it -- and a raw
     * sample never pays for the climb it skips. */
    const climb = (x, target, dist2, g) => {
      const gg = Math.max(g, 1e-8);
      const w = Math.exp((-0.5 * dist2) / (gg * gg));
      return x + w * (target - x);
    };

    /** Advance one controller step. `p` carries the six radii, the sampling ratio `k`, and the
     *  population `P`. */
    this.step = function (p) {
      const k = p.k;

      // A population change lands mid-window: `D` holds P_d_old proposals, which mean nothing at
      // the new size, so a fresh batch is proposed -- off-cycle by at most k-1 steps, invisible at
      // any speed.
      const P = p.P || P_d;
      const resized = P !== P_d;
      if (resized) {
        P_d = P;
        P_a = P;
      }

      // The designer proposes a fresh batch every k steps but learns from every step's evaluation
      // of it: the ratio decides how often designs are proposed, never how often either optimizer
      // updates.
      if (this.t % k === 0 || resized) {
        mu_d_prev = proposedFrom;
        proposedFrom = this.mu_d;
        // The designer is unconditional: its samples sit at (d, mu_a) and its centre at
        // (mu_d, mu_a), so the action term of the joint distance vanishes by construction. It
        // climbs f(., mu_a) -- the controller's CURRENT competence, one iteration stale -- not
        // the oracle marginal, so its view of design quality stays mediated by the controller.
        for (let i = 0; i < P_d; i++) {
          const d = this.mu_d + p.sig_d * randn();
          D[i] = Math.random() < p.e_d ? d
            : climb(d, land.climb(d, this.mu_a, 0), (d - this.mu_d) ** 2, p.g_d);
        }
      }

      // The sampled action still decides WHERE the climb starts, so controller spread keeps its
      // job; only the strength of the climb is attenuated, by joint distance from
      // (mu_d_prev, mu_a). That is what makes designer spread compete with controller
      // generalization: designs far from what the controller has recently seen get played badly.
      const f = land.f;
      for (let i = 0; i < P_d; i++) {
        const d = D[i];
        const dd = (d - mu_d_prev) ** 2;
        let sum = 0;
        for (let j = 0; j < P_a; j++) {
          const a0 = this.mu_a + p.sig_c * randn();
          const a = Math.random() < p.e_c ? a0
            : climb(a0, table.action(a0, d), dd + (a0 - this.mu_a) ** 2, p.g_c);
          const v = f(d, a);
          const q = i * P_a + j;
          A[q] = a;
          V[q] = v;
          sum += v;
        }
        vd[i] = sum / P_a;
      }

      this.mu_d = recombine(this.mu_d, D, vd, P_d, p.alpha, p.topPercent);
      this.mu_a = recombine(this.mu_a, A, V, P_d * P_a, p.alpha, p.topPercent);
      this.t++;
      // Sample q is design q / P_a (rounded down) playing action q -- D is returned rather than a
      // design per sample, which was N*N writes a step for a picture that draws a few dozen.
      return { d: D, a: A, v: V, n: P_d * P_a, perDesign: P_a };
    };
  }

  global.Landscape = { draw, make, valueRange, ClimbTable, Pair };
})(typeof window !== "undefined" ? window : self); // self: the walk runs this in a Worker
