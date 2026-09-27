/* The Landscape walk: one Designer/Controller Pair walking the Landscape, live in the page.
 *
 * The simulation is in landscape.js, run in walk-worker.js; this file is only the widget -- the
 * Plotly surface, the nine sliders, the window that follows the Pair, and the animation loop. The
 * worker is what keeps the loop at the display's frame rate: an expensive setting costs the walk
 * pace, never the picture smoothness.
 *
 * The run never ends. Sliders are read fresh at the top of every step, so moving one takes effect
 * on the next samples rather than at the next Reset, and the Trail keeps the last TRAIL_STEPS
 * steps only -- so it says what the Pair is doing now rather than everything it has ever done.
 */

(function () {
  "use strict";

  // Surface resolution, and how much wider than the window the mesh is built. The margin is what
  // lets the window slide smoothly: the ranges move every frame while the mesh is rebuilt only
  // when the drift is about to run off its edge, and because the rebuilt mesh covers the whole
  // visible window either way, the rebuild itself is invisible. MESH/MESH_MARGIN is the effective
  // resolution inside the window, kept at the 224 of the path-taken figure in analysis.py.
  const MESH = 288;
  const MESH_MARGIN = 1.3;
  const REMESH_AT = 0.75; // rebuild once this much of the margin has been eaten
  const TRAIL_STEPS = 4000; // Trail length, in steps

  // Population detents. One number sets both P_d and P_a, so a step costs N*N evaluations -- 64
  // is what the sweep runs and what this page opens on. The elite fraction stays at the run's own
  // 1% at every N, exactly as upstream, which is why a small population is visibly jumpier and
  // not merely cheaper: the controller recombines over 1 elite at N=8 and 163 at N=128.
  const POPS = [8, 16, 32, 64, 128];
  const POP_DEFAULT = 3;
  const P_MAX = POPS[POPS.length - 1];

  // Playback speed, as multiples of STEPS_PER_SEC. A target rather than a promise: N=128 at 16x
  // asks for 960 steps a second, which is more than the worker can run, so it delivers what it
  // can. 60 is the pace the walk had when it advanced per drawn frame on a 60Hz screen -- and
  // measured in seconds it no longer runs twice as fast on a 120Hz one.
  const SPEEDS = [1, 2, 4, 8, 16];
  const SPEED_DEFAULT = 2;
  const STEPS_PER_SEC = 60;

  // The Sample wake. One generation per worker tick, faded by wall clock, so the tail is 0.75s
  // whatever the playback speed -- counted in steps instead, it would run 0.75s at 1x and 45ms at
  // 16x. The dot count, WAKE_PER_POP, lives in walk-worker.js, which picks the dots.
  // A generation thins as well as dims: drawing every dot for the whole tail would leave a
  // slab on the surface going grey, rather than a cloud evaporating.
  //
  // The tail is only as legible as it is sparse. A dot has 0.75s to dim, but that fade is
  // invisible if the generation behind it arrives thick enough to paint over it -- what reads
  // as a cloud evaporating at a dozen dots a step reads as a solid front at forty. So the
  // count is set low enough that a single dot can be followed all the way out, and nothing
  // caps it: the wake is the one place N is visible as density rather than as frame rate, and
  // a cap flattened exactly the top half of that range. It stays quadratic because the Pair
  // really does evaluate N*N samples a step. The floor of one dot is the only backstop -- the
  // smallest population divides to less than half a dot and would otherwise draw nothing.
  //
  // Every dot now lives the full tail. A generation used to thin as well as dim, dropping dots
  // from its end as it aged, which was how 120 dots over 20 frames stayed affordable -- but
  // thinning is deletion, and a deleted dot pops rather than fades. It killed the effect it was
  // paired with: at sixteen dots a generation the last one to be drawn was cut after two frames,
  // still at 0.81 alpha, and only the first dot of each generation ever reached zero. Drawing
  // them all costs every dot for the whole tail instead of about half that, which the lower count pays
  // for several times over.
  const WAKE_MS = 750;
  // A dot shrinks as well as dims. Alpha alone is measurably correct -- a dot's colour ramps
  // 0.85 to 0 across the tail and 99% of them reach 0 before they are dropped -- and it still
  // read as popping, because a 1.6px point sprite has almost no perceptual range in alpha: it
  // is either lit or it is not. Size is the channel that does not depend on how the GL layer
  // blends a sub-pixel sprite, so the two run together and the dot visibly goes to nothing.
  const WAKE_SIZE_NEW = 3.2;
  const WAKE_SIZE_OLD = 0.3;
  // Reset frames the window from the spread, but compressed. Spread runs over a 130x range, and
  // framing it proportionally makes the narrow settings a pinhole and the wide ones a blur; the
  // power turns that into under 3x, enough to read a narrow run as narrow without the surface
  // lurching every time the slider moves. Chosen so nothing clamps: the whole slider still moves
  // the framing, rather than the bottom half of it sitting flat on HALF_MIN.
  const ZOOM_REF_SIG = 2.0; // the spread the framing is calibrated on (matched radii)
  const ZOOM_REF_HALF = 2.0; // ...and the half-width it gets
  const ZOOM_POWER = 0.2; // below 1 compresses; 1 would be proportional
  const HALF_MIN = 0.8; // backstops only -- the mapping above stays inside them
  const HALF_MAX = 3.0;
  // How the window chases the Pair. Seconds rather than a per-frame rate, and driven by measured
  // elapsed time, so the follow feels the same at 60fps and at the 13fps a big population and a
  // fast playback speed can cost.
  const FOLLOW_TAU = 0.45;
  const DT_MAX = 1 / 20; // a tab left in the background comes back with a huge dt; ignore it

  // Gain on a rotate drag, against gl-plot3d's own default of 1. Needed only since the loop
  // stopped writing the camera back every frame: that write round-tripped the turntable through
  // an eye/center/up matrix, which does not preserve the twist, so it was quietly eating part of
  // every drag and the default read as calm. With the drag left alone the full gain arrives, and
  // on a plot this size a flick crossed the whole surface.
  const ROTATE_SPEED = 0.5;

  // How close the camera opens, against the path-taken figure's own viewpoint. The eye is
  // divided by this and its direction is untouched, so the opening angle is exactly the
  // figure's and only the distance differs -- above 1 starts nearer, below 1 further out.
  // Nothing to do with the ZOOM_* constants above: those frame the window from the spread and
  // are measured in landscape units. This is only where the camera sits, and nothing moves it
  // closer or further afterwards -- the scroll wheel zooms the window instead (VIEW_ZOOM_*).
  const START_ZOOM = 2.0;

  // The scroll wheel zooms the landscape, not the camera: it widens or narrows the window while
  // the box stays the size it is on screen, so zooming out shows more of the terrain rather than
  // a smaller picture of the same patch. A multiplier on Reset's framing, which Reset keeps.
  // The window eases to each new width, since a wheel notch is a jump of about a sixth.
  const VIEW_ZOOM_MIN = 0.75;
  const VIEW_ZOOM_MAX = 4.0;
  const VIEW_ZOOM_PER_PX = 0.0012; // e^(this * deltaY); a notch is ~100px
  const VIEW_ZOOM_TAU = 0.08; // seconds
  const REMESH_ZOOM_IN = 1.4; // rebuild once zoomed in this far, before the mesh reads as coarse
  // A rebuild costs ~75ms inside Plotly, nearly all of it proportional to the vertex count, and a
  // zoom gesture can need one every notch. So while the wheel is moving the mesh is rebuilt this
  // coarse (5x fewer vertices), and once it has been still for ZOOM_SETTLE_MS the full mesh is
  // rebuilt once -- the one hitch left lands when nothing is moving.
  const MESH_ZOOMING = 128;
  const ZOOM_SETTLE_MS = 250;
  const VALUE_RANGE_HALF = 12; // half a basin period (landscape.F_B's base is 24)
  const EYE = { x: 1.7, y: -1.7, z: 2.0 }; // the path-taken figure's opening viewpoint

  const el = (id) => document.getElementById(id);

  // Plotly is 1.6MB of script, and the walk sits below the abstract. So nothing here runs until
  // the page has loaded and gone idle: the page above it is readable first, and the plot box
  // already has its height in the stylesheet, so filling it in later moves nothing.
  const whenIdle = (fn) => window.requestIdleCallback
    ? requestIdleCallback(fn, { timeout: 2000 }) : setTimeout(fn, 200);
  const loadScript = (src) => new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = ok;
    s.onerror = () => fail(new Error("could not load " + src));
    document.head.appendChild(s);
  });

  const boot = () => whenIdle(() => {
    Promise.all([
      loadScript("static/js/plotly-gl3d.min.js"),
      fetch("static/js/landscape-constants.json").then((r) => r.json()),
    ])
      .then(([, K]) => start(K))
      .catch((e) => {
        const box = el("walk-plot");
        if (box) box.textContent = "Could not start the landscape walk: " + e.message;
      });
  });
  if (document.readyState === "complete") boot();
  else window.addEventListener("load", boot);

  function start(K) {
    const worker = new Worker("static/js/walk-worker.js");
    worker.postMessage({ type: "init", K, pMax: P_MAX });
    plot(K, worker);
  }

  function plot(K, worker) {
    // The Landscape draw being walked: the constants file's own on load, a fresh one every Reset.
    // Always 2D -- one design coordinate, one action coordinate.
    let land = null;
    const f = (d, a) => land.f(d, a);
    // Where the Pair is, as of the last step the worker reported.
    const sim = { mu_d: 0, mu_a: 0, epoch: 0 };

    // --- controls -------------------------------------------------------------------------
    // Six radii, unshared, on the ranges they were swept over: spread and generalization are
    // radii on the domain and were geomspaced, so their sliders are log; exploration is a
    // probability and stays linear. The ratio gets the swept values as detents, so the
    // slider and the heatmap below it share an axis.
    const SPEC = {
      sig_d: { scale: "log", range: K.axes.spread },
      sig_c: { scale: "log", range: K.axes.spread },
      e_d: { scale: "linear", range: K.axes.explore },
      e_c: { scale: "linear", range: K.axes.explore },
      g_d: { scale: "log", range: K.axes.gen },
      g_c: { scale: "log", range: K.axes.gen },
    };
    const RATIOS = K.axes.ratios;
    el("walk-k").max = RATIOS.length - 1;

    const toSlider = (p, v) => {
      const s = SPEC[p];
      const [lo, hi] = s.range;
      const t = s.scale === "log"
        ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
        : (v - lo) / (hi - lo);
      return Math.round(Math.min(1, Math.max(0, t)) * 1000);
    };
    const fromSlider = (p, raw) => {
      const s = SPEC[p];
      const [lo, hi] = s.range;
      const t = raw / 1000;
      return s.scale === "log" ? Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo)))
                               : lo + t * (hi - lo);
    };

    const params = { alpha: K.run.alpha, topPercent: K.run.topPercent, k: 1,
                     P: POPS[POP_DEFAULT] };

    function readControls() {
      for (const p of Object.keys(SPEC)) {
        params[p] = fromSlider(p, +el("walk-" + p).value);
        el("walk-" + p + "-val").textContent = params[p].toFixed(2);
      }
      const r = RATIOS[+el("walk-k").value];
      params.k = r;
      el("walk-k-val").textContent = "1:" + r;

      const N = POPS[+el("walk-P").value];
      params.P = N;
      el("walk-P-val").textContent = N;

      const speed = SPEEDS[+el("walk-speed").value];
      el("walk-speed-val").textContent = speed + "\u00d7";
      worker.postMessage({ type: "params", params, stepsPerSec: speed * STEPS_PER_SEC });
    }

    function applyConfig(cfg) {
      for (const p of Object.keys(SPEC)) el("walk-" + p).value = toSlider(p, cfg[p]);
      readControls();
    }

    // --- window ---------------------------------------------------------------------------
    // A fixed-size square of landscape that follows the Pair. The size is set once per Reset,
    // from the spread -- the Pair's own sampling radius is the only scale the widget knows, and
    // a window that ignored it would frame a narrow_greedy run as a dot and a wide one as a
    // scribble off the edge. It never resizes on its own mid-run: a window that breathed while you
    // were dragging would read as the camera fighting you. Only the visitor's wheel resizes it,
    // through `zoom`, a multiplier on that framing.
    const view = { d0: 0, a0: 0, vd: 0, va: 0, half: HALF_MIN, framed: HALF_MIN, zoom: 1,
                   zoomedAt: -Infinity, meshedAt: 0, meshedRes: MESH, meshedD0: 0, meshedA0: 0 };

    // The draw's value range, over a box a basin period wide (the walk wanders well past the
    // starting box), so it covers every window the walk is likely to reach. Recomputed per draw.
    // It pins two things. The z axis (up to the
    // visitor's zoom, which stretches it), so the surface does not bob vertically as
    // the window slides over a taller or flatter patch. And the colourscale: left to autoscale,
    // Viridis would re-normalise to whatever patch was last meshed, and every rebuild would
    // repaint the surface.
    let zLo = 0, zHi = 0, Z_RANGE = [0, 0];

    function setLandscape(spec) {
      land = Landscape.make(K.landscape, spec);
      ({ zLo, zHi } = Landscape.valueRange(land, -VALUE_RANGE_HALF, VALUE_RANGE_HALF, 512));
      Z_RANGE = [zLo - 0.05, zHi + 0.08];
    }
    setLandscape(K.landscape.start);


    // Rebuilt only by remesh(). The frame loop pushes it to Plotly only when `dirtySurface` says
    // it changed -- the other three traces are restyled every frame, but this one is 50k numbers
    // and redrawing it at 60Hz was the one genuinely expensive thing in the loop.
    let surface = null;
    let dirtySurface = false;

    /** Rebuild the mesh around the window at `res` points a side, wide enough for `span`. */
    function remesh(res = MESH, span = view.half) {
      const { d0, a0 } = view;
      const half = span * MESH_MARGIN;
      const axd = new Array(res), axa = new Array(res);
      for (let i = 0; i < res; i++) {
        const t = (2 * half * i) / (res - 1);
        axd[i] = d0 - half + t;
        axa[i] = a0 - half + t;
      }
      const z = []; // [action][design], Plotly's row-major convention for a surface
      for (let j = 0; j < res; j++) {
        const row = new Array(res);
        for (let i = 0; i < res; i++) row[i] = f(axd[i], axa[j]);
        z.push(row);
      }
      surface = {
        type: "surface", x: axd, y: axa, z: z,
        colorscale: "Viridis", cmin: zLo, cmax: zHi, showscale: false, hoverinfo: "skip",
        // A surface draws contour lines that chase the cursor, on by default and independent of
        // both hoverinfo and the axis spikes -- `highlight` is the one that turns them off.
        contours: { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } },
        lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.08 },
      };
      view.meshedAt = span;
      view.meshedRes = res;
      view.meshedD0 = d0;
      view.meshedA0 = a0;
      dirtySurface = true;
    }

    // --- trail and wake -------------------------------------------------------------------
    // Both are ring buffers: fixed allocation, and the oldest entry is simply overwritten.
    const trail = { d: new Float64Array(TRAIL_STEPS), a: new Float64Array(TRAIL_STEPS),
                    z: new Float64Array(TRAIL_STEPS), age: new Float64Array(TRAIL_STEPS),
                    n: 0, head: 0 };

    function pushTrail(d, a) {
      const i = trail.head;
      trail.d[i] = d;
      trail.a[i] = a;
      trail.z[i] = f(d, a) + 0.02; // lifted clear of the mesh so the line is never buried
      trail.head = (i + 1) % TRAIL_STEPS;
      if (trail.n < TRAIL_STEPS) trail.n++;
    }

    /** The trail in walk order, oldest first, with a 0..1 recency for the line colour. */
    function trailArrays() {
      const n = trail.n;
      const d = new Array(n), a = new Array(n), z = new Array(n), c = new Array(n);
      const first = trail.n < TRAIL_STEPS ? 0 : trail.head;
      for (let q = 0; q < n; q++) {
        const i = (first + q) % TRAIL_STEPS;
        d[q] = trail.d[i];
        a[q] = trail.a[i];
        z[q] = trail.z[i];
        c[q] = n > 1 ? q / (n - 1) : 1;
      }
      return { d, a, z, c };
    }

    const wake = []; // newest last; each entry is one worker tick's drawn samples

    function pushWake(wd, wa, wz, t) {
      // Lifted clear of the mesh, like the Trail, so a dot is never buried in it.
      for (let q = 0; q < wz.length; q++) wz[q] += 0.01;
      wake.push({ d: wd, a: wa, z: wz, n: wd.length, t });
    }

    /** The wake flattened, clipped to the window, with an age and a size per point. */
    function wakeArrays(now) {
      while (wake.length && now - wake[0].t >= WAKE_MS) wake.shift();
      const { d0, a0, half } = view;
      const d = [], a = [], z = [], age = [], size = [];
      for (let s = 0; s < wake.length; s++) {
        const w = wake[s];
        const fade = Math.max(0, 1 - (now - w.t) / WAKE_MS); // 1 newest, 0 on the way out
        const px = WAKE_SIZE_OLD + fade * (WAKE_SIZE_NEW - WAKE_SIZE_OLD);
        for (let i = 0; i < w.n; i++) {
          if (Math.abs(w.d[i] - d0) > half || Math.abs(w.a[i] - a0) > half) continue;
          d.push(w.d[i]);
          a.push(w.a[i]);
          z.push(w.z[i]);
          age.push(fade);
          size.push(px);
        }
      }
      return { d, a, z, age, size };
    }

    // --- plot -----------------------------------------------------------------------------

    const gd = el("walk-plot");
    const SURFACE_TRACE = 0, WAKE_TRACE = 1, TRAIL_TRACE = 2, HEAD_TRACE = 3;

    // White early to red late, the same reading as the path-taken figure's upper half-scale.
    const TRAIL_SCALE = [[0, "#ffe6cc"], [0.5, "#ff9100"], [1, "#7f0000"]];
    const WAKE_SCALE = [[0, "rgba(46,204,64,0)"], [1, "rgba(46,204,64,0.85)"]];
    const HEAD_COLOR = "#7f0000"; // the Pair itself, at the hot end of the trail scale

    // The three axes, named for the reader rather than for the model: the surface's two
    // coordinates are `design` and `action` everywhere else, and its height is `value`. The
    // plot says Design / Control / Reward so the axes pair with the Designer and Controller
    // headings the sliders carry directly below it.
    //
    // Each name is keyed to its own edge of the box, which is the whole reason for colouring
    // them -- and it is what lets the numbers go. This is a moving picture of a search, not a
    // figure anyone reads a coordinate off, so the ticks were furniture. Design is a red-orange
    // rather than a true orange to stay clear of the Trail's own #ff9100, and green is left
    // alone entirely because it is the Sample wake's.
    const AXES = {
      x: { text: "Design", color: "#FF5A1F" },
      y: { text: "Control", color: "#1DD3C0" },
      z: { text: "Reward", color: "#FFC400" },
    };
    // The wall grid and backgrounds are deliberately untouched: with the ticks gone the grid is
    // the only thing left saying which way the box is turned while the visitor rotates it.
    const axis = (a, extra) => Object.assign({
      title: { text: a.text, font: { color: a.color, size: 14, weight: 700 } },
      showline: true, linecolor: a.color, linewidth: 3,
      showticklabels: false, ticks: "",
      // showspikes off: the dashed cursor lines are for reading values off a static figure,
      // and this surface is moving.
      showspikes: false,
    }, extra);

    function traces() {
      const t = trailArrays();
      const w = wakeArrays(performance.now());
      const head = trail.n ? (trail.head - 1 + TRAIL_STEPS) % TRAIL_STEPS : 0;
      // The wake opens empty, and gl3d never builds the drawing object for a trace with no points
      // -- which would leave the direct draw below nothing to write into. One dot, fully faded.
      if (!w.d.length) {
        w.d.push(trail.d[head]); w.a.push(trail.a[head]); w.z.push(trail.z[head]);
        w.age.push(0); w.size.push(WAKE_SIZE_OLD);
      }
      return [
        surface,
        { type: "scatter3d", mode: "markers", x: w.d, y: w.a, z: w.z, name: "samples",
          // line.width 0 is not the default it looks like: an ARRAY marker.size makes this a
          // bubble trace, and markerDefaults then defaults the outline to width 1 in
          // Color.background rather than to width 0. Left alone, every dot wears a white ring
          // that is most of its area at these sizes, which reads as a pale mint speck rather
          // than a green one.
          marker: { size: w.size, color: w.age, colorscale: WAKE_SCALE, cmin: 0, cmax: 1,
                    line: { width: 0 } },
          hoverinfo: "skip", showlegend: false },
        { type: "scatter3d", mode: "lines", x: t.d, y: t.a, z: t.z, name: "trail",
          line: { width: 5, color: t.c, colorscale: TRAIL_SCALE, cmin: 0, cmax: 1 },
          hoverinfo: "skip", showlegend: false },
        { type: "scatter3d", mode: "markers", x: [trail.d[head]], y: [trail.a[head]],
          z: [trail.z[head]], name: "pair",
          marker: { size: 5, color: HEAD_COLOR, line: { width: 1, color: "#fff" } },
          hoverinfo: "skip", showlegend: false },
      ];
    }

    /** The axis ranges the window wants. These also clip the Trail: the tail of a long walk falls
     *  outside the window, and without a range Plotly would zoom out to include it. */
    //
    // The reward axis stretches with the window's width, floor anchored. The box keeps its shape on screen,
    // so with the reward range pinned, zooming out packed more ground into the same width at the
    // same height and the terrain read as spikier -- and zooming in flattened it. Scaling the
    // range by the zoom keeps slopes looking the same at every zoom: zoomed out, the terrain sits
    // low in the box; zoomed in, the peaks rise out of its top. Width is the whole of it, so
    // Reset's framing from the spread counts exactly as the visitor's zoom does -- a wide-spread
    // run framed wide is no spikier than a narrow one. The pinned range is what a window of
    // ZOOM_REF_HALF gets.
    const windowRanges = () => {
      const s = view.half / ZOOM_REF_HALF;
      return {
        x: [view.d0 - view.half, view.d0 + view.half],
        y: [view.a0 - view.half, view.a0 + view.half],
        z: [Z_RANGE[0], Z_RANGE[0] + (Z_RANGE[1] - Z_RANGE[0]) * s],
      };
    };

    const layout = {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      scene: {
        // A fixed box, so the visitor's dragged-in rotation and zoom mean the same thing from one
        // frame to the next. Everything that moves, moves in data space.
        aspectmode: "manual",
        aspectratio: { x: 1, y: 1, z: 0.55 },
        xaxis: axis(AXES.x), // range is set from the window, below
        yaxis: axis(AXES.y),
        zaxis: axis(AXES.z, { range: Z_RANGE }),
        camera: { eye: { x: EYE.x / START_ZOOM, y: EYE.y / START_ZOOM, z: EYE.z / START_ZOOM } },
        dragmode: "turntable",
        hovermode: false, // the scene's own, not layout.hovermode -- that one is cartesian
      },
      showlegend: false,
    };

    // --- run ------------------------------------------------------------------------------

    /** Start over somewhere random. `fresh` also draws a new Landscape; only the opening reset,
     *  which walks the constants file's draw, leaves it be. */
    function reset(fresh) {
      let spec = null;
      if (fresh) {
        spec = Landscape.draw(K.landscape, (Math.random() * 2 ** 32) >>> 0);
        setLandscape(spec);
      }
      const [lo, hi] = K.bounds;
      view.d0 = lo + Math.random() * (hi - lo);
      view.a0 = lo + Math.random() * (hi - lo);
      // Framed off the wider of the two spreads, so the busier optimizer stays on screen.
      const sig = Math.max(params.sig_d, params.sig_c);
      const want = ZOOM_REF_HALF * Math.pow(sig / ZOOM_REF_SIG, ZOOM_POWER);
      view.framed = Math.min(HALF_MAX, Math.max(HALF_MIN, want));
      view.half = view.framed * view.zoom;
      view.vd = 0;
      view.va = 0;
      trail.n = 0;
      trail.head = 0;
      wake.length = 0;
      // A new epoch, so steps the worker ran before it heard of the reset are dropped on arrival.
      sim.epoch++;
      sim.mu_d = view.d0;
      sim.mu_a = view.a0;
      worker.postMessage({ type: "reset", epoch: sim.epoch, d0: view.d0, a0: view.a0, spec });
      pushTrail(view.d0, view.a0);
      remesh();
    }

    /**
     * Follow the Pair: chase the window's centre toward it, and ease its width toward the zoom.
     *
     * The camera itself is left entirely alone -- whatever rotation and zoom the visitor dragged
     * into it stays put, and the scene's axis ranges move underneath instead. That is what makes
     * "follows the dot" and "keeps my angle" the same operation: the Pair sits at the centre of
     * a box of fixed size, so a camera aimed at that centre stays aimed at the Pair.
     *
     * A critically damped spring rather than an exponential ease. An ease is smooth in position
     * but its velocity is the raw distance to `mu`, and `mu` moves a fresh random amount every
     * step -- so the ground twitched frame to frame even while it was gliding. Damping the
     * velocity too makes both continuous, and critically damped is the one setting that closes
     * the gap without swinging past the Pair and coming back.
     */
    function updateWindow(dt) {
      const w = 1 / FOLLOW_TAU;
      view.vd += ((sim.mu_d - view.d0) * w * w - 2 * w * view.vd) * dt;
      view.va += ((sim.mu_a - view.a0) * w * w - 2 * w * view.va) * dt;
      view.d0 += view.vd * dt;
      view.a0 += view.va * dt;
      const target = view.framed * view.zoom;
      view.half += (target - view.half) * (1 - Math.exp(-dt / VIEW_ZOOM_TAU));
      if (Math.abs(target - view.half) < 1e-4 * target) view.half = target;
      // The window may wander -- or widen -- anywhere inside the mesh's margin before the mesh has
      // to be rebuilt, and may narrow until the mesh would read as coarse. The rebuild is centred
      // on where the window is now and covers it entirely, so nothing visible changes when it
      // happens.
      const reach = view.half + Math.max(Math.abs(view.d0 - view.meshedD0),
                                         Math.abs(view.a0 - view.meshedA0));
      const cover = view.meshedAt * (1 + (MESH_MARGIN - 1) * REMESH_AT);
      const zooming = performance.now() < view.zoomedAt + ZOOM_SETTLE_MS;
      if (reach > cover || view.half * REMESH_ZOOM_IN < view.meshedAt) {
        // Mid-gesture the coarse mesh is built out to where the zoom is heading rather than where
        // the eased window is now, so a steady zoom out does not rebuild again a frame later.
        if (zooming) remesh(MESH_ZOOMING, Math.max(view.half, target));
        else remesh();
      } else if (!zooming && view.meshedRes !== MESH) {
        remesh();
      }
    }

    /**
     * Slide the window into the scene the redraw is about to apply, and keep the two shadows
     * that let the visitor drive a plot which is redrawing sixty times a second installed.
     *
     * `scene.plot()` reads each axis's `range` straight out of the full layout and hands it to
     * `glplot.setBounds`, so writing the window in here moves the ground under the Pair on the
     * redraw that is already happening -- no second Plotly call, and it can therefore be done
     * every single frame. Pushing them through `relayout` instead only made sense when the mesh
     * had to move with them, which is what used to make the follow lurch: the ground sat still
     * for a dozen frames and then caught up all at once.
     *
     * The camera is not written here at all -- see mutedViewport, which is why it no longer
     * needs to be.
     *
     * Most frames no longer redraw through Plotly at all (see drawDirect, which hands the bounds to
     * the gl plot itself). The ranges are still written here so that the restyle a rebuilt mesh
     * does run frames the scene the same way.
     */
    function holdScene() {
      const fl = gd._fullLayout;
      const sc = fl && fl.scene && fl.scene._scene;
      if (!sc) return;
      // Asserted here rather than once after newPlot: a single missed install -- the scene not
      // built yet, a resize rebuilding it -- would put the camera fight straight back.
      if (sc.updateFx !== mutedFx) sc.updateFx = mutedFx;
      if (sc.setViewport !== mutedViewport) sc.setViewport = mutedViewport;
      // Read fresh out of `camera` on every mouse move, so setting it here is enough.
      if (sc.camera.rotateSpeed !== ROTATE_SPEED) sc.camera.rotateSpeed = ROTATE_SPEED;
      const r = windowRanges();
      gd.layout.scene.xaxis.range = r.x; // survives the supplyDefaults that restyle runs...
      gd.layout.scene.yaxis.range = r.y;
      gd.layout.scene.zaxis.range = r.z;
      fl.scene.xaxis.range = r.x.slice(); // ...and this is the copy setBounds is handed
      fl.scene.yaxis.range = r.y.slice();
      fl.scene.zaxis.range = r.z.slice();
    }

    /* Let the visitor drive the scene while the walk is still running.
     *
     * holdScene keeps the camera's VALUE across a redraw, but `scene.plot()` also calls
     * `updateFx(dragmode, hovermode)`, and its turntable branch is where a drag used to die. It
     * assigns `camera.up = [0,0,1]` and `camera.mode = "turntable"`, and neither setter is
     * idempotent: `up` runs `lookAt(lastT, null, null, [0,0,1])`, and `mode` re-plants the camera
     * at lastT, schedules a second lookAt 500ms out to pull it back to z-up, and flushes the
     * spline history behind it. The mouse listener writes the visitor's rotation into that same
     * history, so every redraw threw away what they had just done -- sixty times a second, which
     * reads as a plot that ignores the mouse. The widget used to answer that by not drawing at
     * all while a pointer was down, which traded the dead mouse for a frozen picture.
     *
     * Neither dragmode nor hovermode ever changes here, so after the opening plot has put the
     * camera in turntable mode there is nothing for updateFx to do but carry hovermode across.
     * Shadowing it with that one line is what lets the walk keep drawing mid-drag.
     */
    function mutedFx(dragmode, hovermode) { this.fullSceneLayout.hovermode = hovermode; }

    /* Let the visitor's rotation survive a redraw, smoothing and all.
     *
     * `scene.plot()` also ends with `setViewport(fullSceneLayout)`, whose first act is
     * `camera.lookAt(eye, center, up)` -- and that resolves to `view.lookAt(view.lastT(), ...)`,
     * which WRITES a keyframe into the turntable's spline at its last recorded time. gl-plot3d's
     * own tick, meanwhile, draws `view.recalcMatrix(now - 32)`: the picture is deliberately 32ms
     * behind the end of the spline, and that lag IS the rotation smoothing.
     *
     * The loop used to read the camera back out with `getCamera()` -- which samples the spline at
     * that same endpoint -- and write it into the layout, so that setViewport re-aimed the camera
     * where it already was. That held its value, but it also flattened the 32ms of smoothing
     * sixty times a second, and it was not even the no-op it looked like: turntable's `lookAt`
     * re-derives (up, right, angle, radius) from the matrix, which is not an exact inverse of the
     * unpack. Idle, that is a stable fixed point and invisible. Mid-drag, with the mouse writing
     * its own keyframes at the same `lastT`, it read as the plot twitching while you rotated it.
     *
     * Shadowing setViewport leaves the drag spline entirely alone, and nothing else it did
     * matters here: the aspect ratio is manual and fixed, and the projection never changes.
     */
    function mutedViewport(sceneLayout) { this.glplot.setAspectratio(sceneLayout.aspectratio); }

    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type !== "steps" || m.epoch !== sim.epoch) return;
      const p = m.path;
      for (let i = 0; i < p.length; i += 2) pushTrail(p[i], p[i + 1]);
      if (perf) perf.steps += p.length / 2;
      sim.mu_d = p[p.length - 2];
      sim.mu_a = p[p.length - 1];
      pushWake(m.wd, m.wa, m.wz, performance.now());
    };

    /* Draw the Trail, the wake and the Pair straight into gl3d's drawing objects.
     *
     * Plotly.restyle, even on three small traces, runs the scene's whole redraw, and that calls
     * update() on EVERY trace -- the surface included, which re-derives 83k vertices, their
     * normals and colours whether or not the mesh changed. Measured, that was 75ms of a 100ms
     * frame at every setting. The Trail's own trip through Plotly's per-point colorscale lookups
     * was another 10.
     *
     * So the per-frame path skips Plotly entirely. The first time round it records the options
     * Plotly handed each gl object (glyph, outline, opacity, projection...), and every frame after
     * passes those same options with only the positions, colours and sizes replaced -- computed
     * here exactly as Plotly's scatter3d conversion would: positions multiplied by the scene's
     * data scale, colours as 0..1 RGBA, and an array marker size doubled (bubble sizing, which
     * halves the 4x every gl3d marker gets). The axis bounds go straight to the gl plot for the
     * same reason. The surface still goes through restyle, but only when it is rebuilt.
     */
    /** A hex colourscale sampled at 256 points, as 0..1 RGBA -- Plotly's linear RGB blend. */
    const lutOf = (scale) => {
      const stops = scale.map(([t, c]) => [t, [1, 3, 5].map((i) => parseInt(c.substr(i, 2), 16))]);
      const lut = [];
      for (let i = 0; i < 256; i++) {
        const t = i / 255;
        let k = 1;
        while (k < stops.length - 1 && stops[k][0] < t) k++;
        const [t0, c0] = stops[k - 1], [t1, c1] = stops[k];
        const u = (t - t0) / (t1 - t0);
        lut.push([0, 1, 2].map((j) => (c0[j] + u * (c1[j] - c0[j])) / 255).concat(1));
      }
      return lut;
    };
    const TRAIL_LUT = lutOf(TRAIL_SCALE);
    const WAKE_RGB = [46 / 255, 204 / 255, 64 / 255];
    const WAKE_ALPHA = 0.85; // WAKE_SCALE's top stop; its bottom is the same green at 0

    const glp = { wake: null, trail: null, head: null };
    const clipBox = [[0, 0, 0], [0, 0, 0]]; // shared by every object; see the end of drawDirect

    /** The gl object `key` of trace `index`, with the options Plotly last updated it with. */
    function capture(sc, index, key) {
      const trace = sc.traces[gd._fullData[index].uid];
      const o = trace && trace[key];
      if (!o) return null;
      const orig = o.update;
      let opts = null;
      o.update = function (p) { opts = p; return orig.call(this, p); };
      trace.update(trace.data);
      o.update = orig;
      return { o, opts };
    }

    function drawDirect(now) {
      const sc = gd._fullLayout.scene._scene;
      const tr = (i) => sc.traces[gd._fullData[i].uid];
      // Recaptured if gl3d ever rebuilt an object (a lost context, a resize that re-plots).
      if (!glp.wake || tr(WAKE_TRACE).scatterPlot !== glp.wake.o) {
        glp.wake = capture(sc, WAKE_TRACE, "scatterPlot");
      }
      if (!glp.trail || tr(TRAIL_TRACE).linePlot !== glp.trail.o) {
        glp.trail = capture(sc, TRAIL_TRACE, "linePlot");
      }
      if (!glp.head || tr(HEAD_TRACE).scatterPlot !== glp.head.o) {
        glp.head = capture(sc, HEAD_TRACE, "scatterPlot");
      }
      const [sx, sy, sz] = sc.dataScale;

      const n = trail.n;
      const first = n < TRAIL_STEPS ? 0 : trail.head;
      const tPos = new Array(n), tCol = new Array(n);
      for (let q = 0; q < n; q++) {
        const i = (first + q) % TRAIL_STEPS;
        tPos[q] = [trail.d[i] * sx, trail.a[i] * sy, trail.z[i] * sz];
        tCol[q] = TRAIL_LUT[n > 1 ? Math.round((255 * q) / (n - 1)) : 255];
      }
      glp.trail.o.update(Object.assign({}, glp.trail.opts, { position: tPos, color: tCol }));

      const w = wakeArrays(now);
      const m = w.d.length;
      const wPos = new Array(m), wCol = new Array(m), wSize = new Array(m);
      for (let i = 0; i < m; i++) {
        wPos[i] = [w.d[i] * sx, w.a[i] * sy, w.z[i] * sz];
        wCol[i] = [WAKE_RGB[0], WAKE_RGB[1], WAKE_RGB[2], WAKE_ALPHA * w.age[i]];
        wSize[i] = 2 * w.size[i];
      }
      glp.wake.o.update(Object.assign({}, glp.wake.opts,
                                      { position: wPos, color: wCol, size: wSize }));

      const h = n ? (trail.head - 1 + TRAIL_STEPS) % TRAIL_STEPS : 0;
      glp.head.o.update(Object.assign({}, glp.head.opts,
                                      { position: [[trail.d[h] * sx, trail.a[h] * sy, trail.z[h] * sz]] }));

      const r = windowRanges();
      sc.glplot.setBounds(0, { min: r.x[0] * sx, max: r.x[1] * sx });
      sc.glplot.setBounds(1, { min: r.y[0] * sy, max: r.y[1] * sy });
      sc.glplot.setBounds(2, { min: r.z[0] * sz, max: r.z[1] * sz });

      // Clip to the window's sides and floor, but not its top: zoomed in, the reward range is
      // shorter than the terrain, and the peaks should rise out of the box rather than be sheared
      // off at the grid. gl-plot3d would otherwise copy the axis bounds into every object's clip
      // box on each render, so that copy is switched off and the box is handed out here instead.
      sc.glplot.clipToBounds = false;
      const c = clipBox;
      c[0][0] = r.x[0] * sx; c[1][0] = r.x[1] * sx;
      c[0][1] = r.y[0] * sy; c[1][1] = r.y[1] * sy;
      c[0][2] = r.z[0] * sz; c[1][2] = 1e9; // effectively unbounded; the shader wants a number
      const surf = tr(SURFACE_TRACE).surface;
      if (surf) surf.clipBounds = c;
      glp.trail.o.clipBounds = c;
      // gl-scatter3d draws its points against `this.axes.bounds` -- the shared axis box -- and only
      // reads its own clipBounds for projections. So each dot object's `axes` answers with a
      // stand-in that inherits everything from the real axes but gives the same open-top box as
      // `bounds`. It has to be an accessor: gl-plot3d re-assigns `axes` on every object right
      // before drawing it, so a plain assignment here is overwritten before it is ever read.
      for (const o of [glp.wake.o, glp.head.o]) {
        o.clipBounds = c;
        if (o._openTopAxes) continue;
        let real = o.axes, proxy = null;
        Object.defineProperty(o, "axes", {
          configurable: true,
          get() {
            if (!real) return real;
            if (!proxy || Object.getPrototypeOf(proxy) !== real) {
              proxy = Object.create(real, { bounds: { value: clipBox } });
            }
            return proxy;
          },
          set(v) { real = v; },
        });
        o._openTopAxes = true;
      }
    }

    let running = false; // the worker and the frame loop are going
    let plotted = false;
    let onScreen = false;
    let lastFrame = 0;
    let rafPending = false;

    // ?perf: once a second, log where the frame goes -- frame rate, the JS the loop itself runs,
    // the restyle inside it, each gl trace's update inside that, and the pace the worker delivered.
    // Diagnostics only; nothing else reads it.
    const perf = new URLSearchParams(location.search).has("perf") && {
      t0: performance.now(), frames: 0, steps: 0, draw: 0, restyle: 0, loop: 0, remesh: 0,
      update: {},
      wrap() {
        const sc = gd._fullLayout.scene._scene;
        for (const uid in sc.traces) {
          const tr = sc.traces[uid];
          if (tr._perfWrapped) continue;
          const name = tr.data.name || tr.data.type, orig = tr.update;
          tr.update = function () {
            const t = performance.now();
            const r = orig.apply(this, arguments);
            perf.update[name] = (perf.update[name] || 0) + performance.now() - t;
            return r;
          };
          tr._perfWrapped = true;
        }
      },
      report(now) {
        const n = this.frames, secs = (now - this.t0) / 1000;
        if (secs < 1) return;
        const per = (x) => (x / n).toFixed(2);
        const upd = Object.entries(this.update).map(([k, v]) => k + " " + per(v)).join(", ");
        console.log(`[walk perf] ${(n / secs).toFixed(0)}fps, ${(this.steps / secs).toFixed(0)} ` +
          `steps/s, ms/frame: loop ${per(this.loop)} (direct draw ${per(this.draw)}, remesh ` +
          `restyle ${per(this.restyle)}; its gl updates ${upd}), remeshes ${this.remesh}, ` +
          `N=${params.P}`);
        Object.assign(this, { t0: now, frames: 0, steps: 0, draw: 0, restyle: 0, loop: 0,
                              remesh: 0, update: {} });
      },
    };

    function frame() {
      rafPending = false;
      if (!running) return;
      const now = performance.now();
      const dt = lastFrame ? Math.min((now - lastFrame) / 1000, DT_MAX) : 1 / 60;
      lastFrame = now;

      if (perf) perf.wrap();
      updateWindow(dt);
      holdScene();

      // The mesh first, when it has been rebuilt: that restyle runs the scene's full redraw,
      // which rewrites every trace from Plotly's own (stale) copy of its data and can move the
      // data scale -- so the direct draw has to come after it, in the same frame. That redraw also
      // RENDERS synchronously before returning, with the stale Trail, wake and Pair; the frame
      // would show without them (the dots blinking off whenever the window moved far enough to
      // rebuild), so a rebuild frame renders once more after the direct draw has put them back.
      const remeshed = dirtySurface;
      if (dirtySurface) {
        const t0 = perf && performance.now();
        Plotly.restyle(gd, { x: [surface.x], y: [surface.y], z: [surface.z],
                             cmin: [surface.cmin], cmax: [surface.cmax] }, [SURFACE_TRACE]);
        dirtySurface = false;
        if (perf) { perf.remesh++; perf.restyle += performance.now() - t0; }
      }
      const tDraw = perf && performance.now();
      drawDirect(now);
      if (remeshed) gd._fullLayout.scene._scene.glplot.redraw();
      if (perf) perf.draw += performance.now() - tDraw;

      if (perf) {
        const end = performance.now();
        perf.frames++;
        perf.loop += end - now;
        perf.report(end);
      }
      if (running) {
        rafPending = true;
        requestAnimationFrame(frame);
      }
    }

    // --- wiring ---------------------------------------------------------------------------

    for (const p of Object.keys(SPEC)) el("walk-" + p).addEventListener("input", readControls);
    for (const id of ["walk-k", "walk-P", "walk-speed"]) {
      el(id).addEventListener("input", readControls);
    }
    el("walk-reset").addEventListener("click", () => reset(true));

    // scrollZoom is off, so gl3d's own wheel dolly never runs; the wheel is taken here instead,
    // and only over the plot -- the page scrolls normally everywhere else.
    gd.addEventListener("wheel", (e) => {
      e.preventDefault();
      const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      view.zoomedAt = performance.now();
      view.zoom = Math.min(VIEW_ZOOM_MAX,
                           Math.max(VIEW_ZOOM_MIN, view.zoom * Math.exp(px * VIEW_ZOOM_PER_PX)));
    }, { passive: false });

    /* Turning the camera is the only thing the mouse does to it.
     *
     * gl-plot3d's controller splits its drags by button: left turns, right pans, and middle
     * dollies -- and the last two move the camera off the centre and distance the walk assumes
     * it keeps, since the window is what zooms here and Reset re-frames that, not the camera.
     * Ctrl and Alt with the left button reach those same two branches, and the turn branch runs
     * only with neither held, so a modified drag is dropped too and nothing is lost.
     *
     * Both events have to be taken, in the capture phase, before they reach the gl canvas inside
     * gd that the controller listens on: its mousemove re-reads `buttons` off every event rather
     * than trusting what mousedown told it, so stopping the press alone would leave the drag to
     * reconstitute itself on the first move. Preventing the press also costs the middle button
     * its autoscroll, which over a plot that already takes the wheel is no loss either.
     */
    gd.addEventListener("mousedown", (e) => {
      if (e.button === 0 && !e.ctrlKey && !e.altKey && !e.metaKey) return;
      e.preventDefault();
      e.stopPropagation();
    }, true);
    gd.addEventListener("mousemove", (e) => {
      if (!(e.buttons & ~1) && !e.ctrlKey && !e.altKey && !e.metaKey) return;
      e.stopPropagation();
    }, true);

    // The walk is only computed while it is watched: scrolled away or in a background tab, both
    // the worker and the frame loop stop, and it picks up where it was. The run still never ends.
    function setRunning() {
      const want = plotted && onScreen && !document.hidden;
      if (want === running) return;
      running = want;
      worker.postMessage({ type: "run", on: want });
      if (want) lastFrame = 0;
      if (want && !rafPending) {
        rafPending = true;
        requestAnimationFrame(frame);
      }
    }
    new IntersectionObserver((entries) => {
      onScreen = entries[entries.length - 1].isIntersecting;
      setRunning();
    }).observe(gd);
    document.addEventListener("visibilitychange", setRunning);

    // Opens on matched radii -- the configuration the argument is read against, and the one whose
    // two radii are comparable, so the first thing a visitor sees is the Pair working. Population
    // and speed have no archetype: they are how the walk is watched, so they open on the run's own
    // population and the pace the page has always run at.
    el("walk-P").value = POP_DEFAULT;
    el("walk-speed").value = SPEED_DEFAULT;
    applyConfig(K.archetypes.matched_radii);

    reset(false);
    const opened = windowRanges();
    layout.scene.xaxis.range = opened.x;
    layout.scene.yaxis.range = opened.y;
    Plotly.newPlot(gd, traces(), layout,
                   { displayModeBar: false, responsive: true, scrollZoom: false })
      .then(() => {
        dirtySurface = false; // newPlot already carries the mesh reset() built
        plotted = true;
        setRunning();
      });
  }
})();
