/* The Landscape walk's simulation, off the main thread.
 *
 * walk.js draws; this walks. Splitting them is what lets the picture stay smooth when the Pair
 * cannot keep up: the page draws every frame whatever this manages, and a large population at a
 * fast playback speed costs pace rather than frame rate.
 *
 * Playback speed arrives as a target in steps per second. Each tick runs the steps owed since the
 * last one, up to a time budget, and never carries a backlog past a tick's worth -- so when the
 * target is out of reach the walk simply runs slower, rather than falling ever further behind and
 * lurching to catch up whenever it gets the chance.
 *
 * Messages in:  init {K, pMax} · params {params, stepsPerSec} · reset {epoch, d0, a0, spec?} · run {on}
 * Messages out: steps {epoch, path, wd, wa, wz}
 *
 * A reset carrying `spec` also swaps the Landscape draw (see Landscape.draw).
 */

importScripts("landscape.js");

const TICK_MS = 16;
const BUDGET_MS = 12; // of each tick; the rest is left for the message and the scheduler
const WAKE_PER_POP = 256; // Sample wake dots per step = N*N / this, at least 1 (see walk.js)

let pair = null;
let params = null;
let stepsPerSec = 0;
let running = false;
let epoch = 0;
let owed = 0;
let last = 0;
let timer = 0;

/* The first `c` entries of a persistent permutation of 0..P-1, freshly shuffled -- a uniform choice
 * of which samples the wake shows, so it has the shape of the population rather than a corner of
 * it. A partial Fisher-Yates leaves a permutation behind, so the buffer is reshuffled rather than
 * re-initialised every step. */
let perm = new Int32Array(0);
function shuffled(P, c) {
  if (perm.length !== P) {
    perm = new Int32Array(P);
    for (let i = 0; i < P; i++) perm[i] = i;
  }
  for (let i = 0; i < c; i++) {
    const j = i + Math.floor(Math.random() * (P - i));
    const t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
  return perm;
}

function tick() {
  timer = 0;
  if (!running || !params) return;
  const now = performance.now();
  owed += (stepsPerSec * Math.min(now - last, 250)) / 1000;
  last = now;

  const want = Math.floor(owed);
  const path = new Float64Array(2 * Math.max(want, 1));
  let done = 0, out = null;
  while (done < want && performance.now() - now < BUDGET_MS) {
    out = pair.step(params);
    path[2 * done] = pair.mu_d;
    path[2 * done + 1] = pair.mu_a;
    done++;
  }
  // Whatever the budget cut short is forgiven, bar one tick's worth.
  owed = Math.min(owed - done, 1 + (stepsPerSec * TICK_MS) / 1000);

  if (done) {
    // Only the last step of the tick is drawn: the wake is a picture of what the Pair is trying
    // now, one generation per tick, however many steps the tick ran.
    const P = out.n;
    const c = Math.max(1, Math.min(Math.round(P / WAKE_PER_POP), P));
    const order = shuffled(P, c);
    const wd = new Float64Array(c), wa = new Float64Array(c), wz = new Float64Array(c);
    for (let q = 0; q < c; q++) {
      const i = order[q];
      wd[q] = out.d[Math.floor(i / out.perDesign)];
      wa[q] = out.a[i];
      wz[q] = out.v[i];
    }
    const p = path.subarray(0, 2 * done).slice();
    postMessage({ type: "steps", epoch, path: p, wd, wa, wz },
                [p.buffer, wd.buffer, wa.buffer, wz.buffer]);
  }
  // The wait tops the tick up to TICK_MS rather than adding to it: a saturated tick has already
  // spent its time, and this thread is the walk's own, so it may run flat out.
  timer = setTimeout(tick, Math.max(0, TICK_MS - (performance.now() - now)));
}

onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    pair = new Landscape.Pair(m.K, m.K.landscape.start, m.pMax);
  } else if (m.type === "params") {
    params = m.params;
    stepsPerSec = m.stepsPerSec;
  } else if (m.type === "reset") {
    epoch = m.epoch;
    if (m.spec) pair.setLandscape(m.spec);
    pair.reset(m.d0, m.a0);
    owed = 0;
  } else if (m.type === "run") {
    running = m.on;
    if (running && !timer) {
      last = performance.now();
      timer = setTimeout(tick, 0);
    }
  }
};
