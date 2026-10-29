#!/usr/bin/env node
// Correctness tests for the simulator embedded in viewer/index.html.
//
// Usage: node viewer/test_sim.js [--no-python]
//
// 1. Extracts the code between the "SIM START"/"SIM END" markers of index.html.
// 2. Compares Life against a naive Set-based reference on random soups and
//    gliders placed across tile borders (incl. negative coordinates), every gen.
// 3. Compares against the Python simulator gol.life (Gosper gun + gliders
//    crossing tile borders, 300 generations), exact cell sets.
// 4. Prints a small benchmark (sparse gliders + still lifes).
"use strict";
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const m = /\/\/ ==== SIM START ====([\s\S]*?)\/\/ ==== SIM END ====/.exec(html);
if (!m) throw new Error("SIM markers not found in index.html");
// new Function rather than vm.runInNewContext: inside a vm context every global
// (Math, ...) goes through a slow interceptor, which would distort the benchmark.
const { parseRLE, Life } = new Function(m[1] + "\nreturn { parseRLE, Life };")();

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

const keyOf = (x, y) => `${x},${y}`;
const asSet = (cells) => new Set(cells.map(([x, y]) => keyOf(x, y)));
function sameSet(a, b) {
  if (a.size !== b.size) return false;
  for (const k of a) if (!b.has(k)) return false;
  return true;
}

// Naive reference implementation on a Set of "x,y" strings.
function refStep(set) {
  const counts = new Map();
  for (const k of set) {
    const [x, y] = k.split(",").map(Number);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (dx || dy) { const n = keyOf(x + dx, y + dy); counts.set(n, (counts.get(n) || 0) + 1); }
    }
  }
  const out = new Set();
  for (const [k, c] of counts) if (c === 3 || (c === 2 && set.has(k))) out.add(k);
  return out;
}

// Deterministic PRNG
function rng(seed) {
  return () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
}

const GLIDER_SE = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
function glider(dir, x, y) {
  const sx = dir.includes("E") ? 1 : -1, sy = dir.includes("S") ? 1 : -1;
  return GLIDER_SE.map(([a, b]) => [x + (sx > 0 ? a : 2 - a), y + (sy > 0 ? b : 2 - b)]);
}

// --- 1. RLE parser ---------------------------------------------------------
{
  const p = parseRLE("#N test\n#C hello\n#C LABEL -5 7 a label\n#R -3 10\nx = 3, y = 3, rule = B3/S23\nbo$2bo$3o!\n");
  check("rle: cells with #R origin",
    sameSet(asSet(p.cells), asSet([[-2, 10], [-1, 11], [-3, 12], [-2, 12], [-1, 12]])));
  check("rle: label", p.labels.length === 1 && p.labels[0].x === -5 && p.labels[0].y === 7 && p.labels[0].text === "a label");
  check("rle: comments", p.comments.length === 2 && p.name === "test");
}

// --- 2. against the naive reference ----------------------------------------
function compareRef(name, cells, gens) {
  const life = new Life();
  life.load(cells);
  let ref = asSet(cells);
  for (let g = 1; g <= gens; g++) {
    life.step();
    ref = refStep(ref);
    const got = asSet(life.cells());
    if (!sameSet(got, ref) || life.pop !== ref.size) {
      check(name, false, `mismatch at gen ${g} (got ${got.size}, pop ${life.pop}, want ${ref.size})`);
      return;
    }
  }
  check(name, true, `${gens} gens, final pop ${ref.size}, tiles ${life.tiles.size}`);
}

{
  // gliders in all directions whose paths cross tile borders and corners, both signs
  const gl = [];
  for (const d of ["SE", "SW", "NE", "NW"]) {
    for (const [x, y] of [[28, 28], [-3, -3], [30, -34], [-36, 29], [62, 1], [1, 62]]) {
      gl.push(...glider(d, x + (d.includes("E") ? 0 : 100), y + (d.includes("S") ? 0 : 100)));
    }
  }
  compareRef("ref: gliders crossing tile borders", gl, 400);

  // random soups centred on tile corners (lots of border activity)
  for (const [seed, cx, cy] of [[1, 0, 0], [2, 32, -32], [3, -64, 96], [4, 31, 31]]) {
    const r = rng(seed), cells = [];
    for (let y = -12; y < 12; y++) for (let x = -12; x < 12; x++) if (r() < 0.4) cells.push([cx + x, cy + y]);
    compareRef(`ref: soup seed ${seed} at (${cx},${cy})`, cells, 600);
  }

  compareRef("ref: blinkers on tile edges", [[31, 0], [31, 1], [31, 2], [32, 40], [33, 40], [34, 40], [-1, -1], [0, -1], [1, -1]], 50);

  // Many small sparse soups straddling a tile edge or corner: lots of tile
  // creation / deletion and births across borders.
  const r = rng(99);
  let bad = null;
  for (let i = 0; i < 400 && !bad; i++) {
    const cx = 32 * Math.floor(r() * 4 - 2), cy = 32 * Math.floor(r() * 4 - 2) + (i % 2 ? 16 : 0);
    const cells = [];
    for (let y = -5; y < 5; y++) for (let x = -5; x < 5; x++) if (r() < 0.3) cells.push([cx + x, cy + y]);
    const life = new Life();
    life.load(cells);
    let ref = asSet(cells);
    for (let g = 1; g <= 120; g++) {
      life.step(); ref = refStep(ref);
      if (!sameSet(asSet(life.cells()), ref)) { bad = `soup ${i} at gen ${g}`; break; }
    }
  }
  check("ref: 400 small soups on tile edges/corners", !bad, bad || "120 gens each");
}

// --- 3. against the Python simulator ----------------------------------------
if (!process.argv.includes("--no-python")) {
  const py = `
import json
from gol import life, patterns as P
cases = {
  "gosper gun": P.GOSPER_GUN + [5, 7],
  "gliders crossing borders": P.union(
      P.glider("SE", 0, 25, 27), P.glider("SW", 1, 40, -5), P.glider("NE", 2, -40, 60),
      P.glider("NW", 3, 70, 70), P.glider("SE", 2, -33, -33), P.glider("NE", 0, 0, 31)),
  "gun + still lifes": P.union(P.GOSPER_GUN + [-20, -20], P.BLOCK + [62, 62], P.BEEHIVE + [31, 30],
      P.EATER1 + [90, 50]),
}
out = {}
for name, cells in cases.items():
    u = life.Universe(cells)
    u.step(300)
    out[name] = {"start": cells.tolist(), "end": u.cells.tolist()}
print(json.dumps(out))
`;
  let data = null;
  for (const exe of [process.env.PYTHON, "python", "python3", "py"].filter(Boolean)) {
    try { data = JSON.parse(execFileSync(exe, ["-c", py], { cwd: root, maxBuffer: 64 << 20 }).toString()); break; }
    catch (e) { /* try next */ }
  }
  if (!data) check("python comparison", false, "could not run python with gol/ (set PYTHON=...)");
  else {
    for (const [name, { start, end }] of Object.entries(data)) {
      const life = new Life();
      life.load(start);
      life.run(300);
      const got = asSet(life.cells()), want = asSet(end);
      check(`python: ${name}`, sameSet(got, want) && life.pop === want.size, `300 gens, pop ${got.size} vs ${want.size}`);
    }
  }
}

// --- 4. benchmark ------------------------------------------------------------
{
  const r = rng(7), cells = [];
  const dirs = ["SE", "SW", "NE", "NW"];
  for (let i = 0; i < 300; i++) cells.push(...glider(dirs[i % 4], Math.floor(r() * 4000), Math.floor(r() * 4000)));
  for (let i = 0; i < 300; i++) {
    const x = Math.floor(r() * 4000), y = Math.floor(r() * 4000);
    cells.push([x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]);   // blocks
  }
  const life = new Life();
  life.load(cells);
  const t0 = Date.now();
  life.run(5000);
  console.log(`bench: 300 gliders + 300 blocks over 4000x4000, 5000 gens in ${Date.now() - t0} ms ` +
    `(pop ${life.pop}, tiles ${life.tiles.size})`);
}

console.log(failures ? `${failures} FAILED` : "all tests passed");
process.exit(failures ? 1 : 0);
