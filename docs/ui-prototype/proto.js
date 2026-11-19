// LUMEN UI prototype (design review only). Query flags: ?theme=light|dark, ?still (fixed positions for screenshots),
// ?mode=compare|lens|develop (result view).
const q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const root = document.documentElement;
if (q.get("theme")) root.dataset.theme = q.get("theme");
const still = q.has("still");

// ------------------------------------------------------------------ theme toggle
const themeBtn = $("themeBtn");
if (themeBtn) {
  const label = () => themeBtn.setAttribute("aria-label", `Switch to the ${root.dataset.theme === "light" ? "dark" : "light"} theme`);
  label();
  themeBtn.addEventListener("click", () => { root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light"; label(); });
}

// ------------------------------------------------------------------ landing: the light pool
const PHOTOS = {
  coffee_dark: [600, 400], chelsea_extremely_dark: [451, 300], astronaut_uneven_light: [512, 512], rocket_night_highlights: [640, 427],
};
const print = $("print");
if (print) {
  const box = $("stagebox");
  const target = { x: 0.58, y: 0.46 };
  const pos = { x: 0.58, y: 0.46 };
  let moved = false;
  const setRadius = () => print.style.setProperty("--r", `${Math.round(print.clientWidth * 0.21)}px`);
  const paint = () => {
    print.style.setProperty("--lx", `${(pos.x * 100).toFixed(2)}%`);
    print.style.setProperty("--ly", `${(pos.y * 100).toFixed(2)}%`);
    const pr = print.getBoundingClientRect(), br = box.getBoundingClientRect();
    box.style.setProperty("--sx", `${pr.left - br.left + pos.x * pr.width}px`);
    box.style.setProperty("--sy", `${pr.top - br.top + pos.y * pr.height}px`);
  };
  const choose = (id) => {
    $("printBefore").src = `/app/web/static/img/landing/${id}_before.jpg`;
    $("printAfter").src = `/app/web/static/img/landing/${id}_after.jpg`;
    print.style.setProperty("--ar", `${PHOTOS[id][0]} / ${PHOTOS[id][1]}`);
    for (const b of document.querySelectorAll("[data-photo]")) if (b.closest(".pick")) b.setAttribute("aria-pressed", String(b.dataset.photo === id));
  };
  for (const b of document.querySelectorAll("[data-photo]")) b.addEventListener("click", () => choose(b.dataset.photo));
  print.addEventListener("pointermove", (e) => {
    const r = print.getBoundingClientRect();
    target.x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    target.y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    moved = true;
  });
  setRadius();
  addEventListener("resize", setRadius);
  choose("coffee_dark");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let t0 = performance.now();
  const tick = (now) => {
    if (!moved && !still && !reduce) {           // idle: the light drifts slowly so the reveal explains itself
      const t = (now - t0) / 1000;
      target.x = 0.5 + 0.22 * Math.sin(t * 0.45);
      target.y = 0.48 + 0.16 * Math.sin(t * 0.7 + 1.1);
    }
    pos.x += (target.x - pos.x) * 0.16;            // softly damped, never jumpy
    pos.y += (target.y - pos.y) * 0.16;
    paint();
    if (!still) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ result view
for (const g of document.querySelectorAll(".gauge[data-g]")) g.style.setProperty("--g", `${g.dataset.g}%`);

const tabs = document.querySelectorAll("[data-mode]");
const views = { compare: $("modeCompare"), lens: $("modeLens"), develop: $("modeDevelop") };
const setMode = (m) => {
  for (const t of tabs) t.setAttribute("aria-selected", String(t.dataset.mode === m));
  for (const [k, v] of Object.entries(views)) if (v) v.hidden = k !== m;
  if ($("developBar")) $("developBar").hidden = m !== "develop";
};
for (const t of tabs) t.addEventListener("click", () => t.dataset.mode !== "trio" && setMode(t.dataset.mode));
if (tabs.length) setMode(q.get("mode") || "develop");

for (const id of ["modeCompare", "modeLens"]) {
  const el = $(id);
  if (!el) continue;
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 100, y = ((e.clientY - r.top) / r.height) * 100;
    if (id === "modeCompare") el.style.setProperty("--pos", `${x}%`);
    else { el.style.setProperty("--lx", `${x}%`); el.style.setProperty("--ly", `${y}%`); }
  });
}

// Develop: now = (in + e) * gain^t - e, with gain = (out + e) / (in + e) per pixel and channel - the real change, applied
// gradually in exposure space (not a cross-fade), so the light comes up where LUMEN added it.
const canvas = $("developCanvas");
if (canvas) {
  const load = (src) => new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = src; });
  Promise.all([load("/app/web/static/img/landing/coffee_dark_before.jpg"), load("/app/web/static/img/landing/coffee_dark_after.jpg")]).then(([a, b]) => {
    const w = a.naturalWidth, h = a.naturalHeight;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const read = (img) => { ctx.drawImage(img, 0, 0); return ctx.getImageData(0, 0, w, h).data; };
    const inn = read(a), out = read(b);
    const E = 1 / 255, n = w * h * 4;
    const base = new Float32Array(n), lg = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if ((i & 3) === 3) continue;
      const x = inn[i] / 255, y = out[i] / 255;
      base[i] = x + E;
      lg[i] = Math.log(Math.min(64, Math.max(1 / 32, (y + E) / (x + E))));
    }
    const frame = ctx.createImageData(w, h);
    const BINS = 64;
    const hist = (px) => {
      const hb = new Float32Array(BINS);
      for (let i = 0; i < n; i += 8) {
        const yv = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
        hb[Math.min(BINS - 1, Math.floor(yv * BINS))] += 1;
      }
      return hb;
    };
    const hIn = hist(inn), hOut = hist(out);
    const path = (hb, top, area) => {
      let d = area ? "M0 64" : "";
      hb.forEach((v, i) => { const x = (i / (BINS - 1)) * 300, y = 64 - (Math.sqrt(v) / top) * 60; d += `${d ? " L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`; });
      return area ? `${d} L300 64 Z` : d;
    };
    const render = (t) => {
      const d = frame.data;
      for (let i = 0; i < n; i += 4) {
        d[i] = Math.min(255, Math.max(0, (base[i] * Math.exp(t * lg[i]) - E) * 255));
        d[i + 1] = Math.min(255, Math.max(0, (base[i + 1] * Math.exp(t * lg[i + 1]) - E) * 255));
        d[i + 2] = Math.min(255, Math.max(0, (base[i + 2] * Math.exp(t * lg[i + 2]) - E) * 255));
        d[i + 3] = 255;
      }
      ctx.putImageData(frame, 0, 0);
      const hNow = hist(d);
      const top = Math.max(...[hIn, hOut, hNow].map((hb) => Math.sqrt(Math.max(...hb)))) || 1;
      $("hIn").setAttribute("d", path(hIn, top, false));
      $("hOut").setAttribute("d", path(hOut, top, false));
      $("hNow").setAttribute("d", path(hNow, top, true));
      $("developOut").textContent = `${Math.round(t * 100)}%`;
    };
    const range = $("developRange");
    range.addEventListener("input", () => render(range.value / 100));
    render(range.value / 100);
    document.body.dataset.ready = "1";
  });
} else {
  document.body.dataset.ready = "1";
}
