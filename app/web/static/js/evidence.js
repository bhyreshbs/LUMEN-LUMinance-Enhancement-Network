// Evidence page. Every number comes from /static/data/evaluation.json, which scripts/hardening_study.py generates from the
// results files; nothing here is typed by hand. Charts are inline SVG marks inside HTML rows (no libraries).
import { initShell } from "./shell.js";

initShell();
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, props = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  n.append(...children.filter((c) => c !== null && c !== undefined));
  return n;
};
const svg = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
};
const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d) : v ?? "–");
const READ = { better: "good", worse: "bad", "not distinguishable": "flat" };

// one horizontal bar on a 0..max scale
function barRow(label, value, max, valueText, light) {
  const s = svg("svg", { viewBox: "0 0 100 10", preserveAspectRatio: "none", "aria-hidden": "true" });
  s.append(svg("rect", { class: "track", x: 0, y: 2, width: 100, height: 6, rx: 1 }),
           svg("rect", { class: light ? "rbar rbar--light" : "rbar", x: 0, y: 2, width: Math.max(0.5, (value / max) * 100), height: 6, rx: 1 }));
  return el("div", { class: "row" }, el("span", { class: "row__label" }, label), s, el("span", { class: "row__value", text: valueText }));
}

function table(headers, rows, numericFrom = 1) {
  const t = document.createDocumentFragment();
  t.append(el("thead", {}, el("tr", {}, ...headers.map((h, i) => el("th", { scope: "col", class: i >= numericFrom ? "num" : "", text: h })))));
  t.append(el("tbody", {}, ...rows.map((r) => el("tr", r.best ? { class: "is-best" } : {},
    ...r.cells.map((c, i) => (i === 0 ? el("th", { scope: "row", text: c }) : el("td", { class: i >= numericFrom ? "num" : "", text: c })))))));
  return t;
}

function headline(ev) {
  $("headSub").textContent = `${ev.dataset}, ${ev.images} test photos with references, one run of the final configuration. Higher is better.`;
  const max = Math.ceil(Math.max(...ev.methods.map((m) => m.psnr)) / 5) * 5;
  $("headChart").replaceChildren(
    el("div", { class: "rows" }, ...ev.methods.map((m) => barRow(el("b", { text: m.name }), m.psnr, max, `${fmt(m.psnr, 2)} dB`, m.best))),
    el("div", { class: "axisnote", "aria-hidden": "true" }, el("span", { text: "0 dB" }), el("span", { text: `${max} dB` })),
    el("figcaption", { text: "Peak signal-to-noise ratio against the reference photo, the most common accuracy measure for this task." }));
  $("evalTable").append(table(["Method", ...ev.columns.map((c) => c.label)],
    ev.methods.map((m) => ({ best: m.best, cells: [m.name, ...ev.columns.map((c) => fmt(m[c.key], c.digits))] }))));
  $("evalNote").textContent = ev.note;
}

// forest plot: one group per metric, each row a difference with its 95 % interval, zero dashed
function forest(rob) {
  const groups = [["psnr", "PSNR difference (dB)"], ["ssim", "SSIM difference"], ["lpips", "LPIPS difference (lower is better)"]];
  const out = [];
  for (const [key, title] of groups) {
    const rows = rob.rows.filter((r) => r.metric_key === key);
    if (!rows.length) continue;
    const MIN_SPAN = { psnr: 1, ssim: 0.01, lpips: 0.01 };   // a tiny interval must look tiny, not fill the chart
    let lo = Math.min(0, ...rows.map((r) => r.lo)), hi = Math.max(0, ...rows.map((r) => r.hi));
    if (hi - lo < MIN_SPAN[key]) { const mid = (hi + lo) / 2; lo = mid - MIN_SPAN[key] / 2; hi = mid + MIN_SPAN[key] / 2; }
    const pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;
    const x = (v) => ((v - lo) / (hi - lo)) * 100;
    out.push(el("p", { class: "group-title", text: title }));
    out.push(el("div", { class: "rows" }, ...rows.map((r) => {
      const tone = READ[r.reading] || "flat";
      const s = svg("svg", { viewBox: "0 0 100 10", preserveAspectRatio: "none", "aria-hidden": "true" });
      s.append(svg("line", { class: "zero", x1: x(0), x2: x(0), y1: 0, y2: 10 }),
               svg("line", { class: `ci ci--${tone}`, x1: x(r.lo), x2: x(r.hi), y1: 5, y2: 5 }),
               svg("rect", { class: `pt--${tone}`, x: x(r.mean) - 0.9, y: 2.5, width: 1.8, height: 5, rx: 0.4 }));
      return el("div", { class: "row" },
        el("span", { class: "row__label" }, el("b", { text: r.comparison }), el("br"), r.dataset),
        s,
        el("span", { class: "row__value" }, el("b", { text: r.difference }), ` ${r.interval}`, el("br"), r.reading));
    })));
    const ticks = el("div", { class: "row", "aria-hidden": "true" }, el("span"), el("div", { class: "ticks" }), el("span"));
    const d = key === "psnr" ? 1 : 3;
    for (const [v, txt] of [[lo, fmt(lo, d)], [0, "0"], [hi, fmt(hi, d)]]) {
      const t = el("span", { text: txt });
      t.style.left = `${x(v)}%`;
      ticks.children[1].append(t);
    }
    out.push(ticks);
  }
  $("forest").replaceChildren(...out);
  $("trustIntro").textContent = rob.intro;
  $("trustTable").append(table(["Test set", "Comparison (A − B)", "Metric", "Difference (95 % interval)", "A is"],
    rob.rows.map((r) => ({ cells: [r.dataset, r.comparison, r.metric, `${r.difference} ${r.interval}`, r.reading] })), 3));
  $("trustCaveat").textContent = rob.caveat;
  $("trust").hidden = false;
}

function noise(rob) {
  const bands = rob.noise_bands || [];
  if (!bands.length) { $("noiseText").textContent = "No noise breakdown is available."; return; }
  const max = Math.ceil(Math.max(...bands.flatMap((b) => Object.values(b.psnr))) / 5) * 5;
  const name = { classical: "Quick", retinexformer: "Natural", qgh: "Highlight-safe" };
  $("noiseChart").replaceChildren(
    el("div", { class: "bands" }, ...bands.map((b) => el("div", { class: "band" },
      el("h3", { class: "h4", text: `${b.label} (${b.n})` }),
      el("p", { class: "note", text: `Input noise σ ${fmt(b.noise_range[0], 3)} to ${fmt(b.noise_range[1], 3)}` }),
      el("div", { class: "rows" }, ...["classical", "retinexformer", "qgh"].map((k) =>
        barRow(name[k], b.psnr[k], max, `${fmt(b.psnr[k], 2)} dB`, k === "retinexformer"))),
      el("p", { class: "small" }, `Natural against Quick: ${b.natural_vs_quick_psnr.mean >= 0 ? "+" : ""}${fmt(b.natural_vs_quick_psnr.mean, 2)} dB `,
        `[${fmt(b.natural_vs_quick_psnr.lo, 2)}, ${fmt(b.natural_vs_quick_psnr.hi, 2)}], ${b.natural_vs_quick_psnr.reading}.`)))),
    el("figcaption", { text: "Development photos from a second public set, unseen by the model. PSNR against the reference, higher is better." }));
  const low = bands.find((b) => b.key === "low"), high = bands.find((b) => b.key === "high");
  if (low && high) {
    $("noiseText").textContent = `The test photos are all low in grain. In a second set we found ${high.n} photos with far more grain `
      + `(noise σ ${fmt(high.noise_range[0], 3)} to ${fmt(high.noise_range[1], 3)}). There, Natural is ${high.natural_vs_quick_psnr.reading} `
      + `compared with Quick, while on the ${low.n} low-grain photos it is clearly ${low.natural_vs_quick_psnr.reading}. `
      + "Highlight-safe, which also reduces grain, does better than Natural on the grainy ones. The workspace suggests it when a photo is that grainy.";
  }
}

function ablation(ev) {
  const max = Math.ceil(Math.max(...ev.ablation.map((a) => a.psnr)) / 5) * 5;
  $("ablChart").replaceChildren(
    el("div", { class: "rows" }, ...ev.ablation.map((a) => barRow(a.part, a.psnr, max, `${fmt(a.psnr, 2)} dB`, false))),
    el("figcaption", { text: "PSNR on the main test set with one part removed. Compare with the full methods above." }));
}

async function boot() {
  try {
    const ev = await (await fetch("/static/data/evaluation.json")).json();
    headline(ev);
    if (ev.robustness) {
      $("summary").replaceChildren(...(ev.robustness.summary || []).map((s) => el("li", { text: s })));
      forest(ev.robustness);
      noise(ev.robustness);
    }
    ablation(ev);
    $("findings").replaceChildren(...ev.findings.map((f) => el("li", { text: f })));
    document.body.dataset.ready = "1";
  } catch {
    $("headSub").textContent = "The evaluation data could not be loaded.";
  }
}

boot();
