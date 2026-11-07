// LUMEN "Under the hood" page: last restoration, evaluation, research lab, system status.

const $ = (id) => document.getElementById(id);
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
const kv = (pairs) => el("dl", { class: "kv" }, ...pairs.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)]));
const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d) : v ?? "–");
const yes = (b) => (b ? "yes" : "no");
const pct = (v) => `${(100 * v).toFixed(2)} %`;

function table(headers, rows, numericFrom = 1) {
  const t = document.createDocumentFragment();
  t.append(el("thead", {}, el("tr", {}, ...headers.map((h, i) => el("th", { scope: "col", class: i >= numericFrom ? "num" : "", text: h })))));
  t.append(el("tbody", {}, ...rows.map((r) => el("tr", r.best ? { class: "is-best" } : {},
    ...r.cells.map((c, i) => (i === 0 ? el("th", { scope: "row", text: c }) : el("td", { class: "num", text: c })))))));
  return t;
}

// ------------------------------------------------------------------ last restoration (sessionStorage)
function renderLast() {
  let last = null;
  try { last = JSON.parse(sessionStorage.getItem("lumen.last") || "null"); } catch { /* ignore */ }
  if (!last) return;
  const d = last.details || {};
  const t = d.timing_ms || {};
  const parts = [
    ["Photo", `${last.file || "photo"} (${last.width} × ${last.height} px processed)`],
    ["Style", `${last.name} · ${d.technical_label || ""}`],
    ["Model", d.model ? `${d.model} · ${d.checkpoint}` : "None (classical image processing)"],
    ["Device", d.model ? `${d.device}${d.fp16 ? " · half precision (FP16)" : ""}` : "CPU"],
    ["Time", [`server ${fmt(last.elapsed_ms, 0)} ms`, t.model !== undefined ? `model ${fmt(t.model, 0)} ms` : null,
              t.refine !== undefined ? `refinement ${fmt(t.refine, 0)} ms` : null, `in your browser ${fmt(last.client_ms, 0)} ms`]
              .filter(Boolean).join(" · ")],
  ];
  if (d.metrics) {
    parts.push(["Output checks", `noise σ ${fmt(d.metrics.noise_sigma, 4)} · edge preservation ${fmt(d.metrics.edge_preservation_vs_input, 3)} · ` +
                `over-exposed ${fmt(d.metrics.overexposed_percent, 2)} %`]);
  }
  if (d.note) parts.push(["Note", d.note]);
  const body = $("lastBody");
  body.replaceChildren(kv(parts));
  if (d.refinements) {
    const names = { noise: "Noise refinement", colour: "Colour correction", contrast: "Local contrast", highlight: "Highlight roll-off" };
    const rows = ["noise", "colour", "contrast", "highlight"].map((k) => {
      const r = d.refinements[k];
      const configured = (d.configured || []).includes(k);
      return { cells: [names[k], configured ? "on" : "off (calibration)", r ? yes(r.active) : "–", r ? pct(r.coverage) : "–", r ? pct(r.area_frac) : "–"] };
    });
    body.append(el("h3", { text: "Highlight-safe refinements" }),
      el("div", { class: "table-wrap" }, el("table", {}, table(["Refinement", "Enabled", "Applied", "Coverage", "Area > 50 %"], rows))),
      el("p", { class: "note", text: d.identity
        ? "No refinement was needed, so the result is exactly the Natural restoration."
        : `Applied only inside the detected regions: ${(d.applied || []).map((a) => names[a] || a).join(", ")}.` }));
  }
}

// ------------------------------------------------------------------ evaluation
async function renderEvaluation() {
  const ev = await (await fetch("/static/data/evaluation.json")).json();
  $("evalSub").textContent = `${ev.dataset}, ${ev.images} test images, one run of the final configuration.`;
  $("evalTable").append(table(["Method", ...ev.columns.map((c) => c.label)],
    ev.methods.map((m) => ({ best: m.best, cells: [m.name, ...ev.columns.map((c) => fmt(m[c.key], c.digits))] }))));
  $("evalNote").textContent = ev.note;
  $("ablationTable").append(table(["Variant", "PSNR", "SSIM", "LPIPS"],
    ev.ablation.map((a) => ({ cells: [a.part, fmt(a.psnr, 2), fmt(a.ssim, 4), fmt(a.lpips, 4)] }))));
  $("findings").replaceChildren(...ev.findings.map((f) => el("li", { text: f })));
}

// ------------------------------------------------------------------ system + research lab
const HYBRID_PARTS = ["classical_branch", "deep_branch", "guided_branch", "fusion", "post_denoise", "post_color"];

function renderSystem(status) {
  const t = status.technical || {};
  for (const n of document.querySelectorAll("#maxSide, .maxSideText")) n.textContent = t.max_side;
  const styleRows = status.styles.map((s) => [s.name, el("span", {}, el("span", { class: `status${s.available ? "" : " is-off"}`, text: s.available ? "available" : "unavailable" }),
    s.available ? null : el("span", { class: "note", text: ` ${s.reason}` }))]);
  $("systemBody").replaceChildren(kv([
    ...styleRows,
    ["Model", t.model ? `${t.model}` : "not loaded"],
    ["Checkpoint", t.checkpoint ? el("span", { class: "mono", text: (t.checkpoint_check || "").includes(t.checkpoint)
      ? t.checkpoint_check : `${t.checkpoint} · ${t.checkpoint_check}` }) : (t.checkpoint_check || "–")],
    ["Device", t.device ? `${t.device}${t.fp16 ? " · FP16" : ""}` : "–"],
    ["Highlight-safe config", t.qg_config ? el("span", { class: "mono", text: t.qg_config }) : "unavailable"],
    ["Processing size", `up to ${t.max_side} px (longest side)`],
    ["Upload limits", `${status.limits.max_file_mb} MB, ${status.limits.max_megapixels} megapixels · ${status.limits.formats}`],
    ["NIQE metric", t.niqe_available ? "installed" : "not installed (optional)"],
  ]));
}

function setupLab(status) {
  const sel = $("labMethod");
  const options = [{ id: "compare", label: "Compare all styles side by side", available: true },
                   ...status.research.map((r) => ({ ...r, label: `Research: ${r.label}` }))];
  sel.replaceChildren(...options.map((o) => el("option", { value: o.id, ...(o.available ? {} : { disabled: "" }),
    text: o.available ? o.label : `${o.label} (unavailable)` })));
  const abl = $("labAblation");
  abl.append(...HYBRID_PARTS.map((p) => el("label", { class: "chip" }, el("input", { type: "checkbox", value: p }), p.replace(/_/g, " "))));
  sel.addEventListener("change", () => { abl.hidden = !sel.value.startsWith("hybrid_"); });
  const unavailable = status.research.filter((r) => !r.available);
  if (unavailable.length) $("labStatus").textContent = unavailable.map((r) => `${r.label}: ${r.reason}`).join(" ");

  $("labForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("labFile").files[0];
    if (!file) return;
    const run = $("labRun");
    run.disabled = true;
    $("labStatus").textContent = "Running…";
    $("labGallery").replaceChildren();
    const body = new FormData();
    body.append("file", file, file.name);
    let url = "/api/compare";
    if (sel.value !== "compare") {
      url = "/api/research";
      body.append("method", sel.value);
      body.append("disabled", [...abl.querySelectorAll("input:checked")].map((i) => i.value).join(","));
    }
    try {
      const res = await fetch(url, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "The run failed.");
      const results = data.results || [data];
      const original = results[0].original;
      $("labGallery").replaceChildren(
        el("figure", {}, el("img", { src: original, alt: "Original (processed size)" }), el("figcaption", { text: "Original" })),
        ...results.map((r) => el("figure", {}, el("img", { src: r.restored, alt: r.name }),
          el("figcaption", { text: `${r.name} · ${fmt(r.details.timing_ms.total ?? r.elapsed_ms, 0)} ms` }))));
      $("labStatus").textContent = sel.value === "compare"
        ? "Natural and Highlight-safe share a single neural-network run."
        : "Research baseline output.";
    } catch (err) {
      $("labStatus").textContent = err.message;
    } finally {
      run.disabled = false;
    }
  });
}

// ------------------------------------------------------------------ section highlighting
function trackSections() {
  const links = [...document.querySelectorAll(".doc__nav a[href^='#']")];
  const obs = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (en.isIntersecting) links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === `#${en.target.id}`));
    }
  }, { rootMargin: "-30% 0px -60% 0px" });
  document.querySelectorAll(".card[id]").forEach((s) => obs.observe(s));
}

async function boot() {
  renderLast();
  trackSections();
  renderEvaluation().catch(() => ($("evalSub").textContent = "The evaluation data could not be loaded."));
  try {
    const status = await (await fetch("/api/status")).json();
    renderSystem(status);
    setupLab(status);
  } catch {
    $("systemBody").replaceChildren(el("div", { class: "empty", text: "The server status could not be loaded." }));
  }
}

boot();
