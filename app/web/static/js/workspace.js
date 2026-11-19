// LUMEN workspace: upload -> choose style -> restore -> compare -> download.
// State machine on <main data-state>: empty | ready | working | done | error.

const $ = (id) => document.getElementById(id);
const els = {
  main: document.querySelector("main"), viewEmpty: $("viewEmpty"), viewWorkspace: $("viewWorkspace"),
  dropzone: $("dropzone"), fileInput: $("fileInput"), limitsText: $("limitsText"),
  examplesBlock: $("examplesBlock"), examples: $("examples"), notice: $("serviceNotice"),
  stage: $("stage"), noPreview: $("noPreview"),
  viewer: $("viewer"), imgAfter: $("imgAfter"), imgBefore: $("imgBefore"), handle: $("handle"),
  overlay: $("overlay"), elapsed: $("elapsed"),
  fileThumb: $("fileThumb"), fileName: $("fileName"), fileMeta: $("fileMeta"), changeBtn: $("changeBtn"),
  styles: $("styles"), restoreBtn: $("restoreBtn"), restoreLabel: $("restoreLabel"), cancelBtn: $("cancelBtn"),
  result: $("result"), successText: $("successText"), downloadBtn: $("downloadBtn"), resultMeta: $("resultMeta"),
  alert: $("alert"), alertTitle: $("alertTitle"), alertText: $("alertText"), alertActions: $("alertActions"),
  compareHint: $("compareHint"), live: $("live"),
  overlayTitle: $("overlayTitle"), queueNote: $("queueNote"), steps: $("steps"),
  report: $("report"), reportText: $("reportText"), stepsLog: $("stepsLog"),
};

const ACCEPT = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]);
const ACCEPT_EXT = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
const state = {
  status: null, file: null, fileUrl: null, style: "standard", results: new Map(), shown: null,
  controller: null, timer: null, limitBytes: 20 * 1024 * 1024, downloadUrl: null,
};

// ------------------------------------------------------------------ helpers
const announce = (msg) => { els.live.textContent = ""; requestAnimationFrame(() => (els.live.textContent = msg)); };
const formatBytes = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
// Decode a data: URL locally. (fetch() on data: URLs is blocked by the page's CSP, connect-src 'self'.)
function dataUrlToBlob(url) {
  const [head, b64] = url.split(",", 2);
  const mime = (head.match(/^data:([^;]+)/) || [])[1] || "application/octet-stream";
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
const baseName = (name) => (name || "photo").replace(/\.[^.]+$/, "").replace(/[^\w\-]+/g, "_").slice(0, 60) || "photo";

function setState(s) {
  els.main.dataset.state = s;
  const inWorkspace = s !== "empty";
  els.viewEmpty.hidden = inWorkspace;
  els.viewWorkspace.hidden = !inWorkspace;
  const working = s === "working";
  els.overlay.hidden = !working;
  els.cancelBtn.hidden = !working;
  els.restoreBtn.disabled = working;
  els.changeBtn.disabled = working;
  els.styles.disabled = working;
}

function showAlert(title, text, actions = []) {
  els.alertTitle.textContent = title;
  els.alertText.textContent = text;
  els.alertActions.replaceChildren(...actions.map(([label, fn, primary]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = primary ? "btn btn--primary" : "btn";
    b.textContent = label;
    b.addEventListener("click", fn);
    return b;
  }));
  els.alert.hidden = false;
  announce(`${title}. ${text}`);
  // on phones the alert sits below the style list: bring it on screen so the problem is never missed
  if (matchMedia("(max-width: 860px)").matches) {
    const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    requestAnimationFrame(() => els.alert.scrollIntoView({ block: "center", behavior: smooth ? "smooth" : "auto" }));
  }
}
const hideAlert = () => { els.alert.hidden = true; };

// ------------------------------------------------------------------ styles (enhancement options)
function renderStyles() {
  const legend = els.styles.querySelector("legend");
  els.styles.replaceChildren(legend);
  for (const s of state.status.styles) {
    const label = document.createElement("label");
    label.className = "style";
    label.innerHTML = `<input type="radio" name="style">
      <span class="style__box"><span class="style__dot" aria-hidden="true"></span>
      <span class="style__name"></span><p class="style__desc"></p></span>`;
    const input = label.querySelector("input");
    input.value = s.id;
    input.disabled = !s.available;
    input.checked = s.id === state.style;
    label.querySelector(".style__name").textContent = s.name;
    if (s.recommended) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = "Recommended";
      label.querySelector(".style__name").append(pill);
    }
    label.querySelector(".style__desc").textContent = s.available ? s.summary : "Not available right now.";
    input.addEventListener("change", () => { state.style = s.id; onStyleChange(); });
    els.styles.append(label);
  }
}

function firstAvailableStyle() {
  const ok = state.status.styles.filter((s) => s.available);
  return (ok.find((s) => s.id === state.style) || ok.find((s) => s.recommended) || ok[0] || {}).id;
}

function onStyleChange() {
  hideAlert();
  const cached = state.results.get(state.style);
  if (cached) { showResult(cached); return; }
  if (state.shown) {           // a result for another style is on screen: offer to restore with this style
    els.result.hidden = true;
    els.report.hidden = true;
    els.restoreBtn.hidden = false;
    const name = state.status.styles.find((s) => s.id === state.style).name;
    els.restoreLabel.textContent = `Restore with ${name}`;
  }
}

// ------------------------------------------------------------------ fitting the photo to the stage
const narrow = window.matchMedia("(max-width: 860px)");
function fitViewer() {
  const img = els.imgAfter;
  if (!img.naturalWidth || els.viewer.hidden) return;
  const pad = narrow.matches ? 0 : 32;
  const availW = els.stage.clientWidth - 2 * pad - (narrow.matches ? 24 : 0);
  const availH = narrow.matches ? window.innerHeight * 0.58 : els.stage.clientHeight - 2 * pad;
  const scale = Math.max(0.01, Math.min(availW / img.naturalWidth, availH / img.naturalHeight, 3));
  els.viewer.style.width = `${Math.floor(img.naturalWidth * scale)}px`;
  els.viewer.style.height = `${Math.floor(img.naturalHeight * scale)}px`;
}

function showPreview(ok) {
  els.viewer.hidden = !ok;
  els.noPreview.hidden = ok;
  if (ok) fitViewer();
}

// ------------------------------------------------------------------ comparison slider
function setPos(pct) {
  const p = Math.min(100, Math.max(0, pct));
  els.viewer.style.setProperty("--pos", `${p}%`);
  els.handle.setAttribute("aria-valuenow", String(Math.round(p)));
  els.handle.setAttribute("aria-valuetext", `${Math.round(p)}% original`);
}

function initSlider() {
  let dragging = false;
  const fromEvent = (e) => {
    const r = els.viewer.getBoundingClientRect();
    setPos(((e.clientX - r.left) / r.width) * 100);
  };
  els.viewer.addEventListener("pointerdown", (e) => {
    if (!els.viewer.classList.contains("is-compare") || e.button > 0) return;
    dragging = true;
    els.viewer.classList.add("is-dragging");
    els.viewer.setPointerCapture(e.pointerId);
    fromEvent(e);
  });
  els.viewer.addEventListener("pointermove", (e) => dragging && fromEvent(e));
  const stop = () => { dragging = false; els.viewer.classList.remove("is-dragging"); };
  els.viewer.addEventListener("pointerup", stop);
  els.viewer.addEventListener("pointercancel", stop);
  els.handle.addEventListener("keydown", (e) => {
    const now = Number(els.handle.getAttribute("aria-valuenow"));
    const step = e.shiftKey ? 10 : 2;
    const map = { ArrowLeft: now - step, ArrowDown: now - step, ArrowRight: now + step, ArrowUp: now + step,
                  Home: 0, End: 100, PageDown: now - 20, PageUp: now + 20 };
    if (e.key in map) { e.preventDefault(); setPos(map[e.key]); }
  });
}

// ------------------------------------------------------------------ choosing a photo
function validateFile(file) {
  if (!file) return "No file was selected.";
  if (/\.(heic|heif)$/i.test(file.name) || /hei[cf]/i.test(file.type))
    return "HEIC photos aren't supported yet. Please export the photo as JPEG or PNG and try again.";
  if (!(ACCEPT.has(file.type) || ACCEPT_EXT.test(file.name)))
    return "That file doesn't look like a photo. Please choose a JPEG, PNG, WebP, BMP or TIFF image.";
  if (file.size === 0) return "That file is empty. Please choose another photo.";
  if (file.size > state.limitBytes)
    return `This photo is ${formatBytes(file.size)}. The limit is ${formatBytes(state.limitBytes)} — please choose a smaller file.`;
  return null;
}

function selectFile(file) {
  const problem = validateFile(file);
  if (problem) {
    if (els.main.dataset.state === "empty") {
      els.notice.textContent = problem;
      els.notice.hidden = false;
      announce(problem);
    } else {
      showAlert("Can't use this file", problem);
    }
    return;
  }
  els.notice.hidden = true;
  if (state.controller) state.controller.abort();
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  state.results.clear();
  state.shown = null;
  state.style = firstAvailableStyle();
  renderStyles();
  els.fileName.textContent = file.name || "Pasted photo";
  els.fileMeta.textContent = formatBytes(file.size);
  els.fileThumb.classList.remove("is-missing");
  els.fileThumb.src = state.fileUrl;
  els.imgAfter.src = state.fileUrl;
  els.imgAfter.alt = "Your original photo";
  els.viewer.classList.remove("is-compare");
  els.result.hidden = true;
  els.report.hidden = true;
  els.compareHint.hidden = true;
  els.restoreBtn.hidden = false;
  els.restoreLabel.textContent = "Restore photo";
  hideAlert();
  setState("ready");
  els.restoreBtn.focus({ preventScroll: true });
  announce(`${els.fileName.textContent} selected. Choose a style and press Restore photo.`);
}

async function selectExample(ex) {
  try {
    const res = await fetch(ex.url);
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    selectFile(new File([blob], `${ex.title}.jpg`, { type: "image/jpeg" }));
  } catch {
    els.notice.textContent = "This example couldn't be loaded. Please try another one.";
    els.notice.hidden = false;
  }
}

// ------------------------------------------------------------------ progress and report
// The server reports each stage as it really finishes. Everything shown here comes from those reports:
// no stage is invented, and every sentence of the summary maps to one reported value.
const STEP_LABELS = {
  decoded: "Reading your photo",
  analysed: "Checking the lighting",
  restored: "Restoring brightness and detail",
  refined: "Protecting bright areas and colour",
};
const LIGHTING = { extremely_dark: "very dark", non_uniform: "unevenly lit", low_light: "dark", normal: "fairly bright" };
const seconds = (ms) => (ms < 100 ? "under 0.1 s" : `${(ms / 1000).toFixed(1)} s`);
const share = (fraction) => (fraction * 100 < 0.1 ? "under 0.1%" : `${(fraction * 100).toFixed(1)}%`);

const newRun = (style) => ({
  order: style === "classical" ? ["decoded", "analysed", "restored"] : Object.keys(STEP_LABELS),
  ms: new Map(), events: {}, last: 0, active: "decoded",
});

function renderSteps(list, run) {
  list.replaceChildren(...run.order.map((stage) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = STEP_LABELS[stage];
    li.append(label);
    if (run.ms.has(stage)) {
      li.className = "is-done";
      const time = document.createElement("span");
      time.className = "steps__time";
      time.textContent = seconds(run.ms.get(stage));
      li.append(time);
    } else if (stage === run.active) {
      li.className = "is-active";
    }
    return li;
  }));
}

function onStage(run, kind, d) {
  if (kind === "queued") {
    els.queueNote.textContent = d.ahead === 1 ? "Waiting for 1 other photo to finish…" : `Waiting for ${d.ahead} other photos to finish…`;
    els.queueNote.hidden = false;
    return;
  }
  if (!(kind in STEP_LABELS)) return;          // a stage this page does not know: ignore it
  els.queueNote.hidden = true;
  if (kind === "decoded" && Array.isArray(d.plan)) run.order = d.plan.filter((s) => s in STEP_LABELS);
  run.events[kind] = d;
  run.ms.set(kind, Math.max(0, d.t_ms - run.last));
  run.last = d.t_ms;
  run.active = run.order[run.order.indexOf(kind) + 1];
  els.overlayTitle.textContent = run.active ? `${STEP_LABELS[run.active]}…` : "Finishing up…";
  renderSteps(els.steps, run);
}

function buildReport(run, data) {
  const ev = run.events;
  const out = [];
  if (ev.analysed && ev.analysed.category) {
    out.push(`Your photo looked ${LIGHTING[ev.analysed.category] || "dark"} (average brightness ${Math.round(ev.analysed.mean_lum * 100)}%).`);
  }
  if (ev.restored) out.push(`The ${data.name} style lifted the brightness and brought back detail in ${seconds(ev.restored.ms)}.`);
  const r = ev.refined;
  if (r) {
    const hue = r.highlight_hue;
    const found = r.refinements || {};
    const applied = r.applied || [];
    if (hue && hue.applied) out.push(`Bright spots such as windows or lamps kept their original colour (${share(hue.coverage)} of the photo).`);
    if (applied.includes("highlight")) out.push(`Near-white areas were softened so they don't wash out (${share(found.highlight.coverage)} of the photo).`);
    if (applied.includes("noise")) out.push(`Grain in smooth areas was reduced (${share(found.noise.coverage)} of the photo).`);
    if (applied.includes("colour")) out.push(`Colours were nudged back towards the original (${share(found.colour.coverage)} of the photo).`);
    if (applied.includes("contrast")) out.push(`Flattened detail was sharpened (${share(found.contrast.coverage)} of the photo).`);
    if (r.identity === true && !(hue && hue.applied)) out.push("Nothing else needed fixing.");
    else if (r.refinements === undefined && hue && !hue.applied) out.push("No bright areas needed their colour protected.");
  }
  return out.join(" ");
}

function showReport(report) {
  if (!report || !report.text) { els.report.hidden = true; return; }
  els.reportText.textContent = report.text;
  renderSteps(els.stepsLog, { order: report.order, ms: report.ms, active: null });
  els.report.hidden = false;
}

// Server-Sent Events over fetch (EventSource cannot POST a file).
function parseSse(raw) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; } catch { return null; }
}

// Resolves with the finished result, or null if streaming is not available (the caller then uses the plain request).
async function streamRestore(body, signal, onEvent) {
  const res = await fetch("/api/restore/stream", { method: "POST", body, signal });
  if (!res.ok) {
    if (res.status === 404 || res.status === 405) return null;
    const data = await res.json().catch(() => ({}));
    throw Object.assign(new Error(data.error || "The photo couldn't be restored."), { status: res.status, code: data.code });
  }
  const streaming = (res.headers.get("content-type") || "").includes("text/event-stream");
  if (!streaming || !res.body || !res.body.getReader) return null;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let seen = 0;
  let result = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const ev = parseSse(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (!ev) continue;
      seen += 1;
      if (ev.event === "done") result = ev.data;
      else if (ev.event === "error") throw Object.assign(new Error(ev.data.error), { status: ev.data.status, code: ev.data.code });
      else onEvent(ev.event, ev.data);
    }
  }
  if (result) return result;
  if (seen === 0) return null;
  throw new Error("The connection was interrupted before the photo was finished.");
}

async function plainRestore(body, signal) {
  const res = await fetch("/api/restore", { method: "POST", body, signal });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.error || "The photo couldn't be restored."), { status: res.status, code: data.code });
  return data;
}

// ------------------------------------------------------------------ restoring
async function restore() {
  if (!state.file || !state.style) return;
  hideAlert();
  setState("working");
  const started = performance.now();
  els.elapsed.textContent = "0";
  state.timer = setInterval(() => (els.elapsed.textContent = String(Math.floor((performance.now() - started) / 1000))), 250);
  state.controller = new AbortController();
  const body = new FormData();
  body.append("file", state.file, state.file.name || "photo.jpg");
  body.append("style", state.style);
  const style = state.style;
  const run = newRun(style);
  els.report.hidden = true;
  els.queueNote.hidden = true;
  els.overlayTitle.textContent = "Restoring your photo…";
  els.steps.hidden = false;
  renderSteps(els.steps, run);
  try {
    let data = await streamRestore(body, state.controller.signal, (kind, d) => onStage(run, kind, d));
    if (data) {
      data.report = { text: buildReport(run, data), order: run.order.filter((s) => run.ms.has(s)), ms: run.ms };
    } else {                                     // streaming is not available here: the plain request still works
      els.steps.hidden = true;
      data = await plainRestore(body, state.controller.signal);
    }
    data.clientMs = performance.now() - started;
    state.results.set(style, data);
    rememberForDetails(data);
    if (style === state.style) showResult(data);
  } catch (err) {
    if (err.name === "AbortError") { setState(state.shown ? "done" : "ready"); announce("Restoration cancelled."); return; }
    setState("error");
    const retry = ["Try again", restore, true];
    const another = ["Choose another photo", () => els.fileInput.click()];
    if (err.status === 422 || err.status === 413) showAlert("This photo can't be restored", err.message, [another]);
    else if (err.status === 503) showAlert("LUMEN is busy", err.message, [retry]);
    else if (err.status === 409) {
      // the server's reason is technical (shown on the details page); refresh the style list so it can't be picked again
      showAlert("This style isn't available", "It can't be used on this server right now. Please choose another style.", []);
      fetch("/api/status").then((r) => r.json()).then((st) => {
        state.status = st;
        if (!st.styles.find((s) => s.id === state.style)?.available) state.style = firstAvailableStyle();
        renderStyles();
      }).catch(() => {});
    }
    else if (err.status) showAlert("Something went wrong", err.message, [retry, another]);
    else showAlert("Can't reach LUMEN", "Check your connection and try again.", [retry]);
  } finally {
    clearInterval(state.timer);
    state.controller = null;
  }
}

function showResult(data) {
  state.shown = data;
  els.imgBefore.src = data.original;        // server's processed original: pixel-aligned with the result
  els.imgAfter.src = data.restored;
  els.imgAfter.alt = `Restored photo (${data.name})`;
  if (els.fileThumb.classList.contains("is-missing")) {   // e.g. TIFF: the browser could not preview the upload
    els.fileThumb.classList.remove("is-missing");
    els.fileThumb.src = data.original;
  }
  els.viewer.classList.add("is-compare");
  setPos(50);
  els.restoreBtn.hidden = true;
  els.result.hidden = false;
  els.compareHint.hidden = false;
  showReport(data.report);
  els.successText.querySelector("span").textContent = `Restored with ${data.name}`;
  const secs = (data.clientMs ?? data.elapsed_ms) / 1000;
  els.resultMeta.textContent = `PNG · ${data.width} × ${data.height}${secs ? ` · took ${secs.toFixed(1)} s` : ""}`;
  if (state.downloadUrl) URL.revokeObjectURL(state.downloadUrl);
  state.downloadUrl = null;
  els.downloadBtn.href = data.restored;    // works immediately; swapped for a lighter blob URL below
  try {
    state.downloadUrl = URL.createObjectURL(dataUrlToBlob(data.restored));
    els.downloadBtn.href = state.downloadUrl;
  } catch { /* keep the data URL */ }
  els.downloadBtn.download = `${baseName(state.file?.name)}-lumen-${data.name.toLowerCase().replace(/[^a-z]+/g, "-")}.png`;
  setState("done");
  els.downloadBtn.focus({ preventScroll: true });
  announce(`Photo restored with ${data.name}. Drag across the photo to compare, or download it.`);
}

function rememberForDetails(data) {
  try {
    sessionStorage.setItem("lumen.last", JSON.stringify({
      at: new Date().toISOString(), file: state.file?.name, bytes: state.file?.size, style: data.style, name: data.name,
      width: data.width, height: data.height, elapsed_ms: data.elapsed_ms, client_ms: Math.round(data.clientMs),
      details: data.details,
    }));
  } catch { /* storage may be unavailable (private mode); details page will show its empty state */ }
}

const THUMB_PLACEHOLDER = "data:image/svg+xml," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#8a91a6" stroke-width="1.6" '
  + 'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/>'
  + '<path d="m3 16 5-5 4 4 3-3 6 6"/><circle cx="15.5" cy="9" r="1.5"/></svg>');

// ------------------------------------------------------------------ boot
async function boot() {
  initSlider();
  setState("empty");
  els.imgAfter.addEventListener("load", () => showPreview(true));
  els.imgAfter.addEventListener("error", () => { if (!state.shown) showPreview(false); });
  els.fileThumb.addEventListener("error", () => {
    // the browser cannot preview this file: show a neutral picture icon instead of a broken-image glyph
    if (els.fileThumb.src === THUMB_PLACEHOLDER) return;
    els.fileThumb.classList.add("is-missing");
    els.fileThumb.src = THUMB_PLACEHOLDER;
  });
  new ResizeObserver(fitViewer).observe(els.stage);
  window.addEventListener("resize", fitViewer);

  els.fileInput.addEventListener("change", () => { if (els.fileInput.files[0]) selectFile(els.fileInput.files[0]); els.fileInput.value = ""; });
  els.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); } });
  for (const target of [els.dropzone, els.viewWorkspace]) {
    target.addEventListener("dragover", (e) => { e.preventDefault(); els.dropzone.classList.add("is-over"); });
    target.addEventListener("dragleave", () => els.dropzone.classList.remove("is-over"));
    target.addEventListener("drop", (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("is-over");
      if (els.main.dataset.state !== "working" && e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
    });
  }
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === "file" && i.type.startsWith("image/"));
    if (item && els.main.dataset.state !== "working") selectFile(item.getAsFile());
  });
  els.changeBtn.addEventListener("click", () => els.fileInput.click());
  els.restoreBtn.addEventListener("click", restore);
  els.cancelBtn.addEventListener("click", () => state.controller?.abort());
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  try {
    const res = await fetch("/api/status");
    state.status = await res.json();
    state.limitBytes = state.status.limits.max_file_mb * 1024 * 1024;
    els.limitsText.textContent = `JPEG, PNG, WebP, BMP or TIFF · up to ${state.status.limits.max_file_mb} MB`;
    if (!state.status.styles.some((s) => s.available)) {
      els.notice.textContent = "Restoration is temporarily unavailable on this server. Please try again later.";
      els.notice.hidden = false;
    } else if (!state.status.styles.find((s) => s.recommended)?.available) {
      els.notice.textContent = "Some restoration styles are unavailable right now; the others still work.";
      els.notice.hidden = false;
    }
  } catch {
    els.notice.textContent = "LUMEN can't be reached right now. Please refresh the page in a moment.";
    els.notice.hidden = false;
    return;
  }
  try {
    const examples = await (await fetch("/api/examples")).json();
    els.examples.replaceChildren(...examples.map((ex) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "example";
      b.innerHTML = `<img alt="" loading="lazy"><span></span>`;
      b.querySelector("img").src = ex.url;
      b.querySelector("span").textContent = ex.title;
      b.setAttribute("aria-label", `Try the example photo: ${ex.title}`);
      b.addEventListener("click", () => selectExample(ex));
      li.append(b);
      return li;
    }));
    els.examplesBlock.hidden = examples.length === 0;
  } catch { /* examples are optional */ }
}

boot();
