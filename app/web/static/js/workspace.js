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
  try {
    const res = await fetch("/api/restore", { method: "POST", body, signal: state.controller.signal });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(data.error || "The photo couldn't be restored."), { status: res.status, code: data.code });
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
