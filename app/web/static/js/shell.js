// Shared by every page: theme toggle (with a smooth whole-page transition), back buttons, toast, offline notice.
const root = document.documentElement;
const media = window.matchMedia("(prefers-color-scheme: light)");
const current = () => root.dataset.theme || (media.matches ? "light" : "dark");
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function syncThemeButton(btn) {
  const next = current() === "light" ? "dark" : "light";
  btn.setAttribute("aria-label", `Switch to the ${next} theme`);
  btn.title = `Switch to the ${next} theme`;
}

function commitTheme(next) {
  root.dataset.theme = next;
  try { localStorage.setItem("lumen.theme", next); } catch { /* not remembered, still applied */ }
  // the browser chrome (mobile address bar) follows the chosen theme, not only the system one
  const bg = getComputedStyle(root).getPropertyValue("--bg").trim();
  if (bg) for (const m of document.querySelectorAll('meta[name="theme-color"]')) m.setAttribute("content", bg);
  document.dispatchEvent(new CustomEvent("lumen:theme", { detail: next }));
}

// The new theme spreads across the whole page as a circle of light from the button (View Transitions API, about
// 0.45 s). Browsers without it get a short colour cross-fade; reduced-motion users get an instant switch.
function applyTheme(next, from) {
  root.classList.add("theme-switched");   // lets the sun / moon icon turn in (not on page load)
  if (reducedMotion()) { commitTheme(next); return; }
  if (typeof document.startViewTransition === "function") {
    const r = from.getBoundingClientRect();
    const x = r.left + r.width / 2;
    const y = r.top + r.height / 2;
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    root.classList.add("vt-theme");
    const vt = document.startViewTransition(() => commitTheme(next));
    vt.ready.then(() => {
      root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 450, easing: "cubic-bezier(0.16, 1, 0.3, 1)", pseudoElement: "::view-transition-new(root)" });
    }).catch(() => { /* transition skipped: the theme is applied anyway */ });
    vt.finished.finally(() => root.classList.remove("vt-theme"));
    return;
  }
  root.classList.add("theme-fade");
  commitTheme(next);
  setTimeout(() => root.classList.remove("theme-fade"), 320);
}

// Back buttons on the content pages: return to where the visitor came from on this site, else to the start page.
// The site sends no referrer (privacy), so each page notes its own address in this tab's sessionStorage as it is left;
// nothing leaves the browser. A page opened from another LUMEN page in the same tab can then simply go back.
function initBackLinks() {
  let cameFromHere = false;
  try {
    const prev = sessionStorage.getItem("lumen.prev");
    cameFromHere = !!prev && prev !== location.pathname;
    window.addEventListener("pagehide", () => {
      try { sessionStorage.setItem("lumen.prev", location.pathname); } catch { /* storage unavailable */ }
    });
  } catch { /* storage unavailable: Back is a plain link to the start page */ }
  for (const a of document.querySelectorAll("[data-back]")) {
    if (!cameFromHere) {
      a.querySelector("span").textContent = "Back to LUMEN";
      continue;                                   // a plain link to "/"
    }
    a.addEventListener("click", (e) => {
      if (history.length > 1) { e.preventDefault(); history.back(); }
    });
  }
}

export function initShell() {
  const btn = document.getElementById("themeBtn");
  if (btn) {
    syncThemeButton(btn);
    btn.addEventListener("click", () => {
      applyTheme(current() === "light" ? "dark" : "light", btn);
      syncThemeButton(btn);
    });
    media.addEventListener("change", () => syncThemeButton(btn));
  }
  initBackLinks();
  const bar = document.querySelector(".bar");
  const offline = document.createElement("p");
  offline.className = "offline";
  offline.setAttribute("role", "status");
  offline.textContent = "You are offline. Restoring needs a connection; it will work again as soon as you are back online.";
  offline.hidden = navigator.onLine !== false;
  bar?.after(offline);
  window.addEventListener("offline", () => { offline.hidden = false; });
  window.addEventListener("online", () => { offline.hidden = true; });
}

let toastTimer = null;
export function toast(message) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.className = "toast";
    el.setAttribute("role", "status");
    document.body.append(el);
  }
  el.textContent = message;
  el.classList.add("is-on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("is-on"), 2600);
}
