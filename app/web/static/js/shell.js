// Shared by every page: theme toggle, toast, offline notice.
const root = document.documentElement;
const media = window.matchMedia("(prefers-color-scheme: light)");
const current = () => root.dataset.theme || (media.matches ? "light" : "dark");

function syncThemeButton(btn) {
  const next = current() === "light" ? "dark" : "light";
  btn.setAttribute("aria-label", `Switch to the ${next} theme`);
  btn.title = `Switch to the ${next} theme`;
}

export function initShell() {
  const btn = document.getElementById("themeBtn");
  if (btn) {
    syncThemeButton(btn);
    btn.addEventListener("click", () => {
      const next = current() === "light" ? "dark" : "light";
      root.dataset.theme = next;
      try { localStorage.setItem("lumen.theme", next); } catch { /* not remembered, still applied */ }
      syncThemeButton(btn);
      document.dispatchEvent(new CustomEvent("lumen:theme", { detail: next }));
    });
    media.addEventListener("change", () => syncThemeButton(btn));
  }
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
