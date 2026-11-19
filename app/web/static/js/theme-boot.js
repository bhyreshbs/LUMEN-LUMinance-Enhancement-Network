// Applies a remembered theme before the first paint (loaded as a plain script in <head>; no inline script under the CSP).
(function () {
  try {
    var t = window.localStorage.getItem("lumen.theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage unavailable: follow the system theme */ }
})();
