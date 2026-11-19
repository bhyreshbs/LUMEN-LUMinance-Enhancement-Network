// "How it works": the shared shell, plus this server's real processing size.
import { initShell } from "./shell.js";

initShell();
fetch("/api/status").then((r) => r.json()).then((st) => {
  const side = st.technical && st.technical.max_side;
  if (side) for (const n of document.querySelectorAll(".js-max-side")) n.textContent = String(side);
}).catch(() => { /* keep the documented default */ });
