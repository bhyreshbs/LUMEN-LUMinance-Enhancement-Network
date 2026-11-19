// Landing scene: a dark photo the visitor lights with pointer or touch. Inside the light, the same exposure blend as
// Develop (the real change LUMEN made), with t falling off with distance, so the reveal is a pool of real restoration,
// not a cut-out. WebGL2 when available; otherwise a CSS mask. With reduced motion or Save-Data: no idle drift.

const VERT = `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;
const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uIn; uniform sampler2D uOut; uniform vec2 uLight; uniform float uRadius; uniform float uAspect;
in vec2 uv; out vec4 color;
void main() {
  const float E = 1.0 / 255.0;
  vec2 d = (uv - uLight) * vec2(uAspect, 1.0);
  float t = 1.0 - smoothstep(uRadius * 0.35, uRadius, length(d));
  vec3 x = texture(uIn, uv).rgb;
  vec3 y = texture(uOut, uv).rgb;
  vec3 g = clamp((y + E) / (x + E), 1.0 / 32.0, 64.0);
  color = vec4(clamp((x + E) * pow(g, vec3(t)) - E, 0.0, 1.0), 1.0);
}`;

const loadImage = (src) => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = reject;
  img.src = src;
});

export async function initLanding() {
  const print = document.getElementById("print");
  const slab = document.getElementById("slab");
  const pick = document.getElementById("printPick");
  if (!print || !slab) return;
  let manifest;
  try { manifest = await (await fetch("/static/img/landing/manifest.json")).json(); } catch { slab.hidden = true; return; }
  const photos = manifest.photos || [];
  if (!photos.length) { slab.hidden = true; return; }

  const before = document.getElementById("printBefore");
  const after = document.getElementById("printAfter");
  const canvas = document.getElementById("printCanvas");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  const pos = { x: 0.58, y: 0.46 }, target = { x: 0.58, y: 0.46 };
  let touched = false, visible = true, raf = 0, t0 = performance.now();

  // ---- WebGL2, if the device has it
  let gl = null, prog = null, tex = [];
  if (!saveData) {
    try { gl = canvas.getContext("webgl2", { premultipliedAlpha: false, antialias: false }); } catch { gl = null; }
  }
  if (gl) {
    try {
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader");
        return s;
      };
      prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link");
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, "p");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); gl = null; print.classList.remove("is-gl"); }, { once: true });
    } catch { gl = null; }
  }

  const radius = () => 0.24;                     // of the print's height, in the shader's aspect-corrected space
  function size() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(print.clientWidth * dpr)), h = Math.max(1, Math.round(print.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    print.style.setProperty("--r", `${Math.round(print.clientHeight * radius() * 1.15)}px`);
  }
  function draw() {
    print.style.setProperty("--lx", `${(pos.x * 100).toFixed(2)}%`);
    print.style.setProperty("--ly", `${(pos.y * 100).toFixed(2)}%`);
    const pr = print.getBoundingClientRect(), sr = slab.getBoundingClientRect();
    slab.style.setProperty("--sx", `${pr.left - sr.left + pos.x * pr.width}px`);
    slab.style.setProperty("--sy", `${pr.top - sr.top + pos.y * pr.height}px`);
    if (!gl || tex.length < 2) return;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex[0]);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, tex[1]);
    gl.uniform1i(gl.getUniformLocation(prog, "uIn"), 0);
    gl.uniform1i(gl.getUniformLocation(prog, "uOut"), 1);
    gl.uniform2f(gl.getUniformLocation(prog, "uLight"), pos.x, pos.y);
    gl.uniform1f(gl.getUniformLocation(prog, "uRadius"), radius());
    gl.uniform1f(gl.getUniformLocation(prog, "uAspect"), print.clientWidth / Math.max(1, print.clientHeight));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ---- the loop runs only while something moves and the scene is on screen
  const drifting = () => !touched && !reduce && !saveData;
  function tick(now) {
    raf = 0;
    if (drifting()) {
      const t = (now - t0) / 1000;
      target.x = 0.5 + 0.22 * Math.sin(t * 0.45);
      target.y = 0.48 + 0.16 * Math.sin(t * 0.7 + 1.1);
    }
    const k = reduce ? 1 : 0.16;
    pos.x += (target.x - pos.x) * k;
    pos.y += (target.y - pos.y) * k;
    draw();
    const settled = Math.abs(target.x - pos.x) < 0.0005 && Math.abs(target.y - pos.y) < 0.0005;
    if (visible && !document.hidden && (drifting() || !settled)) raf = requestAnimationFrame(tick);
  }
  const wake = () => { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(tick); };
  new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; if (visible) wake(); }).observe(print);
  document.addEventListener("visibilitychange", wake);
  window.addEventListener("resize", () => { size(); wake(); });

  const point = (e) => {
    const r = print.getBoundingClientRect();
    target.x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    target.y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    touched = true;
    wake();
  };
  print.addEventListener("pointermove", point);
  print.addEventListener("pointerdown", point);      // a tap places the light (touch has no hover)

  async function choose(photo) {
    print.style.setProperty("--ar", `${photo.width} / ${photo.height}`);
    print.setAttribute("aria-label", `${photo.title}: a dark photo. Where the light falls you see the same photo restored by LUMEN.`);
    before.src = photo.before;
    after.src = photo.after;
    for (const b of pick.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === photo.id));
    size();
    if (gl) {
      try {
        const [a, b] = await Promise.all([loadImage(photo.before), loadImage(photo.after)]);
        tex.forEach((t) => gl.deleteTexture(t));
        tex = [a, b].map((img) => {
          const t = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          return t;
        });
        print.classList.add("is-gl");
      } catch { print.classList.remove("is-gl"); }
    }
    draw();
    wake();
  }

  pick.replaceChildren(...photos.map((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.id = p.id;
    b.setAttribute("aria-label", `Show the example: ${p.title}`);
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => choose(p));
    return b;
  }));
  if (new URLSearchParams(location.search).has("still")) touched = true;   // fixed light, for screenshots and tests
  await choose(photos.find((p) => p.id === "coffee_dark") || photos[0]);
  document.body.dataset.landing = gl ? "webgl2" : "css";
}
