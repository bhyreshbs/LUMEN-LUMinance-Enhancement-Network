// Develop: applies the change LUMEN made to each pixel, gradually, in exposure space:
//     now = (input + e) * gain^t - e,   gain = clamp((restored + e) / (input + e), 1/32, 64),   e = 1/255
// so light comes up where it was added and by how much (a cross-fade cannot show that). WebGL2 when available,
// otherwise Canvas2D. Also returns brightness histograms of the original, the current frame and the restored photo.

const E = 1 / 255;
const GMIN = 1 / 32;
const GMAX = 64;
const BINS = 64;
const HIST_SIDE = 200;   // histograms are computed on a small copy: cheap, and plenty for a 64-bin curve

const VERT = `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;
const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uIn; uniform sampler2D uOut; uniform float uT;
in vec2 uv; out vec4 color;
void main() {
  const float E = 1.0 / 255.0;
  vec3 x = texture(uIn, uv).rgb;
  vec3 y = texture(uOut, uv).rgb;
  vec3 g = clamp((y + E) / (x + E), 1.0 / 32.0, 64.0);
  color = vec4(clamp((x + E) * pow(g, vec3(uT)) - E, 0.0, 1.0), 1.0);
}`;

const loadImage = (src) => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = reject;
  img.src = src;
});

function pixels(img, maxSide) {
  const s = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * s)), h = Math.max(1, Math.round(img.naturalHeight * s));
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { w, h, data: ctx.getImageData(0, 0, w, h).data };
}

function histogram(rgba) {
  const hb = new Float32Array(BINS);
  for (let i = 0; i < rgba.length; i += 4) {
    const y = (0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2]) / 255;
    hb[Math.min(BINS - 1, Math.floor(y * BINS))] += 1;
  }
  return hb;
}

function developInto(out, base, lg, t) {
  for (let i = 0; i < base.length; i += 4) {
    out[i] = Math.min(255, Math.max(0, (base[i] * Math.exp(t * lg[i]) - E) * 255));
    out[i + 1] = Math.min(255, Math.max(0, (base[i + 1] * Math.exp(t * lg[i + 1]) - E) * 255));
    out[i + 2] = Math.min(255, Math.max(0, (base[i + 2] * Math.exp(t * lg[i + 2]) - E) * 255));
    out[i + 3] = 255;
  }
}

function prepare(inp, res) {
  const base = new Float32Array(inp.length), lg = new Float32Array(inp.length);
  for (let i = 0; i < inp.length; i++) {
    if ((i & 3) === 3) continue;
    const x = inp[i] / 255, y = res[i] / 255;
    base[i] = x + E;
    lg[i] = Math.log(Math.min(GMAX, Math.max(GMIN, (y + E) / (x + E))));
  }
  return { base, lg };
}

export function createDevelop(canvas) {
  let gl = null;
  try { gl = canvas.getContext("webgl2", { premultipliedAlpha: false, antialias: false }); } catch { gl = null; }
  let program = null, textures = [], ctx2d = null, full = null, small = null, ready = false;

  function initGL() {
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader");
      return s;
    };
    program = gl.createProgram();
    gl.attachShader(program, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("link");
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }
  if (gl) {
    try { initGL(); } catch { gl = null; }
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); gl = null; ready = false; }, { once: true });
  }

  function texture(img) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    return t;
  }

  return {
    get engine() { return gl ? "webgl2" : "canvas"; },
    async setImages(beforeUrl, afterUrl) {
      ready = false;
      const [a, b] = await Promise.all([loadImage(beforeUrl), loadImage(afterUrl)]);
      const si = pixels(a, HIST_SIDE), so = pixels(b, HIST_SIDE);
      small = { ...prepare(si.data, so.data), frame: new Uint8ClampedArray(si.data.length), hIn: histogram(si.data), hOut: histogram(so.data) };
      canvas.width = a.naturalWidth;
      canvas.height = a.naturalHeight;
      if (gl) {
        textures.forEach((t) => gl.deleteTexture(t));
        textures = [texture(a), texture(b)];
        gl.viewport(0, 0, canvas.width, canvas.height);
      } else {
        const fi = pixels(a, 1600), fo = pixels(b, 1600);
        canvas.width = fi.w; canvas.height = fi.h;
        ctx2d = ctx2d || canvas.getContext("2d");
        full = { ...prepare(fi.data, fo.data), image: ctx2d.createImageData(fi.w, fi.h) };
      }
      ready = true;
    },
    render(t) {
      if (!ready) return null;
      if (gl) {
        gl.useProgram(program);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, textures[0]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, textures[1]);
        gl.uniform1i(gl.getUniformLocation(program, "uIn"), 0);
        gl.uniform1i(gl.getUniformLocation(program, "uOut"), 1);
        gl.uniform1f(gl.getUniformLocation(program, "uT"), t);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      } else {
        developInto(full.image.data, full.base, full.lg, t);
        ctx2d.putImageData(full.image, 0, 0);
      }
      developInto(small.frame, small.base, small.lg, t);
      return { input: small.hIn, now: histogram(small.frame), restored: small.hOut };
    },
  };
}

// SVG path data for the three histogram curves, on a 300 x 64 box. sqrt counts so the dark spike does not flatten the rest.
export function histogramPaths(h) {
  const top = Math.max(...[h.input, h.now, h.restored].map((b) => Math.sqrt(Math.max(...b)))) || 1;
  const path = (b, area) => {
    let d = area ? "M0 64" : "";
    b.forEach((v, i) => {
      const x = (i / (BINS - 1)) * 300, y = 64 - (Math.sqrt(v) / top) * 60;
      d += `${d ? " L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    });
    return area ? `${d} L300 64 Z` : d;
  };
  return { input: path(h.input, false), now: path(h.now, true), restored: path(h.restored, false) };
}
