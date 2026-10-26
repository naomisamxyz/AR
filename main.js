const DEFAULTS = {
  distortion: 0.47,
  waveSpeed: 0.99,
  rippleScale: 0.97,
  edgeSensitivity: 0.52,
  glistenIntensity: 1.77,
  glintThreshold: 0.14,
  contrast: 1.17,
  brightness: 0.33,
  vignette: 0.36,
  grain: 0.07,
  murk: 0.41,
  saturation: 0.19,
  ghostOffset: 6.7,
  overlayOpacity: 0.21,
  focusBlur: 1.15,
};

const KEYS = Object.keys(DEFAULTS);
const STORAGE_KEY = "undertow_params_v9";

// cheap smooth pseudo-noise (sum of incommensurate sines) to drive slow,
// organic wave-direction/turbulence drift on the CPU instead of paying
// for simplex noise on the GPU for every pixel, every frame
function pnoise(t) {
  return Math.sin(t) * 0.5 + Math.sin(t * 2.17 + 1.3) * 0.25 + Math.sin(t * 4.06 + 2.1) * 0.125;
}
function smoothstepJS(e0, e1, x) {
  const v = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return v * v * (3 - 2 * v);
}

function loadParams() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULTS, ...parsed };
  } catch (e) {
    return { ...DEFAULTS };
  }
}

function saveParams(p) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch (e) {}
}

let params = loadParams();

const canvas = document.getElementById("gl");
const video = document.getElementById("camera");
const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" });

if (!gl) {
  document.body.innerHTML = '<div style="color:#fff;padding:40px;font-family:sans-serif">WebGL2 is not available in this browser.</div>';
  throw new Error("no webgl2");
}

let currentFacing = "environment";
let mirrored = false;

async function startCamera(facing) {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facing },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings ? track.getSettings() : {};
  mirrored = settings.facingMode === "user" || facing === "user";
}

async function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error("Shader compile error: " + log);
  }
  return shader;
}

async function fetchText(url) {
  const res = await fetch(url);
  return res.text();
}

async function linkProgram(gl, vsSrc, fsSrc) {
  const vs = await compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = await compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error("Program link error: " + gl.getProgramInfoLog(program));
  }
  return program;
}

// ---- focus-breathing blur, ported from the AR reference project: an
// asymmetric cycle (dwell sharp, snap to blurry, dwell blurry, ease back
// to sharp) that itself randomly switches between "cycling" and "locked
// sharp" for random stretches, so zero blur dominates overall while the
// blurry moments never feel metronomic ----
let focusMode = "locked";
let focusSpeed = 1.4;
let focusPhase = 0;
let focusModeEnd = 2 + Math.random() * 4;
let blurElapsed = 0;
let lastFrameMs = null;

function updateFocusBlur(maxBlurPx) {
  const nowMs = performance.now();
  const delta = lastFrameMs == null ? 0 : Math.min((nowMs - lastFrameMs) / 1000, 0.1);
  lastFrameMs = nowMs;
  blurElapsed += delta;

  if (blurElapsed >= focusModeEnd) {
    focusMode = focusMode === "cycle" ? "locked" : "cycle";
    if (focusMode === "cycle") {
      focusSpeed = 0.6 + Math.random() * 0.9;
      focusPhase = 0;
      // one short blip, not a sustained wobble
      focusModeEnd = blurElapsed + 1.0 + Math.random() * 1.4;
    } else {
      // sharp/no-blur dominates heavily — long rests between blips
      focusModeEnd = blurElapsed + 9 + Math.random() * 16;
    }
  }

  if (focusMode === "cycle") focusPhase += focusSpeed * delta;
  const cyclePos = (focusPhase / (2 * Math.PI)) % 1;
  const sharpDwell = 0.4, riseWidth = 0.05, blurDwell = 0.08;
  const fallStart = sharpDwell + riseWidth + blurDwell;
  let norm;
  if (cyclePos < sharpDwell) norm = 0;
  else if (cyclePos < sharpDwell + riseWidth) norm = (cyclePos - sharpDwell) / riseWidth;
  else if (cyclePos < fallStart) norm = 1;
  else {
    const t = (cyclePos - fallStart) / (1 - fallStart);
    norm = 1 - (3 * t * t - 2 * t * t * t);
  }
  const blur = focusMode === "locked" ? 0 : norm * maxBlurPx;
  canvas.style.filter = blur > 0.01 ? `blur(${blur.toFixed(2)}px)` : "";
}

async function init() {
  const [vsSrc, fsSrc] = await Promise.all([
    fetchText("shaders/vertex.glsl"),
    fetchText("shaders/fragment.glsl"),
  ]);

  const program = await linkProgram(gl, vsSrc, fsSrc);

  const quad = new Float32Array([-1, -1, 3, -1, -1, 3]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const loc = (name) => gl.getUniformLocation(program, name);
  const U = {
    uCamera: loc("uCamera"),
    uResolution: loc("uResolution"),
    uCameraAspectFit: loc("uCameraAspectFit"),
    uTime: loc("uTime"),
    uMirror: loc("uMirror"),
    uDistortion: loc("uDistortion"),
    uWaveSpeed: loc("uWaveSpeed"),
    uRippleScale: loc("uRippleScale"),
    uEdgeSensitivity: loc("uEdgeSensitivity"),
    uGlistenIntensity: loc("uGlistenIntensity"),
    uGlintThreshold: loc("uGlintThreshold"),
    uContrast: loc("uContrast"),
    uBrightness: loc("uBrightness"),
    uVignette: loc("uVignette"),
    uGrain: loc("uGrain"),
    uMurk: loc("uMurk"),
    uSaturation: loc("uSaturation"),
    uTurbulence: loc("uTurbulence"),
    uAngleA: loc("uAngleA"),
    uAngleB: loc("uAngleB"),
    uGhostOffset: loc("uGhostOffset"),
    uOverlayOpacity: loc("uOverlayOpacity"),
  };

  function resize() {
    // full native resolution — the shader is now a single cheap pass
    // (see the Jacobian-approximated warp in fragment.glsl), so there's
    // no need to render under the screen's real pixel density anymore
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(window.innerWidth * dpr);
    const h = Math.floor(window.innerHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener("resize", resize);
  resize();

  const startTime = performance.now();

  function frame() {
    requestAnimationFrame(frame);
    if (video.readyState < video.HAVE_CURRENT_DATA || video.videoWidth === 0) return;

    const now = (performance.now() - startTime) / 1000;

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);

    const screenAspect = canvas.width / canvas.height;
    const camAspect = video.videoWidth / video.videoHeight;
    let fitX = 1, fitY = 1;
    if (screenAspect > camAspect) {
      fitY = camAspect / screenAspect;
    } else {
      fitX = screenAspect / camAspect;
    }

    const turbRaw = pnoise(now * 0.035 + 9.17);
    const turbulence = 0.45 + smoothstepJS(-0.25, 0.55, turbRaw) * 1.15;
    const angleA = now * 0.11 + pnoise(now * 0.02 + 0.5) * 3.14159;
    const angleB = angleA * 1.37 + 2.4 + pnoise(now * 0.017 + 4.0) * 2.0;

    updateFocusBlur(params.focusBlur);

    // ---- the ghost fades in and out on its own slow, irregular cycle —
    // reuses the same organic pseudo-noise as the wave drift so it never
    // reads as a metronomic pulse, and regularly dips to fully invisible
    // rather than just dimming toward a nonzero floor ----
    // slower cycle, and biased so the ghost sits at zero far more often
    // than it shows — only the rare high points of the noise reach full
    // strength, mirroring the focus blur's "mostly off" character
    const ghostRaw = pnoise(now * 0.035 + 17.3);
    const ghostEnvelope = smoothstepJS(-0.1, 0.8, ghostRaw);
    const ghostOpacity = params.overlayOpacity * ghostEnvelope;

    // ---- the only shader draw this frame: straight to the screen,
    // single pass, single render target — the ghost overlay is just two
    // extra same-frame samples of the live camera, computed inside this
    // same draw (see shaders/fragment.glsl) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U.uCamera, 0);
    gl.uniform2f(U.uResolution, canvas.width, canvas.height);
    gl.uniform2f(U.uCameraAspectFit, fitX, fitY);
    gl.uniform1f(U.uTime, now);
    gl.uniform1f(U.uMirror, mirrored ? 1.0 : 0.0);
    gl.uniform1f(U.uDistortion, params.distortion);
    gl.uniform1f(U.uWaveSpeed, params.waveSpeed);
    gl.uniform1f(U.uRippleScale, params.rippleScale);
    gl.uniform1f(U.uEdgeSensitivity, params.edgeSensitivity);
    gl.uniform1f(U.uGlistenIntensity, params.glistenIntensity);
    gl.uniform1f(U.uGlintThreshold, params.glintThreshold);
    gl.uniform1f(U.uContrast, params.contrast);
    gl.uniform1f(U.uBrightness, params.brightness);
    gl.uniform1f(U.uVignette, params.vignette);
    gl.uniform1f(U.uGrain, params.grain);
    gl.uniform1f(U.uMurk, params.murk);
    gl.uniform1f(U.uSaturation, params.saturation);
    gl.uniform1f(U.uTurbulence, turbulence);
    gl.uniform1f(U.uAngleA, angleA);
    gl.uniform1f(U.uAngleB, angleB);
    gl.uniform1f(U.uGhostOffset, params.ghostOffset);
    gl.uniform1f(U.uOverlayOpacity, ghostOpacity);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  requestAnimationFrame(frame);
}

// ---- UI wiring ----
function wireControls() {
  for (const key of KEYS) {
    const el = document.getElementById("p_" + key);
    if (!el) continue;
    el.value = params[key];
    el.addEventListener("input", () => {
      params[key] = parseFloat(el.value);
      saveParams(params);
    });
  }

  document.getElementById("resetBtn").addEventListener("click", () => {
    params = { ...DEFAULTS };
    saveParams(params);
    for (const key of KEYS) {
      const el = document.getElementById("p_" + key);
      if (el) el.value = params[key];
    }
  });

  const panel = document.getElementById("panel");
  const gearBtn = document.getElementById("gearBtn");
  gearBtn.addEventListener("click", () => panel.classList.toggle("open"));

  document.getElementById("flipBtn").addEventListener("click", async () => {
    currentFacing = currentFacing === "environment" ? "user" : "environment";
    try {
      await startCamera(currentFacing);
    } catch (e) {
      console.error(e);
    }
  });
}

// ---- boot sequence gated on a user tap (camera permission + autoplay) ----
const startOverlay = document.getElementById("startOverlay");
document.getElementById("startBtn").addEventListener("click", async () => {
  startOverlay.classList.add("hidden");
  try {
    await startCamera(currentFacing);
  } catch (e) {
    alert("Camera access failed: " + e.message);
    startOverlay.classList.remove("hidden");
    return;
  }
  wireControls();
  init().catch((e) => {
    console.error(e);
    alert("Failed to start renderer: " + e.message);
  });
});
