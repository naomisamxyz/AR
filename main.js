// set to match the reference screenshot's slider positions
const DEFAULTS = {
  distortion: 1.34,
  waveSpeed: 1.94,
  rippleScale: 1.80,
  contrast: 1.58,
  brightness: 0.04,
  vignette: 0.44,
  grain: 0.08,
  murk: 0.43,
  saturation: 0.23,
  // line-doubling + glow, ported from AR1 (filter.frag) — values below
  // are AR1's own current defaults, carried over as-is
  double: 7.5,
  ghostOpacity: 0.5,
  angle: 180,
  drift: 0.77,
  ghostSpeed: 1.5,
  ghostWaveAmount: 6,
  ghostWaveScale: 2,
  ghostWaveSpeed: 6.87,
  glow: 2.5,
  focusBlur: 1.15,
  // overlay video layer (additive glow on black, see main.js/fragment.glsl)
  overlayOpacity: 0.54,
  // static authored placement, in screen-size-independent units (a
  // fraction of the shorter screen dimension, not raw canvas-uv — see
  // uOverlayOffset's use in fragment.glsl) — where the object sits
  // BEFORE any gyro pan is added. Negative X + positive Y is up-and-left
  // (vUv.x increases rightward, vUv.y increases UPWARD here).
  overlayOffsetX: -0.7,
  // separate Y offsets per screen shape — the conversion to actual
  // on-screen uv shift (see uOverlayOffset's use in fragment.glsl)
  // scales Y by shorterSide/height, which on its own doesn't move the
  // object up by the same amount on a tall portrait phone as on a wide
  // landscape screen for the same authored number, so these are tuned
  // independently rather than sharing one value
  overlayOffsetYPortrait: 0.69,
  overlayOffsetYLandscape: 0.38,
  overlayScale: 0.4,
  // strength of the separate shading/shadow layer's darkening (see
  // shadingVideo/uShading above and fragment.glsl) — 0 disables it
  shadingStrength: 0.10,
  // halo shadow sampled from the glow's own shape at a larger scale,
  // so it reaches past the object's edges — see uOverlayShadowStrength
  overlayShadowStrength: 0,
};

const KEYS = Object.keys(DEFAULTS);
const STORAGE_KEY = "ar4_params_v5";

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
// WebGL1, not 2 — ported so this runs on GPUs that only support GLES2
// (e.g. the Raspberry Pi 3's VideoCore IV). Needs the VAO extension
// since WebGL1 has no native vertex array objects.
const gl = canvas.getContext("webgl", { antialias: false, powerPreference: "high-performance" });

if (!gl) {
  document.body.innerHTML = '<div style="color:#fff;padding:40px;font-family:sans-serif">WebGL is not available in this browser.</div>';
  throw new Error("no webgl");
}

const vaoExt = gl.getExtension("OES_vertex_array_object");
if (!vaoExt) {
  document.body.innerHTML = '<div style="color:#fff;padding:40px;font-family:sans-serif">Required WebGL extension OES_vertex_array_object is not available.</div>';
  throw new Error("no OES_vertex_array_object");
}

let currentFacing = "environment";
let mirrored = false;

// ---- gyroscope-driven overlay pan (phone only) ----
// The idea: the overlay object is anchored to wherever the phone is
// pointing at the moment each overlay-video loop restarts, and as you
// physically turn the phone away from that heading the object's screen
// position shifts opposite to the rotation — exactly like it's sitting
// still in the room and you're looking away from it, then back. This
// is relative-rotation only (no compass/magnetometer, no real-world
// anchoring) — simple and good enough for a convincing "look away and
// back" illusion without the drift/complexity of true world tracking.
// disabled for now — flip back to true to restore it, nothing else
// needs to change (the permission request/listener just never attach,
// so latestOrientation stays null and panX/panY stay 0 everywhere)
const GYRO_ENABLED = false;
let latestOrientation = null; // {alpha, beta} in degrees, most recent deviceorientation reading
let panHome = null; // the orientation reading captured at the last loop restart
const PAN_SENSITIVITY = 0.012; // canvas-uv units per degree of rotation

function angleDeltaDeg(a, b) {
  // shortest signed difference a-b, wrapped into [-180, 180] — plain
  // subtraction breaks near the 0/360 wraparound point
  let d = a - b;
  d = ((d + 180) % 360 + 360) % 360 - 180;
  return d;
}

function onDeviceOrientation(e) {
  if (e.alpha == null || e.beta == null) return;
  latestOrientation = { alpha: e.alpha, beta: e.beta };
}

// iOS 13+ gates deviceorientation behind an explicit, gesture-triggered
// permission prompt; every other browser just fires the event with no
// such request. Resolves true when it's safe to add the listener.
async function requestOrientationPermission() {
  try {
    if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function") {
      const result = await DeviceOrientationEvent.requestPermission();
      return result === "granted";
    }
  } catch (e) {
    console.error(e);
  }
  return true;
}

// ---- tap/click interactivity, ported from AR2-dev: a local turbulence
// boost that travels outward from the tap point (see tapTurbBoost() in
// fragment.glsl), adapted to work on this project's existing
// distortion/turbulence system rather than AR2-dev's seam-network
// effect, which this project doesn't have ----
let tapPos = { x: 0.5, y: 0.5 };
let tapAtMs = -Infinity;
function triggerTap(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  tapPos = {
    x: (clientX - rect.left) / rect.width,
    y: 1 - (clientY - rect.top) / rect.height,
  };
  tapAtMs = performance.now();
}
canvas.addEventListener("pointerdown", (e) => {
  triggerTap(e.clientX, e.clientY);
});

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
  // GLSL ES 100 has no layout(location=...) qualifier, so the
  // attribute location has to be bound explicitly before linking
  gl.bindAttribLocation(program, 0, "aPos");
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
  const vao = vaoExt.createVertexArrayOES();
  vaoExt.bindVertexArrayOES(vao);
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

  // ---- overlay layer: a glow video rendered on black, composited
  // ADDITIVELY (see fragment.glsl) rather than chroma-keyed — a
  // greenscreen version was tried first, but a glow's soft edge blends
  // continuously into the key color, which showed up as a translucent
  // green halo no amount of threshold tuning could fully clear;
  // additive-on-black has no key color at all, so that problem doesn't
  // exist. assets/overlay_object.mp4 is the real footage, already
  // spliced into a single forward-then-reverse "boomerang" clip (see
  // the project's scratchpad for how it was built) so playback at
  // runtime is just one ordinary forward play, no reverse-scrubbing
  // tricks needed. ----
  const overlayVideo = document.createElement("video");
  overlayVideo.src = "assets/overlay_object.mp4";
  overlayVideo.muted = true;
  overlayVideo.playsInline = true;
  overlayVideo.loop = false; // looped manually (see onOverlayLoopRestart below) so we get a clean "it just looped" signal to re-anchor the gyro pan on
  // overlayAspect has a safe default (1) until loadedmetadata actually
  // fires, in the background — NOT awaited before continuing, since
  // that used to block the camera from ever appearing until the
  // (49-second, few-MB) overlay video's metadata had loaded, adding a
  // real startup delay before you'd see the camera at all. The frame
  // loop below also checks the video's own readyState before touching
  // it either way.
  let overlayAspect = 1;
  overlayVideo.addEventListener("loadedmetadata", () => {
    overlayAspect = overlayVideo.videoWidth / overlayVideo.videoHeight || 1;
  }, { once: true });
  overlayVideo.load();

  // a second, separate clip containing ONLY shadow/shading detail (no
  // glow, no object color) on a black background — its own luminance
  // is read directly as "how much to darken here" (black = none, grey
  // = darken by that amount), multiplied into the camera image BEFORE
  // the glow overlay is added on top. Played in lockstep with
  // overlayVideo (see scheduleOverlayPlay/ended below); its boomerang
  // length isn't frame-identical to overlayVideo's, so the two drift
  // slightly within a loop cycle, but both restart together each loop
  // so that drift never accumulates.
  const shadingVideo = document.createElement("video");
  shadingVideo.src = "assets/shading_layer.mp4";
  shadingVideo.muted = true;
  shadingVideo.playsInline = true;
  shadingVideo.loop = false;
  shadingVideo.load();

  // timeline, repeating forever: wait -> play forward+back (fading in
  // over the first second) -> fade out -> wait -> play again. All of
  // this is separate from the shader effects' own clock, which just
  // runs continuously and is untouched by any of it. The video has no
  // fade baked in on either end anymore, so both fades are done here.
  // Plain constants for now (wait "will be increased later", per the
  // request this was built for).
  const OVERLAY_WAIT_MS = 5000;
  const OVERLAY_FADE_IN_MS = 1000;
  const OVERLAY_FADE_OUT_MS = 5000;

  // null = not in that phase; set to a performance.now() timestamp the
  // instant that phase begins, so the per-frame fade math (see
  // overlayFadeMultiplier() in the frame loop) can measure elapsed time
  // against it
  let overlayPlayStartMs = null;
  let overlayEndedAtMs = null;

  // flips true for exactly one frame, right when playback actually
  // STARTS (not during the wait before it) — the frame loop below uses
  // that moment to re-anchor the gyro pan's "home" orientation to
  // wherever the phone currently happens to be pointing, matching
  // "whenever the loop starts from 0 the overlay is positioned in the
  // current view" exactly
  let overlayLoopRestarted = false;

  function scheduleOverlayPlay() {
    setTimeout(() => {
      overlayVideo.currentTime = 0;
      shadingVideo.currentTime = 0;
      overlayVideo
        .play()
        .then(() => {
          overlayPlayStartMs = performance.now();
          overlayEndedAtMs = null;
          overlayLoopRestarted = true;
        })
        .catch((e) => console.error("overlay video play failed:", e));
      shadingVideo.play().catch((e) => console.error("shading video play failed:", e));
    }, OVERLAY_WAIT_MS);
  }

  overlayVideo.addEventListener("ended", () => {
    overlayEndedAtMs = performance.now();
    shadingVideo.pause();
    setTimeout(scheduleOverlayPlay, OVERLAY_FADE_OUT_MS);
  });
  scheduleOverlayPlay(); // first play also waits, same as every loop after it

  // 0 before anything has ever played, ramps 0->1 over the first second
  // of each playback, stays at 1 until the clip ends, then ramps back
  // down to 0 over the fade-out window before the next wait begins
  function overlayFadeMultiplier(nowMs) {
    if (overlayEndedAtMs != null) {
      const t = (nowMs - overlayEndedAtMs) / OVERLAY_FADE_OUT_MS;
      return Math.max(0, 1 - Math.min(t, 1));
    }
    if (overlayPlayStartMs != null) {
      const t = (nowMs - overlayPlayStartMs) / OVERLAY_FADE_IN_MS;
      return Math.min(Math.max(t, 0), 1);
    }
    return 0;
  }

  const overlayTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, overlayTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const shadingTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, shadingTex);
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
    uOverlay: loc("uOverlay"),
    uOverlayAspectFit: loc("uOverlayAspectFit"),
    uOverlayOpacity: loc("uOverlayOpacity"),
    uOverlayOffset: loc("uOverlayOffset"),
    uOverlayScale: loc("uOverlayScale"),
    uShading: loc("uShading"),
    uShadingStrength: loc("uShadingStrength"),
    uOverlayShadowStrength: loc("uOverlayShadowStrength"),
    uDistortion: loc("uDistortion"),
    uWaveSpeed: loc("uWaveSpeed"),
    uRippleScale: loc("uRippleScale"),
    uContrast: loc("uContrast"),
    uBrightness: loc("uBrightness"),
    uVignette: loc("uVignette"),
    uGrain: loc("uGrain"),
    uMurk: loc("uMurk"),
    uSaturation: loc("uSaturation"),
    uTurbulence: loc("uTurbulence"),
    uAngleA: loc("uAngleA"),
    uAngleB: loc("uAngleB"),
    uDouble: loc("uDouble"),
    uGhostOpacity: loc("uGhostOpacity"),
    uAngle: loc("uAngle"),
    uDrift: loc("uDrift"),
    uGhostSpeed: loc("uGhostSpeed"),
    uGhostWaveAmount: loc("uGhostWaveAmount"),
    uGhostWaveScale: loc("uGhostWaveScale"),
    uGhostWaveSpeed: loc("uGhostWaveSpeed"),
    uGlow: loc("uGlow"),
    uTapPos: loc("uTapPos"),
    uTapTime: loc("uTapTime"),
  };

  // the shader is no longer the single cheap pass it was when
  // RENDER_SCALE=1 was first justified — line-doubling, the 6-direction
  // glow bloom (3-tap luminance average per direction = 18 extra
  // samples), the domain-warp noise, and the overlay/shading/halo
  // compositing have all stacked up since. Rendering at full native
  // resolution times device pixel ratio makes this fill-rate-bound on
  // weak GPUs (the Raspberry Pi 3 kiosk especially) — this scales the
  // actual render target down and lets the GPU/compositor upscale the
  // canvas back up to fill the screen via CSS, which is far cheaper
  // than running the full per-pixel cost at every physical pixel.
  const RENDER_SCALE = 0.65;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(window.innerWidth * dpr * RENDER_SCALE);
    const h = Math.floor(window.innerHeight * dpr * RENDER_SCALE);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener("resize", resize);
  resize();

  const startTime = performance.now();
  // `now` always starts at exactly 0 on every launch, which made the
  // turbulence/angle pseudo-noise below start at the exact same fixed
  // phase every single time — and that fixed phase happens to sit near
  // the top of the turbulence curve, which is why it always looked like
  // "a lot of distortion at start, settling down slowly" on every load,
  // not just an occasional fluke. A random phase offset, picked once
  // per page load, starts the noise at a random point in its cycle
  // instead, so it varies run to run like the rest of the drift does.
  const phaseOffset = Math.random() * 1000;

  // ---- framerate watchdog ----
  // If the device (the Pi, most likely) gets overworked, automatically
  // dial back the heaviest optional effects rather than just letting
  // everything stutter. A rolling average (not raw per-frame time)
  // avoids reacting to a single stutter, and a cooldown between changes
  // avoids flip-flopping if the average happens to hover right at a
  // threshold. Glow goes first (it's the heaviest thing in this
  // pipeline), then line-doubling if it's still struggling after that;
  // it recovers automatically, in reverse order, once the framerate
  // comes back up.
  let avgFrameMs = 16.7;
  let lastFrameTimeMs = null;
  // 0 = everything on, 1 = glow off, 2 = glow + line-doubling off,
  // 3 = also the overlay video layer (object + shading + halo shadow)
  // off, which additionally skips all three of ITS per-frame video
  // texture uploads below — those texImage2D calls aren't free either,
  // so this is worth having as a last resort on something as weak as
  // the Pi 3
  let perfDegradeLevel = 0;
  let lastDegradeChangeMs = 0;
  const PERF_BAD_MS = 45; // sustained average below ~22fps -> degrade further
  const PERF_GOOD_MS = 28; // sustained average above ~35fps -> recover a step
  const PERF_COOLDOWN_MS = 3000; // minimum time between degrade/recover steps

  function frame() {
    requestAnimationFrame(frame);
    if (video.readyState < video.HAVE_CURRENT_DATA || video.videoWidth === 0) return;

    const frameNowMs = performance.now();
    if (lastFrameTimeMs != null) {
      const dt = frameNowMs - lastFrameTimeMs;
      avgFrameMs += (dt - avgFrameMs) * 0.05; // exponential moving average
    }
    lastFrameTimeMs = frameNowMs;
    if (frameNowMs - lastDegradeChangeMs > PERF_COOLDOWN_MS) {
      if (avgFrameMs > PERF_BAD_MS && perfDegradeLevel < 3) {
        perfDegradeLevel++;
        lastDegradeChangeMs = frameNowMs;
      } else if (avgFrameMs < PERF_GOOD_MS && perfDegradeLevel > 0) {
        perfDegradeLevel--;
        lastDegradeChangeMs = frameNowMs;
      }
    }
    const effectiveGlow = perfDegradeLevel >= 1 ? 0 : params.glow;
    const overlayDegraded = perfDegradeLevel >= 3;
    const effectiveGhostOpacity = perfDegradeLevel >= 2 ? 0 : params.ghostOpacity;

    const now = (performance.now() - startTime) / 1000 + phaseOffset;

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);

    if (!overlayDegraded && overlayVideo.readyState >= overlayVideo.HAVE_CURRENT_DATA) {
      gl.bindTexture(gl.TEXTURE_2D, overlayTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, overlayVideo);
    }

    if (!overlayDegraded && shadingVideo.readyState >= shadingVideo.HAVE_CURRENT_DATA) {
      gl.bindTexture(gl.TEXTURE_2D, shadingTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, shadingVideo);
    }

    // re-anchor the gyro pan's "home" heading exactly once, on the
    // frame the overlay video just looped back to its start
    if (overlayLoopRestarted && latestOrientation) {
      panHome = { ...latestOrientation };
      overlayLoopRestarted = false;
    }
    // no manual toggle needed: a device that never fires a real
    // orientation event (the stationary Pi, or any desktop without a
    // gyro) just never populates latestOrientation/panHome, so pan
    // naturally stays at 0 forever on its own
    let panX = 0, panY = 0;
    if (latestOrientation && panHome) {
      const yawDelta = angleDeltaDeg(latestOrientation.alpha, panHome.alpha);
      const pitchDelta = latestOrientation.beta - panHome.beta;
      panX = yawDelta * PAN_SENSITIVITY;
      panY = -pitchDelta * PAN_SENSITIVITY;
    }

    const screenAspect = canvas.width / canvas.height;
    const camAspect = video.videoWidth / video.videoHeight;
    let fitX = 1, fitY = 1;
    if (screenAspect > camAspect) {
      fitY = camAspect / screenAspect;
    } else {
      fitX = screenAspect / camAspect;
    }

    let ovFitX = 1, ovFitY = 1;
    if (screenAspect > overlayAspect) {
      ovFitY = overlayAspect / screenAspect;
    } else {
      ovFitX = screenAspect / overlayAspect;
    }

    const turbRaw = pnoise(now * 0.035 + 9.17);
    // floor raised (0.45 -> 0.85) so the calm end of the cycle never
    // reads as "almost no distortion" the way it used to
    const turbulence = 0.85 + smoothstepJS(-0.25, 0.55, turbRaw) * 1.15;
    const angleA = now * 0.11 + pnoise(now * 0.02 + 0.5) * 3.14159;
    const angleB = angleA * 1.37 + 2.4 + pnoise(now * 0.017 + 4.0) * 2.0;

    updateFocusBlur(params.focusBlur);

    // ---- the only shader draw this frame: straight to the screen,
    // single pass, single render target — the line-doubling/ghost
    // overlay is just two extra same-frame samples of the live camera,
    // computed inside this same draw (see shaders/fragment.glsl) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U.uCamera, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, overlayTex);
    gl.uniform1i(U.uOverlay, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, shadingTex);
    gl.uniform1i(U.uShading, 2);
    gl.uniform2f(U.uResolution, canvas.width, canvas.height);
    gl.uniform2f(U.uCameraAspectFit, fitX, fitY);
    gl.uniform2f(U.uOverlayAspectFit, ovFitX, ovFitY);
    gl.uniform1f(U.uOverlayOpacity, overlayDegraded ? 0 : params.overlayOpacity * overlayFadeMultiplier(frameNowMs));
    // clamp only the X (side) offset so the object can't get pushed off
    // the left/right edge on resize — Y stays fully unclamped on every
    // screen shape, including portrait, so it can move as far up (or
    // down) as the slider value pushes it, off-frame if need be.
    const shortSidePx = Math.min(canvas.width, canvas.height);
    const maxOffsetX = Math.max(0, (canvas.width / shortSidePx - params.overlayScale) / 2);
    const clampedOffsetX = Math.min(Math.max(params.overlayOffsetX + panX, -maxOffsetX), maxOffsetX);
    const overlayOffsetY = canvas.height >= canvas.width
      ? params.overlayOffsetYPortrait
      : params.overlayOffsetYLandscape;
    gl.uniform2f(U.uOverlayOffset, clampedOffsetX, overlayOffsetY + panY);
    gl.uniform1f(U.uOverlayScale, params.overlayScale);
    gl.uniform1f(U.uShadingStrength, overlayDegraded ? 0 : params.shadingStrength * overlayFadeMultiplier(frameNowMs));
    gl.uniform1f(U.uOverlayShadowStrength, overlayDegraded ? 0 : params.overlayShadowStrength * overlayFadeMultiplier(frameNowMs));
    gl.uniform1f(U.uTime, now);
    gl.uniform1f(U.uMirror, mirrored ? 1.0 : 0.0);
    gl.uniform1f(U.uDistortion, params.distortion);
    gl.uniform1f(U.uWaveSpeed, params.waveSpeed);
    gl.uniform1f(U.uRippleScale, params.rippleScale);
    gl.uniform1f(U.uContrast, params.contrast);
    gl.uniform1f(U.uBrightness, params.brightness);
    gl.uniform1f(U.uVignette, params.vignette);
    gl.uniform1f(U.uGrain, params.grain);
    gl.uniform1f(U.uMurk, params.murk);
    gl.uniform1f(U.uSaturation, params.saturation);
    gl.uniform1f(U.uTurbulence, turbulence);
    gl.uniform1f(U.uAngleA, angleA);
    gl.uniform1f(U.uAngleB, angleB);
    gl.uniform1f(U.uDouble, params.double);
    gl.uniform1f(U.uGhostOpacity, effectiveGhostOpacity);
    gl.uniform1f(U.uAngle, params.angle);
    gl.uniform1f(U.uDrift, params.drift);
    gl.uniform1f(U.uGhostSpeed, params.ghostSpeed);
    gl.uniform1f(U.uGhostWaveAmount, params.ghostWaveAmount);
    gl.uniform1f(U.uGhostWaveScale, params.ghostWaveScale);
    gl.uniform1f(U.uGhostWaveSpeed, params.ghostWaveSpeed);
    gl.uniform1f(U.uGlow, effectiveGlow);
    gl.uniform2f(U.uTapPos, tapPos.x, tapPos.y);
    gl.uniform1f(U.uTapTime, tapAtMs === -Infinity ? -1.0 : (performance.now() - tapAtMs) / 1000);
    vaoExt.bindVertexArrayOES(vao);
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
  // best-effort — iOS Safari doesn't support the Fullscreen API at all
  // (there's no fullscreen to request there), and some browsers reject
  // it outside very specific conditions, so this is never allowed to
  // block or fail the rest of startup
  try {
    await document.documentElement.requestFullscreen();
  } catch (e) {
    console.error("requestFullscreen failed:", e);
  }
  // requested FIRST, before anything else awaits — iOS only honors the
  // permission prompt when it's called directly inside the click
  // handler's own gesture, not after some other async gap (like the
  // camera permission prompt) has already used it up
  if (GYRO_ENABLED && await requestOrientationPermission()) {
    window.addEventListener("deviceorientation", onDeviceOrientation);
  }
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
