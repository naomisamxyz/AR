// On the Pi kiosk there's no devtools access, and even alert() may not
// render in some kiosk-mode Chromium setups — so any uncaught error
// (including inside the rAF-driven render loop, which is outside the
// init().catch() below entirely) gets written directly onto the page
// as plain text instead, where it's actually readable. This is purely
// a diagnostic safety net and changes no other behavior.
function showFatalError(msg) {
  let el = document.getElementById("fatalError");
  if (!el) {
    el = document.createElement("div");
    el.id = "fatalError";
    el.style.cssText =
      "position:fixed;inset:0;z-index:999;background:#000;color:#f55;" +
      "font:13px/1.5 monospace;padding:20px;white-space:pre-wrap;" +
      "overflow:auto;";
    document.body.appendChild(el);
  }
  el.textContent = "Fatal error:\n" + msg;
}
window.addEventListener("error", (e) => {
  const msg = (e.error && e.error.stack) || e.message || "Unknown error";
  showFatalError(msg);
});

// set to match the reference screenshot's slider positions
const DEFAULTS = {
  distortion: 1.34,
  waveSpeed: 2.6,
  rippleScale: 1.80,
  contrast: 1.58,
  brightness: 0.04,
  saturation: 0.23,
  // line-doubling + glow, ported from AR1 (filter.frag) — values below
  // are AR1's own current defaults, carried over as-is
  double: 2.5,
  ghostOpacity: 0.5,
  angle: 180,
  drift: 0.77,
  ghostSpeed: 1.5,
  ghostWaveAmount: 6,
  ghostWaveScale: 2,
  ghostWaveSpeed: 6.87,
  glow: 1.5,
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
  overlayOffsetYLandscape: 0.35,
  overlayScale: 0.5,
};

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

// Uploads a video element's current frame to a texture only when a new
// frame has actually arrived, instead of unconditionally re-uploading
// every requestAnimationFrame tick — rAF can run faster than the
// video's own frame rate, so most of those re-uploads would just be
// the same pixels again. The shader's own animation (distortion, time
// uniform, etc.) keeps running every rAF regardless; only the
// texImage2D upload itself is skipped. Uses requestVideoFrameCallback
// where supported (fires exactly once per decoded frame); falls back
// to comparing currentTime on a plain poll() call everywhere else.
// Camera switches and loop restarts both just look like "a new/changed
// frame arrived" to this, so they need no special-casing — and once a
// video pauses (e.g. the overlay's "ended" event), nothing marks it
// dirty again, which is exactly what keeps the last uploaded frame
// showing through its fade-out instead of going blank.
function createFrameWatcher(videoEl) {
  let dirty = true; // always upload the first available frame
  if (typeof videoEl.requestVideoFrameCallback === "function") {
    const onFrame = () => {
      dirty = true;
      videoEl.requestVideoFrameCallback(onFrame);
    };
    videoEl.requestVideoFrameCallback(onFrame);
    return {
      poll() {},
      consume() {
        if (!dirty) return false;
        dirty = false;
        return true;
      },
    };
  }
  let lastTime = -1;
  return {
    poll() {
      if (videoEl.currentTime !== lastTime) {
        lastTime = videoEl.currentTime;
        dirty = true;
      }
    },
    consume() {
      if (!dirty) return false;
      dirty = false;
      return true;
    },
  };
}

// no more UI to change these at runtime — tune by editing DEFAULTS
// above directly and reloading, instead of via sliders + localStorage
const params = DEFAULTS;

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

const piCameraMode = new URLSearchParams(location.search).get("pi") === "1";
let cameraSettings = {};

async function startCamera(facing) {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facing },
      ...(piCameraMode ? {
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        frameRate: { ideal: 30, max: 30 },
      } : {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      }),
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings ? track.getSettings() : {};
  cameraSettings = settings;
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

// dev-only A/B comparison: true swaps the glow ring to two opposing
// samples instead of four (half the texture reads, more directional/
// less round halo) — see GLOW_TWO_TAP in fragment.glsl. Four-sample
// stays the shipped default; flip this on only to compare.
const GLOW_TWO_TAP_TEST = false;

async function init() {
  const [vsSrc, fsSrc] = await Promise.all([
    fetchText("shaders/vertex.glsl"),
    fetchText("shaders/fragment.glsl"),
  ]);
  const fsSrcFinal = GLOW_TWO_TAP_TEST ? "#define GLOW_TWO_TAP\n" + fsSrc : fsSrc;

  const program = await linkProgram(gl, vsSrc, fsSrcFinal);
  // none of this ever changes frame to frame — there's only ever one
  // program and one VAO in use, uCamera/uOverlay always sample from
  // texture units 0/1, and there's only ever the default framebuffer
  // (this never renders to an offscreen target) — so it's all set up
  // once here instead of being redundantly re-issued to the GL driver
  // on every single frame
  gl.useProgram(program);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  const quad = new Float32Array([-1, -1, 3, -1, -1, 3]);
  const vao = vaoExt.createVertexArrayOES();
  vaoExt.bindVertexArrayOES(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  // bound to texture unit 0 for the program's whole lifetime — nothing
  // else ever touches unit 0, so this never needs re-binding in the
  // draw call below, only when actually uploading a new frame (see
  // camFrameWatcher.consume() further down, which explicitly sets
  // activeTexture(0) again before its bind for exactly this reason)
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
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

  const camFrameWatcher = createFrameWatcher(video);
  const overlayFrameWatcher = createFrameWatcher(overlayVideo);

  // timeline, repeating forever: wait -> play forward+back (fading in
  // over the first second) -> fade out -> wait -> play again. All of
  // this is separate from the shader effects' own clock, which just
  // runs continuously and is untouched by any of it. The video has no
  // fade baked in on either end anymore, so both fades are done here.
  const OVERLAY_WAIT_MIN_MS = 5000;
  const OVERLAY_WAIT_MAX_MS = 15000;
  function randomOverlayWaitMs() {
    return OVERLAY_WAIT_MIN_MS + Math.random() * (OVERLAY_WAIT_MAX_MS - OVERLAY_WAIT_MIN_MS);
  }
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

  function scheduleOverlayPlay(waitMs) {
    setTimeout(() => {
      overlayVideo.currentTime = 0;
      overlayVideo
        .play()
        .then(() => {
          overlayPlayStartMs = performance.now();
          overlayEndedAtMs = null;
          overlayLoopRestarted = true;
        })
        .catch((e) => console.error("overlay video play failed:", e));
    }, waitMs != null ? waitMs : randomOverlayWaitMs());
  }

  overlayVideo.addEventListener("ended", () => {
    overlayEndedAtMs = performance.now();
    setTimeout(() => scheduleOverlayPlay(), OVERLAY_FADE_OUT_MS);
  });
  // shorter, fixed wait just for the very first play on page load —
  // every loop after this one goes back to the normal random 5-15s wait
  const FIRST_OVERLAY_WAIT_MS = 5000;
  scheduleOverlayPlay(FIRST_OVERLAY_WAIT_MS);

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

  // same deal as `tex` above, but on unit 1
  const overlayTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, overlayTex);
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
    uDistortion: loc("uDistortion"),
    uWaveSpeed: loc("uWaveSpeed"),
    uRippleScale: loc("uRippleScale"),
    uContrast: loc("uContrast"),
    uBrightness: loc("uBrightness"),
    uSaturation: loc("uSaturation"),
    uTurbulence: loc("uTurbulence"),
    uWaveDirA: loc("uWaveDirA"),
    uWaveDirB: loc("uWaveDirB"),
    uWarpRot: loc("uWarpRot"),
    uGhostBaseOffset: loc("uGhostBaseOffset"),
    uRefResolution: loc("uRefResolution"),
    uGhostOpacity: loc("uGhostOpacity"),
    uGhostWaveAmount: loc("uGhostWaveAmount"),
    uGhostWaveScale: loc("uGhostWaveScale"),
    uGhostWaveSpeed: loc("uGhostWaveSpeed"),
    uGlow: loc("uGlow"),
    uBloomOffsetH: loc("uBloomOffsetH"),
    uBloomOffsetV: loc("uBloomOffsetV"),
    uTapPos: loc("uTapPos"),
    uTapTime: loc("uTapTime"),
  };
  // fixed for the program's whole lifetime — set once instead of every
  // frame (the program is already current from the gl.useProgram(program)
  // call right after linking above)
  gl.uniform1i(U.uCamera, 0);
  gl.uniform1i(U.uOverlay, 1);

  // the shader is no longer the single cheap pass it was when
  // RENDER_SCALE=1 was first justified — line-doubling, the glow
  // bloom, the domain-warp noise, and the overlay compositing have all
  // stacked up since. Rendering at full native resolution times device
  // pixel ratio makes this fill-rate-bound on weak GPUs (the Raspberry
  // Pi 3 kiosk especially) — this scales the actual render target down
  // and lets the GPU/compositor upscale the canvas back up to fill the
  // screen via CSS (still 100vw/100vh — the page itself stays
  // fullscreen at whatever the real display resolution is, e.g. the
  // Pi's 800x480 touchscreen; only the WebGL render TARGET shrinks),
  // which is far cheaper than running the full per-pixel cost at every
  // physical pixel.
  //
  // The watchdog below (see perfDegradeLevel) now tries these steps,
  // in order, BEFORE touching glow/line-doubling/the overlay — a
  // softer render still has the full look (nothing visually disabled),
  // and recovering back up is just as cheap, so it's tried first.
  // uDouble's offset and the glow radius are both already expressed as
  // a UV fraction of the canvas (divided by uResolution in the
  // shader), which is why the apparent on-screen width of the doubled
  // lines and the glow stays the same across all of these steps — the
  // canvas is always stretched to fill the same physical screen
  // regardless of its internal resolution, so a fixed UV fraction maps
  // to the same fraction of the final displayed image either way.
  const RENDER_SCALE_STEPS = piCameraMode
    ? [0.45, 0.40, 0.35, 0.28]
    : [0.45, 0.40, 0.35, 0.28, 0.22];
  let renderScaleStepIndex = 0;

  // FIXED reference render-target size — always the dimensions the
  // 0.45 (un-degraded) scale step would produce at the current window
  // size, regardless of which step is actually active right now. Any
  // "pixel-sized" effect magnitude (the ghost offset, the glow radius,
  // ghostWaves' jitter amount) is computed against THIS instead of the
  // live canvas.width/height — otherwise, since the canvas is always
  // CSS-stretched to the same physical screen size no matter its
  // internal resolution, dividing by a SMALLER live resolution (as the
  // watchdog drops scale) would make those offsets cover a LARGER
  // fraction of the screen, growing the apparent width of the doubled
  // lines and the glow exactly when the watchdog is trying to make
  // things cheaper, not bigger. Only updates when the window itself
  // resizes, not when the watchdog changes steps.
  let refWidth = 1, refHeight = 1;
  function updateRefDims() {
    const dpr = window.devicePixelRatio || 1;
    refWidth = Math.max(1, Math.floor(window.innerWidth * dpr * RENDER_SCALE_STEPS[0]));
    refHeight = Math.max(1, Math.floor(window.innerHeight * dpr * RENDER_SCALE_STEPS[0]));
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const scale = RENDER_SCALE_STEPS[renderScaleStepIndex];
    const w = Math.floor(window.innerWidth * dpr * scale);
    const h = Math.floor(window.innerHeight * dpr * scale);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    updateRefDims();
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
  // 0-2 = render scale steps (0.45 -> 0.40 -> 0.35, see
  // RENDER_SCALE_STEPS above) — tried FIRST since they keep the full
  // look (nothing visually disabled) just at a softer resolution.
  // 3 = glow off, 4 = also line-doubling off, 5 = also the overlay
  // video layer off (which additionally skips its per-frame video
  // texture upload — that texImage2D call isn't free either). Each
  // step recovers automatically, in reverse order, once the framerate
  // comes back up.
  let perfDegradeLevel = 0;
  let lastDegradeChangeMs = 0;
  const PERF_BAD_MS = 45; // sustained average below ~22fps -> degrade further
  const PERF_GOOD_MS = 28; // sustained average above ~35fps -> recover a step
  const PERF_COOLDOWN_MS = 3000; // minimum time between degrade/recover steps
  const PERF_MAX_LEVEL = RENDER_SCALE_STEPS.length - 1 + 3; // render-scale steps + glow/lines/overlay

  // cached across frames — see the fitKey check inside frame() below;
  // these only get recomputed when canvas size, camera/overlay aspect,
  // or mirroring actually changes, not on every single frame
  let lastFitKey = null;
  let fitX = 1, fitY = 1, ovFitX = 1, ovFitY = 1;
  let bloomOffsetH = [0, 0], bloomOffsetV = [0, 0];

  // temporary Pi-debugging readout (see index.html's #debugInfo)
  const debugEl = document.getElementById("debugInfo");
  let lastDebugUpdateMs = 0;

  // dev-only A/B comparison, untested on the Pi from here: true
  // allocates each texture's storage once (via texImage2D) and reuses
  // it on every later frame via texSubImage2D, instead of calling
  // texImage2D (which can internally reallocate) on every upload.
  // Whichever call is actually faster depends on the GPU driver, which
  // I can't benchmark without the real hardware — texImage2D stays the
  // default/baseline until you've confirmed sub-image is actually
  // faster on the Pi itself.
  const USE_TEX_SUBIMAGE = false;
  const uploadDims = { cam: null, overlay: null }; // {w,h} of whatever's currently allocated in each texture, or null before the first upload
  function uploadVideoFrame(videoEl, key) {
    if (!USE_TEX_SUBIMAGE) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, videoEl);
      return;
    }
    const w = videoEl.videoWidth, h = videoEl.videoHeight;
    const allocated = uploadDims[key];
    if (!allocated || allocated.w !== w || allocated.h !== h) {
      // first frame, or the source changed size (e.g. a camera switch
      // to a different resolution) — (re)allocate storage
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, videoEl);
      uploadDims[key] = { w, h };
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGB, gl.UNSIGNED_BYTE, videoEl);
    }
  }

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
      if (avgFrameMs > PERF_BAD_MS && perfDegradeLevel < PERF_MAX_LEVEL) {
        perfDegradeLevel++;
        lastDegradeChangeMs = frameNowMs;
      } else if (avgFrameMs < PERF_GOOD_MS && perfDegradeLevel > 0) {
        perfDegradeLevel--;
        lastDegradeChangeMs = frameNowMs;
      }
    }
    const newRenderScaleStepIndex = Math.min(perfDegradeLevel, RENDER_SCALE_STEPS.length - 1);
    if (newRenderScaleStepIndex !== renderScaleStepIndex) {
      renderScaleStepIndex = newRenderScaleStepIndex;
      resize();
    }
    const effectiveGlow = perfDegradeLevel >= RENDER_SCALE_STEPS.length ? 0 : params.glow;
    const effectiveGhostOpacity = perfDegradeLevel >= RENDER_SCALE_STEPS.length + 1 ? 0 : params.ghostOpacity;
    const overlayDegraded = perfDegradeLevel >= RENDER_SCALE_STEPS.length + 2;

    // temporary Pi-debugging readout — updated a few times a second
    // (not every frame) so the DOM write itself doesn't add meaningful
    // cost; remove debugEl/this block once performance is sorted
    if (debugEl && frameNowMs - lastDebugUpdateMs > 500) {
      lastDebugUpdateMs = frameNowMs;
      const fps = avgFrameMs > 0 ? (1000 / avgFrameMs).toFixed(1) : "?";
      debugEl.textContent =
        `fps: ${fps}\n` +
        `avgFrameMs: ${avgFrameMs.toFixed(1)}\n` +
        `renderScale: ${RENDER_SCALE_STEPS[renderScaleStepIndex]} (step ${renderScaleStepIndex}/${RENDER_SCALE_STEPS.length - 1})\n` +
        `perfDegradeLevel: ${perfDegradeLevel}/${PERF_MAX_LEVEL}\n` +
        `canvas: ${canvas.width}x${canvas.height}\n` +
        `camera: ${cameraSettings.width ?? video.videoWidth}x${cameraSettings.height ?? video.videoHeight} @ ${cameraSettings.frameRate ?? "?"}`;
    }

    const now = (performance.now() - startTime) / 1000 + phaseOffset;

    camFrameWatcher.poll();
    overlayFrameWatcher.poll();

    if (camFrameWatcher.consume()) {
      // activeTexture set explicitly here (not left to whatever the
      // previous frame's draw call left it on) so unit 0 always holds
      // `tex` afterward, and the per-frame draw call below never has
      // to re-bind it itself
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      uploadVideoFrame(video, "cam");
    }

    if (!overlayDegraded && overlayVideo.readyState >= overlayVideo.HAVE_CURRENT_DATA && overlayFrameWatcher.consume()) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, overlayTex);
      uploadVideoFrame(overlayVideo, "overlay");
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

    // fitX/fitY/ovFitX/ovFitY/bloomOffsetH/V all depend only on canvas
    // size, camera/overlay aspect, and mirroring — NONE of them depend
    // on time — so they're only actually recomputed when one of those
    // changes, instead of redoing the same arithmetic (and re-deriving
    // bloomOffsetH/V from it) on every single frame regardless.
    const fitKey = canvas.width + "x" + canvas.height + ":" + video.videoWidth + "x" + video.videoHeight
      + ":" + overlayAspect + ":" + mirrored;
    if (fitKey !== lastFitKey) {
      lastFitKey = fitKey;
      const screenAspect = canvas.width / canvas.height;
      const camAspect = video.videoWidth / video.videoHeight;
      fitX = 1; fitY = 1;
      if (screenAspect > camAspect) {
        fitY = camAspect / screenAspect;
      } else {
        fitX = screenAspect / camAspect;
      }

      ovFitX = 1; ovFitY = 1;
      if (screenAspect > overlayAspect) {
        ovFitY = overlayAspect / screenAspect;
      } else {
        ovFitX = screenAspect / overlayAspect;
      }

      // precomputed glow bloom tap offsets — see bloomTapOffset()/
      // uBloomOffsetH/V in fragment.glsl for why this used to be
      // redone on the GPU for every one of the 4 taps. Divided by the
      // FIXED reference resolution (see refWidth/refHeight above), not
      // the live canvas size, for the same reason uGhostBaseOffset and
      // ghostWaves() are below — otherwise the glow's apparent radius
      // would grow every time the watchdog dropped the render scale.
      const bloomRadius = 10 / Math.min(refWidth, refHeight);
      const mirrorSign = mirrored ? -1 : 1;
      bloomOffsetH = [bloomRadius * fitX * mirrorSign, 0];
      bloomOffsetV = [0, bloomRadius * fitY];
    }

    const turbRaw = pnoise(now * 0.035 + 9.17);
    // floor raised (0.45 -> 0.85) so the calm end of the cycle never
    // reads as "almost no distortion" the way it used to
    const turbulence = 0.85 + smoothstepJS(-0.25, 0.55, turbRaw) * 1.15;
    const angleA = now * 0.11 + pnoise(now * 0.02 + 0.5) * 3.14159;
    const angleB = angleA * 1.37 + 2.4 + pnoise(now * 0.017 + 4.0) * 2.0;

    // angleA/angleB are the SAME for every pixel this frame, so the
    // trig the shader used to redo per-pixel (cos/sin of each, plus
    // cos/sin of angleA*0.4 for the ripple-noise rotation) is done
    // once here instead — see uWaveDirA/B/uWarpRot in fragment.glsl
    const waveDirA = [Math.cos(angleA), Math.sin(angleA)];
    const waveDirB = [Math.cos(angleB), Math.sin(angleB)];
    const rotAngle = angleA * 0.4;
    const rotC = Math.cos(rotAngle), rotS = Math.sin(rotAngle);
    // same (c,-s,s,c) column layout the shader's old rot2() built
    const warpRot = [rotC, -rotS, rotS, rotC];

    // the whole line-doubling base offset (direction + amount +
    // breathing) has no uv in it anywhere, but DOES depend on time
    // (breathing), so unlike fitX/bloomOffsetH/V above this can't be
    // cached — it's still computed once per frame instead of once per
    // pixel, which is the part that actually mattered. Divided by the
    // FIXED reference resolution (not canvas.width/height) so its
    // apparent on-screen width doesn't grow as the watchdog drops the
    // live render scale — see refWidth/refHeight above.
    const ghostAngleRad = (params.angle * Math.PI) / 180;
    const ghostBreathing = 1.0 + params.drift * Math.sin(now * params.ghostSpeed * 0.6);
    const ghostBaseOffset = [
      Math.cos(ghostAngleRad) * params.double * ghostBreathing / refWidth,
      Math.sin(ghostAngleRad) * params.double * ghostBreathing / refHeight,
    ];

    // ---- the only shader draw this frame: straight to the screen,
    // single pass, single render target — the line-doubling/ghost
    // overlay is just two extra same-frame samples of the live camera,
    // computed inside this same draw (see shaders/fragment.glsl).
    // Framebuffer binding, viewport, and the two texture units are all
    // ALREADY correct at this point — framebuffer is set once at setup
    // below (never changes, there's only ever the default framebuffer),
    // viewport is set by resize() whenever canvas size actually
    // changes, and units 0/1 are only ever touched by the upload step
    // above (which explicitly re-binds them when it uploads) — so none
    // of that needs re-issuing to the GL driver on every draw call. ----
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
    gl.uniform1f(U.uTime, now);
    gl.uniform1f(U.uMirror, mirrored ? 1.0 : 0.0);
    gl.uniform1f(U.uDistortion, params.distortion);
    gl.uniform1f(U.uWaveSpeed, params.waveSpeed);
    gl.uniform1f(U.uRippleScale, params.rippleScale);
    gl.uniform1f(U.uContrast, params.contrast);
    gl.uniform1f(U.uBrightness, params.brightness);
    gl.uniform1f(U.uSaturation, params.saturation);
    gl.uniform1f(U.uTurbulence, turbulence);
    gl.uniform2f(U.uWaveDirA, waveDirA[0], waveDirA[1]);
    gl.uniform2f(U.uWaveDirB, waveDirB[0], waveDirB[1]);
    gl.uniform4f(U.uWarpRot, warpRot[0], warpRot[1], warpRot[2], warpRot[3]);
    gl.uniform2f(U.uGhostBaseOffset, ghostBaseOffset[0], ghostBaseOffset[1]);
    gl.uniform2f(U.uRefResolution, refWidth, refHeight);
    gl.uniform1f(U.uGhostOpacity, effectiveGhostOpacity);
    gl.uniform1f(U.uGhostWaveAmount, params.ghostWaveAmount);
    gl.uniform1f(U.uGhostWaveScale, params.ghostWaveScale);
    gl.uniform1f(U.uGhostWaveSpeed, params.ghostWaveSpeed);
    gl.uniform1f(U.uGlow, effectiveGlow);
    gl.uniform2f(U.uBloomOffsetH, bloomOffsetH[0], bloomOffsetH[1]);
    gl.uniform2f(U.uBloomOffsetV, bloomOffsetV[0], bloomOffsetV[1]);
    gl.uniform2f(U.uTapPos, tapPos.x, tapPos.y);
    gl.uniform1f(U.uTapTime, tapAtMs === -Infinity ? -1.0 : (performance.now() - tapAtMs) / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  requestAnimationFrame(frame);
}

// ---- UI wiring ---- (effect controls removed — tune via DEFAULTS in
// code from now on; only the camera-flip button remains)
function wireControls() {
  document.getElementById("flipBtn").addEventListener("click", async () => {
    currentFacing = currentFacing === "environment" ? "user" : "environment";
    try {
      await startCamera(currentFacing);
    } catch (e) {
      console.error(e);
    }
  });
}

// ---- keep the screen from sleeping while the piece is running (the
// Pi kiosk especially — its own display power-save/DPMS would
// otherwise blank the screen mid-artwork). Supported in Chromium;
// best-effort elsewhere. Wake locks are released automatically if the
// tab is ever hidden, so it's re-requested on visibilitychange too. ----
let wakeLock = null;
async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch (e) {
    console.error("wake lock request failed:", e);
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") requestWakeLock();
});

// ---- boot sequence gated on a user tap (camera permission + autoplay) ----
const startOverlay = document.getElementById("startOverlay");
document.getElementById("startBtn").addEventListener("click", async () => {
  startOverlay.classList.add("hidden");
  requestWakeLock();
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
    showFatalError("Camera access failed: " + (e.stack || e.message));
    alert("Camera access failed: " + e.message);
    startOverlay.classList.remove("hidden");
    return;
  }
  wireControls();
  init().catch((e) => {
    console.error(e);
    showFatalError("Failed to start renderer: " + (e.stack || e.message));
    alert("Failed to start renderer: " + e.message);
  });
});
