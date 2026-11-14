precision highp float;

// WebGL1/GLSL ES 100 — varying instead of in/out, gl_FragColor instead
// of a declared out variable, texture2D(...) instead of texture(...)
varying vec2 vUv;

uniform sampler2D uCamera;
uniform vec2 uResolution;
uniform vec2 uCameraAspectFit; // scale to apply for cover-fit
uniform float uTime;
uniform float uMirror;

// overlay layer: a glow/light-effect video rendered on BLACK and
// composited ADDITIVELY (black contributes nothing when added, so the
// glow just adds its own light on top of the camera image) — not
// chroma-keyed. Chroma key was tried first, but a glow's edge is a
// smooth gradient blending into the key color, which showed up as a
// translucent green-tinted halo no threshold tuning could fully clear;
// additive-on-black has no key color at all, so that problem doesn't
// exist.
uniform sampler2D uOverlay;
uniform vec2 uOverlayAspectFit; // cover-fit scale, same idea as uCameraAspectFit
uniform float uOverlayOpacity; // combines the user's own opacity slider with the 1s fade-in (see main.js)
// where the object sits on screen: a static authored placement PLUS,
// on phones, a gyroscope-driven pan computed in JS (re-anchored to the
// current phone orientation every time the overlay video loops) so the
// object appears to stay put "in the room" as you look away and back —
// disabled (left at just the static placement) on the stationary Pi
// kiosk, which has no orientation sensor to pan with anyway
uniform vec2 uOverlayOffset;
// shrinks/grows the object on screen around its own center (1.0 = its
// natural cover-fit size, 0.5 = half that size, etc.) — applied after
// the offset, so moving it around and resizing it stay independent
uniform float uOverlayScale;


// slow, organic values computed once per frame on the CPU (cheap sine
// noise, see main.js) instead of per-pixel on the GPU — these drive the
// wave travel direction and overall choppiness so the pattern wanders
// and cross-interferes instead of looking like one repeating ripple.
// uWaveDirA/uWaveDirB/uWarpRot are the unit direction vectors and
// rotation matrix that waveOffset() used to recompute every pixel via
// cos()/sin() of uAngleA/uAngleB — those angles are the SAME for every
// pixel in a frame, so main.js now does the trig once per frame on the
// CPU and uploads the result directly, instead of every fragment
// redoing identical cos/sin work.
uniform float uTurbulence;
uniform vec2 uWaveDirA; // (cos(angleA), sin(angleA))
uniform vec2 uWaveDirB; // (cos(angleB), sin(angleB))
uniform vec4 uWarpRot;  // rot2(angleA*0.4) as (c,-s,s,c), same column layout rot2() used

uniform float uDistortion;
uniform float uWaveSpeed;
uniform float uRippleScale;
uniform float uContrast;
uniform float uBrightness;
uniform float uSaturation;

// line-doubling + glow, ported from AR1 (filter.frag): two extra
// same-frame samples of the live camera offset in opposite directions
// (base*(1-2*op) + (ghost+opposite)*op — no history buffer, so it can't
// accumulate/"burn in"), plus a bloom ring kept only where bright.
// uGhostBaseOffset replaces uDouble/uAngle/uDrift/uGhostSpeed: the
// offset direction, breathing, and division by resolution were all
// frame-invariant (no uv in them at all), so main.js now computes the
// whole base offset vector once per frame and uploads it directly —
// only ghostWaves() below still needs its own per-pixel uv-dependent
// uniforms, since it genuinely varies across the screen.
uniform vec2 uGhostBaseOffset;
uniform float uGhostOpacity;
uniform float uGhostWaveAmount;
uniform float uGhostWaveScale;
uniform float uGhostWaveSpeed;
uniform float uGlow;
// a FIXED reference render-target size (main.js's RENDER_SCALE_STEPS[0]
// baseline, i.e. the un-degraded 0.45 scale) — used only to convert
// ghostWaves()'s "pixel-sized" wave amount into a UV fraction. Using
// the ACTUAL current uResolution for that divide (as this used to)
// meant the apparent on-screen size of that jitter would grow every
// time the performance watchdog dropped the render scale, since the
// same uGhostWaveAmount divided by a SMALLER live resolution produces
// a LARGER UV fraction — and the canvas is always stretched via CSS to
// the same physical screen size regardless of its internal resolution,
// so a bigger UV fraction directly means a bigger apparent size. uBloomOffsetH/V
// and uGhostBaseOffset avoid this same trap by being computed in JS
// against this same fixed reference instead of the live canvas size.
uniform vec2 uRefResolution;
// precomputed camera-space bloom tap offsets (see uBloomOffsetH/V in
// main.js) — recomputed in JS whenever viewport size, camera fit, or
// mirroring changes, instead of every bloom tap redoing the same
// cover-fit + mirror transform on the GPU for a result that's
// identical across the whole frame
uniform vec2 uBloomOffsetH;
uniform vec2 uBloomOffsetV;

// tap/click interactivity, ported from AR2-dev: canvas-space tap
// position and seconds since the tap (negative = no active tap) —
// adapted to boost this project's own turbulence/distortion locally
// near the tap point (see tapTurbBoost() below), since this project has
// no "seam network" to drive a visible ring line like AR2-dev's does
uniform vec2 uTapPos;
uniform float uTapTime;

// ---------- hash / simplex noise (Ashima-style) ----------
vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec2 mod289(vec2 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec3 permute(vec3 x){ return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                      -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
          + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m; m = m*m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  vec3 g;
  g.x  = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// was 3 octaves, then 2; cut further to 1 (a single snoise call, no
// loop/rotation needed) for the Pi — this runs on EVERY pixel
// unconditionally (the base warp has no on/off switch), so it's the
// single biggest fixed per-pixel cost in the whole shader. To restore
// 2 octaves, change the loop bound back to "i < 2". Loses a layer of
// cross-interference detail — the warp looks a bit more like one
// simple noise field and less "choppy"/complex.
float fbm(vec2 p){
  float v = 0.0;
  float amp = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 1; i++) {
    v += amp * snoise(p);
    p = rot * p * 2.02;
    amp *= 0.55;
  }
  return v;
}


// fit a texture to screen like CSS background-size: cover, given an
// arbitrary aspect-fit scale (so the same math works for both the
// camera and the overlay, each with their own fit)
vec2 coverUvFit(vec2 uv, vec2 fit){
  return (uv - 0.5) * fit + 0.5;
}

vec2 coverUv(vec2 uv){
  return coverUvFit(uv, uCameraAspectFit);
}

// a LOCAL, expanding boost added on top of the normal uTurbulence value
// — not a separate visual effect, just more of the exact same thing,
// confined to a single thin ring that travels outward from the tap
// point over time and fades out. Ported directly from AR2-dev's
// tapTurbBoost(); its tapRingLine() (the explicit visible seam-network
// stroke) is NOT ported, since this project has no seam network for it
// to draw into.
float tapTurbBoost(vec2 canvasUv, float tapT){
  // tailFade below already reaches exactly 0 at tapT>=8 (smoothstep(7,8,...)),
  // so once a tap is older than that it can never contribute anything —
  // bail out before doing any of the distance/ring/fade math at all
  if (uTapTime < 0.0 || tapT < 0.0 || tapT > 8.0) return 0.0;

  vec2 size = max(uResolution, vec2(1.0));
  vec2 d = (canvasUv - uTapPos) * size;
  float dist = length(d) / min(size.x, size.y);

  float speed = 0.62;
  float radius = 0.035 + tapT * speed;
  float ringWidth = 0.1;
  float ring = exp(-pow((dist - radius) / ringWidth, 2.0));

  float fade = exp(-tapT * 0.45);
  float tailFade = 1.0 - smoothstep(7.0, 8.0, tapT);

  return ring * fade * tailFade * 2.6;
}

// underwater domain-warp displacement: two wave trains whose travel
// direction slowly rotates (uWaveDirA/uWaveDirB, precomputed per frame
// in main.js) and cross-interfere, plus a rotating ripple-noise layer,
// modulated by a slow turbulence factor so the whole thing drifts
// between calm and choppy instead of looping.
// `turb` is passed in rather than reading uTurbulence directly so a
// tap's local turbulence boost (see tapTurbBoost above) can perturb
// this same field right at the tap point.
vec2 waveOffset(vec2 uv, float t, float turb){
  float swellA = sin(dot(uv, uWaveDirA) * 7.0 + t * 0.9);
  float swellB = sin(dot(uv, uWaveDirB) * 5.3 - t * 0.65 + 1.7);
  vec2 swell = (uWaveDirA * swellA + uWaveDirB * swellB) * 0.012 * turb;

  mat2 r = mat2(uWarpRot.x, uWarpRot.y, uWarpRot.z, uWarpRot.w);
  vec2 ruv = r * uv * uRippleScale;
  vec2 n = vec2(
    fbm(ruv + vec2(0.0, t * 0.18)),
    fbm(ruv + vec2(37.2, -t * 0.15))
  );

  return swell * uDistortion + n * 0.035 * uDistortion * turb;
}

// full camera-space warped + mirrored uv (pre video-flip), evaluated
// once per pixel to seed a local linear (Jacobian) approximation so we
// don't have to re-run the noise for every Sobel tap
vec2 warpedCameraUv(vec2 canvasUv, float t){
  vec2 base = coverUv(canvasUv);
  float turb = uTurbulence + tapTurbBoost(canvasUv, uTapTime);
  vec2 duv = base + waveOffset(base, t, turb);
  if (uMirror > 0.5) duv.x = 1.0 - duv.x;
  return duv;
}

float luminance(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 sampleCameraAt(vec2 cameraUv){
  vec2 c = clamp(cameraUv, 0.0015, 0.9985);
  return texture2D(uCamera, vec2(c.x, 1.0 - c.y)).rgb;
}

// small local wave field at several wavelengths/rates, ported directly
// from AR1's localWaves() — moves the line-doubling ghost samples only,
// cheap (plain sines, no noise calls), never static
vec2 ghostWaves(vec2 uv){
  vec2 size = max(uResolution, vec2(1.0));
  vec2 q = (uv - 0.5) * size / min(size.x, size.y);
  float t = uTime * uGhostWaveSpeed;
  vec2 p = q * uGhostWaveScale;
  float broad = sin(dot(p, vec2(5.0, 8.0)) - t * 0.43);
  float middle = sin(dot(p, vec2(-14.0, 9.0)) + t * 0.79 + 1.3);
  float envelope = 0.5 + 0.5 * sin(p.x * 3.2 + p.y * 1.1);
  vec2 field = vec2(
    broad * (0.35 + 0.20 * envelope),
    middle * (0.22 + 0.18 * envelope)
  );
  // divides by the FIXED reference resolution, not the live `size`
  // above (which is only used to aspect-correct the wave pattern's
  // phase/orientation, not its final magnitude) — see uRefResolution
  return field * uGhostWaveAmount / max(uRefResolution, vec2(1.0));
}

// cover-fit + mirrored, but WITHOUT the wave warp — used below to apply
// one single shared warp displacement to the whole glow ring (see
// warpDisp in main()), rather than letting each tap re-evaluate the
// noisy warp independently at its own nearby position: that decorrelated
// the 8 ring samples from each other (each landing on a slightly
// different noise value) and showed up as small circular blob artifacts
// instead of one coherent bloom patch.
vec2 plainCameraUv(vec2 canvasUv){
  vec2 duv = coverUv(canvasUv);
  if (uMirror > 0.5) duv.x = 1.0 - duv.x;
  return duv;
}

// small average around a point — used below to tell a real glowing
// AREA apart from a single bright pixel (grain speck, tiny specular
// fleck): a lone bright pixel gets dragged back down by its dim
// neighbors in this average, while a genuinely larger bright patch
// stays bright, since its neighbors are bright too.
// Was 5 taps, then 3 (center + one diagonal pair); cut further to a
// single center tap for the Pi — to restore the 3-tap average, bring
// back the l1/l2 lines (each was commented out just below) and the
// /3.0 divide. Loses the "ignore single-pixel specks" filtering this
// was for, so isolated bright dots may bloom individually again.
float areaLuminance(vec2 cameraUv){
  // vec2 px = 1.5 / uResolution;
  float l0 = luminance(sampleCameraAt(cameraUv));
  // float l1 = luminance(sampleCameraAt(cameraUv + vec2(px.x, px.y)));
  // float l2 = luminance(sampleCameraAt(cameraUv - vec2(px.x, px.y)));
  return l0; // was (l0 + l1 + l2) / 3.0;
}

// glow: a ring of samples around this pixel, kept only where bright,
// added back as a soft bloom around highlights — ported from AR1's
// bloomTap(). `centerUv` is the pixel's own already-warped camera
// coordinate (baseCamUv + warpDisp, computed once in main() and passed
// in), and `offset` is one of the precomputed uBloomOffsetH/V
// uniforms — this used to re-derive the full cover-fit+mirror
// transform for every one of the taps via plainCameraUv(uv+dir*radius);
// since that transform is linear, its effect on a fixed offset is the
// SAME for every pixel in the frame, so main.js now computes it once
// per frame on the CPU and this just adds it directly.
vec3 bloomTapOffset(vec2 centerUv, vec2 offset){
  float l = areaLuminance(centerUv + offset);
  return vec3(l) * smoothstep(0.55, 0.95, l);
}

void main(){
  vec2 uv = vUv;
  float t = uTime * uWaveSpeed;

  // computed once, reused for warpDisp below AND for every glow bloom
  // tap further down (see uBloomOffsetH/V) — no need for each to
  // re-derive the same cover-fit+mirror transform independently
  vec2 baseCamUv = plainCameraUv(uv);
  vec2 duv0 = warpedCameraUv(uv, t);
  // single shared warp displacement, reused by every glow bloom tap
  // below instead of each one re-evaluating the noisy warp on its own
  vec2 warpDisp = duv0 - baseCamUv;

  // ---- line-doubling (ghost double-exposure), ported from AR1:
  // blended on the RAW camera samples, before grading, same as AR1's
  // own pipeline order (and unlike the previous ghost effect this
  // replaces, which graded each ghost sample individually) ----
  float op = clamp(uGhostOpacity, 0.0, 0.5);
  // at op==0.5 (the shipped default) this weight is exactly 0, meaning
  // rawBase never actually contributes to the final "base" color below
  // — so skip sampling it at all in that case rather than paying for a
  // texture fetch whose result gets multiplied away
  float rawBaseWeight = 1.0 - 2.0 * op;
  vec3 rawBase = (abs(rawBaseWeight) > 0.001) ? sampleCameraAt(duv0) : vec3(0.0);
  vec3 base = rawBase;
  if (op > 0.001) {
    // uGhostBaseOffset bakes in the direction/angle, the double-amount,
    // AND the breathing oscillation — all of it frame-invariant (no uv
    // anywhere in the formula), so main.js computes it once per frame
    // on the CPU instead of every pixel redoing radians()/cos()/sin()
    // for an identical result
    vec2 offset = uGhostBaseOffset + ghostWaves(uv);

    // plainCameraUv(uv+offset) used to be called here (and again for
    // uv-offset) to re-derive the cover-fit+mirror transform for each
    // ghost sample — but that transform is affine, so its effect on a
    // delta is just the delta scaled by uCameraAspectFit (with the x
    // sign flipped when mirrored), computed once and reused for both
    // the + and - sample instead of two full re-transforms. duv0 is
    // already baseCamUv's warped camera coordinate (duv0 ==
    // baseCamUv + warpDisp), so adding/subtracting this directly from
    // duv0 lands at exactly the same position the old two-call version
    // did, just without redoing the transform.
    vec2 offsetCam = vec2(
      uCameraAspectFit.x * offset.x * (uMirror > 0.5 ? -1.0 : 1.0),
      uCameraAspectFit.y * offset.y
    );
    // reuse the base pixel's ALREADY-COMPUTED warp displacement
    // (baked into duv0) instead of each ghost sample re-running the
    // full noisy warp (waveOffset -> 2x fbm -> 6 snoise calls, plus
    // tapTurbBoost again) completely independently — this was tripling
    // the shader's single heaviest per-pixel cost (one full noise
    // evaluation for the base, then one each for ghost and opposite)
    // for a difference in the water-warp that's too subtle to notice
    // on two samples this close together anyway.
    vec3 ghost = sampleCameraAt(duv0 + offsetCam);
    vec3 opposite = sampleCameraAt(duv0 - offsetCam);

    base = rawBase * rawBaseWeight + (ghost + opposite) * op;
  }

  float lum = luminance(base);

  // ---- tonal grade (B&W) ----
  float result = (lum - 0.5) * uContrast + 0.5 + uBrightness;


  // soft highlight knee: above a threshold, roll values off toward 1.0
  // asymptotically instead of hard-clamping — keeps full contrast (the
  // black end is completely untouched, this only reshapes the top),
  // but stops bright areas from flattening into a single "burnt out"
  // pure white the way a hard clamp does
  float knee = 0.45;
  if (result > knee) {
    result = knee + (1.0 - knee) * (1.0 - exp(-(result - knee) / (1.0 - knee)));
  }
  result = clamp(result, 0.0, 1.0);

  // ---- b&w saturation: the tonal grade above
  // was computed as a single luminance delta from the (possibly
  // line-doubled) base sample; to reintroduce color we apply that
  // exact same delta to every channel (so highlights, grain etc. look
  // identical) instead of recomputing the whole pipeline in color ----
  vec3 toned = base + (result - lum);
  vec3 outColor = mix(vec3(result), clamp(toned, 0.0, 1.0), clamp(uSaturation, 0.0, 1.0));

  // ---- glow (bloom), ported from AR1 ----
  // ring was 8 taps, then 6 (60° apart); cut further to 4 (90° apart,
  // the shipped default) for the Pi, combined with areaLuminance's cut
  // to 1 tap above — roughly 40 texture reads per pixel down to 4.
  // uBloomOffsetH/V are precomputed per frame in main.js (see their
  // declaration above) so each tap is just one vector add instead of
  // redundantly re-deriving the cover-fit+mirror transform on the GPU.
  //
  // GLOW_TWO_TAP is an optional dev-only A/B comparison: two opposing
  // samples instead of four, more directional (less round) but half
  // the texture reads. It's OFF by default (see GLOW_TWO_TAP_TEST in
  // main.js) — four-sample stays the shipped default either way.
  if (uGlow > 0.001) {
    vec2 bloomCenter = baseCamUv + warpDisp;
#ifdef GLOW_TWO_TAP
    vec3 bloom = bloomTapOffset(bloomCenter, uBloomOffsetH)
        + bloomTapOffset(bloomCenter, -uBloomOffsetH);
    outColor += (bloom / 2.0) * uGlow;
#else
    vec3 bloom = bloomTapOffset(bloomCenter, uBloomOffsetH)
        + bloomTapOffset(bloomCenter, -uBloomOffsetH)
        + bloomTapOffset(bloomCenter, uBloomOffsetV)
        + bloomTapOffset(bloomCenter, -uBloomOffsetV);
    outColor += (bloom / 4.0) * uGlow;
#endif
    outColor = clamp(outColor, 0.0, 1.0);
  }

  // ---- overlay video layer, composited last, on top of everything
  // else — rendered on black and added in (not blended/keyed — see the
  // uOverlay comment above), and NOT warped/distorted with the water,
  // since it's meant to sit as a distinct layer above the effect.
  // uOverlayOffset shifts where it sits on screen: a static authored
  // placement plus, on phones, a gyroscope pan computed in JS (see
  // main.js) ----
  // uOverlayOffset is authored as a fraction of the SHORTER screen
  // side, not raw canvas-uv — a plain uv offset would mean the same
  // number represents a different physical distance on a narrow
  // portrait phone vs. the wide landscape Pi (uv.x always spans the
  // full width, uv.y always spans the full height, so the same offset
  // reads as a far bigger shift on whichever axis is shorter). This
  // conversion keeps the authored placement looking the same
  // regardless of screen shape.
  float shorterSide = min(uResolution.x, uResolution.y);
  vec2 overlayOffsetUv = uOverlayOffset * (shorterSide / uResolution);
  vec2 baseOvUv = coverUvFit(uv - overlayOffsetUv, uOverlayAspectFit);

  // shrink/grow around the object's own center (0.5,0.5 in its own uv
  // space) — dividing by a scale < 1 widens the sampled range, so the
  // object reads smaller on screen with more of the (contributes-
  // nothing) black margin visible around it, and vice versa for > 1
  vec2 ovUv = (baseOvUv - 0.5) / max(uOverlayScale, 0.001) + 0.5;

  // uOverlayOpacity checked FIRST so the texture2D sample below is
  // skipped entirely whenever the sculpture is fully invisible anyway
  // — during the wait phase between loops, or at the very start/end of
  // its fade — rather than sampling a texture whose result gets
  // multiplied by zero regardless
  if (uOverlayOpacity > 0.001 && ovUv.x >= 0.0 && ovUv.x <= 1.0 && ovUv.y >= 0.0 && ovUv.y <= 1.0) {
    vec3 ov = texture2D(uOverlay, vec2(ovUv.x, 1.0 - ovUv.y)).rgb;
    // fade the overlay out well before its own frame edge, instead of
    // cutting off hard right at it — the glow in the source footage
    // doesn't quite reach pure black by the edge of its own frame, so
    // without this a faint square outline of the video's own boundary
    // was visible
    vec2 edgeDist = abs(ovUv - 0.5) * 2.0;
    float vignette = 1.0 - smoothstep(0.7, 1.0, max(edgeDist.x, edgeDist.y));
    outColor += ov * vignette * clamp(uOverlayOpacity, 0.0, 1.0);
    outColor = clamp(outColor, 0.0, 1.0);
  }

  gl_FragColor = vec4(outColor, 1.0);
}
