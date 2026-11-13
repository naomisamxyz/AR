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
// and cross-interferes instead of looking like one repeating ripple
uniform float uTurbulence;
uniform float uAngleA;
uniform float uAngleB;

uniform float uDistortion;
uniform float uWaveSpeed;
uniform float uRippleScale;
uniform float uContrast;
uniform float uBrightness;
uniform float uVignette;
uniform float uGrain;
uniform float uMurk;
uniform float uSaturation;

// line-doubling + glow, ported from AR1 (filter.frag): two extra
// same-frame samples of the live camera offset in opposite directions
// (base*(1-2*op) + (ghost+opposite)*op — no history buffer, so it can't
// accumulate/"burn in"), plus an 8-direction bloom ring kept only where
// bright
uniform float uDouble;
uniform float uGhostOpacity;
uniform float uAngle;
uniform float uDrift;
uniform float uGhostSpeed;
uniform float uGhostWaveAmount;
uniform float uGhostWaveScale;
uniform float uGhostWaveSpeed;
uniform float uGlow;

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

// kept to 3 octaves — this now only runs ONCE per pixel (see below),
// so we can afford real simplex noise without it costing the frame rate
float fbm(vec2 p){
  float v = 0.0;
  float amp = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 3; i++) {
    v += amp * snoise(p);
    p = rot * p * 2.02;
    amp *= 0.55;
  }
  return v;
}

mat2 rot2(float a){
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
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
  if (uTapTime < 0.0 || tapT < 0.0) return 0.0;

  vec2 size = max(uResolution, vec2(1.0));
  vec2 d = (canvasUv - uTapPos) * size;
  float dist = length(d) / min(size.x, size.y);

  float speed = 0.62;
  float radius = 0.035 + tapT * speed;
  float ringWidth = 0.1;
  float ring = exp(-pow((dist - radius) / ringWidth, 2.0));

  float fade = exp(-tapT * 0.45);
  float tailFade = 1.0 - smoothstep(7.0, 8.0, tapT);

  return ring * fade * tailFade * 1.3;
}

// underwater domain-warp displacement: two wave trains whose travel
// direction slowly rotates (uAngleA/uAngleB) and cross-interfere, plus a
// rotating ripple-noise layer, modulated by a slow turbulence factor so
// the whole thing drifts between calm and choppy instead of looping.
// `turb` is passed in rather than reading uTurbulence directly so a
// tap's local turbulence boost (see tapTurbBoost above) can perturb
// this same field right at the tap point.
vec2 waveOffset(vec2 uv, float t, float turb){
  vec2 dirA = vec2(cos(uAngleA), sin(uAngleA));
  vec2 dirB = vec2(cos(uAngleB), sin(uAngleB));

  float swellA = sin(dot(uv, dirA) * 7.0 + t * 0.9);
  float swellB = sin(dot(uv, dirB) * 5.3 - t * 0.65 + 1.7);
  vec2 swell = (dirA * swellA + dirB * swellB) * 0.012 * turb;

  mat2 r = rot2(uAngleA * 0.4);
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
  float fine = sin(dot(p, vec2(26.0, 19.0)) - t * 1.17 + 3.1);
  float crossing = sin(dot(p, vec2(11.0, -21.0)) + t * 0.58 + 0.7);
  float left = 0.5 + 0.5 * sin(p.x * 3.2 + p.y * 1.1);
  float upper = 0.5 + 0.5 * cos(p.y * 4.0 - p.x * 0.8);
  vec2 field = vec2(
    broad * (0.35 + 0.20 * left) + fine * (0.12 + 0.10 * upper),
    middle * (0.22 + 0.18 * upper) + crossing * (0.10 + 0.10 * (1.0 - left))
  );
  return field * uGhostWaveAmount / size;
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
// Cut from 5 taps to 3 (center + one diagonal pair) as part of halving
// the glow pipeline's overall cost — still enough to filter out
// single-pixel specks, just less perfectly isotropic about it.
float areaLuminance(vec2 cameraUv){
  vec2 px = 1.5 / uResolution;
  float l0 = luminance(sampleCameraAt(cameraUv));
  float l1 = luminance(sampleCameraAt(cameraUv + vec2(px.x, px.y)));
  float l2 = luminance(sampleCameraAt(cameraUv - vec2(px.x, px.y)));
  return (l0 + l1 + l2) / 3.0;
}

// glow: a ring of 8 samples around this pixel, kept only where bright,
// added back as a soft bloom around highlights — ported from AR1's
// bloomTap(), shifted by the same single warp displacement as the rest
// of the image so it moves with the water without each tap aliasing
// into its own separate circular artifact. Thresholded on the AREA
// average above (not a single-pixel sample, back to AR1's original
// 0.55/0.95 range) so small isolated bright dots don't each spawn their
// own 8-dot ring — only patches of brightness large enough to survive
// the neighbor average actually bloom.
vec3 bloomTap(vec2 uv, vec2 dir, float radius, vec2 warpDisp){
  vec2 cuv = plainCameraUv(uv + dir * radius) + warpDisp;
  float l = areaLuminance(cuv);
  return vec3(l) * smoothstep(0.55, 0.95, l);
}

void main(){
  vec2 uv = vUv;
  float t = uTime * uWaveSpeed;

  vec2 duv0 = warpedCameraUv(uv, t);
  vec3 rawBase = sampleCameraAt(duv0);
  // single shared warp displacement, reused by every glow bloom tap
  // below instead of each one re-evaluating the noisy warp on its own
  vec2 warpDisp = duv0 - plainCameraUv(uv);

  // ---- line-doubling (ghost double-exposure), ported from AR1:
  // blended on the RAW camera samples, before grading, same as AR1's
  // own pipeline order (and unlike the previous ghost effect this
  // replaces, which graded each ghost sample individually) ----
  vec3 base = rawBase;
  float op = clamp(uGhostOpacity, 0.0, 0.5);
  if (op > 0.001) {
    float angle = radians(uAngle);
    vec2 dir = vec2(cos(angle), sin(angle));
    float breathing = 1.0 + uDrift * sin(uTime * uGhostSpeed * 0.6);
    vec2 offset = dir * uDouble * breathing / max(uResolution, vec2(1.0));
    offset += ghostWaves(uv);

    vec3 ghost = sampleCameraAt(warpedCameraUv(uv + offset, t));
    vec3 opposite = sampleCameraAt(warpedCameraUv(uv - offset, t));

    base = rawBase * (1.0 - 2.0 * op) + (ghost + opposite) * op;
  }

  float lum = luminance(base);

  // ---- tonal grade (B&W) ----
  float bw = (lum - 0.5) * uContrast + 0.5 + uBrightness;
  float result = bw;

  // ---- underwater murk / depth falloff — cheap sine fog, no noise ----
  float d = distance(uv, vec2(0.5, 0.45));
  float fog = sin(uv.x*2.0 + uTime*0.07) * cos(uv.y*2.3 - uTime*0.05) * 0.5 + 0.5;
  result = mix(result, result * 0.75 + fog * 0.12, uMurk);

  // ---- vignette ----
  float vig = smoothstep(0.95, 0.25, d * (1.0 + uVignette));
  result *= mix(1.0, vig, clamp(uVignette, 0.0, 1.0));

  // ---- film grain ----
  float grainN = fract(sin(dot(uv * uResolution + uTime * 60.0, vec2(12.9898, 78.233))) * 43758.5453);
  result += (grainN - 0.5) * uGrain;

  result = clamp(result, 0.0, 1.0);

  // ---- b&w saturation: all of the grading/vignette/grain/murk above
  // was computed as a single luminance delta from the (possibly
  // line-doubled) base sample; to reintroduce color we apply that
  // exact same delta to every channel (so highlights, grain etc. look
  // identical) instead of recomputing the whole pipeline in color ----
  vec3 toned = base + (result - lum);
  vec3 outColor = mix(vec3(result), clamp(toned, 0.0, 1.0), clamp(uSaturation, 0.0, 1.0));

  // ---- glow (bloom), ported from AR1 ----
  // ring cut from 8 taps to 6 (60° apart instead of 45°), combined with
  // areaLuminance's 5->3 cut above: roughly 40 texture reads per pixel
  // down to 18 — about half the original cost, in exchange for a
  // slightly less perfectly round/isotropic bloom shape
  if (uGlow > 0.001) {
    float radius = 10.0 / min(uResolution.x, uResolution.y);
    vec3 bloom = bloomTap(uv, vec2(1.0, 0.0), radius, warpDisp)
        + bloomTap(uv, vec2(0.5, 0.8660254), radius, warpDisp)
        + bloomTap(uv, vec2(-0.5, 0.8660254), radius, warpDisp)
        + bloomTap(uv, vec2(-1.0, 0.0), radius, warpDisp)
        + bloomTap(uv, vec2(-0.5, -0.8660254), radius, warpDisp)
        + bloomTap(uv, vec2(0.5, -0.8660254), radius, warpDisp);
    outColor += (bloom / 6.0) * uGlow;
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
  if (ovUv.x >= 0.0 && ovUv.x <= 1.0 && ovUv.y >= 0.0 && ovUv.y <= 1.0) {
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
