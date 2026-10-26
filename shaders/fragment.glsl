#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uCamera;
uniform vec2 uResolution;
uniform vec2 uCameraAspectFit; // scale to apply for cover-fit
uniform float uTime;
uniform float uMirror;

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
uniform float uEdgeSensitivity;
uniform float uGlistenIntensity;
uniform float uGlintThreshold;
uniform float uContrast;
uniform float uBrightness;
uniform float uVignette;
uniform float uGrain;
uniform float uMurk;
uniform float uSaturation;

// ghost double-exposure overlay, structurally ported from the reference
// AR project's line-doubling study: two extra same-frame samples of the
// live camera offset in opposite directions, blended with weights that
// always sum to 1 (base*(1-2*op) + (ghost+opposite)*op). There is no
// history buffer here — it cannot accumulate or "burn in" over time,
// because every frame is computed from scratch off the live image.
uniform float uGhostOffset;
uniform float uOverlayOpacity;

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

// fit camera texture to screen like CSS background-size: cover
vec2 coverUv(vec2 uv){
  return (uv - 0.5) * uCameraAspectFit + 0.5;
}

// underwater domain-warp displacement: two wave trains whose travel
// direction slowly rotates (uAngleA/uAngleB) and cross-interfere, plus a
// rotating ripple-noise layer, modulated by a slow turbulence factor so
// the whole thing drifts between calm and choppy instead of looping
vec2 waveOffset(vec2 uv, float t){
  vec2 dirA = vec2(cos(uAngleA), sin(uAngleA));
  vec2 dirB = vec2(cos(uAngleB), sin(uAngleB));

  float swellA = sin(dot(uv, dirA) * 7.0 + t * 0.9);
  float swellB = sin(dot(uv, dirB) * 5.3 - t * 0.65 + 1.7);
  vec2 swell = (dirA * swellA + dirB * swellB) * 0.012 * uTurbulence;

  mat2 r = rot2(uAngleA * 0.4);
  vec2 ruv = r * uv * uRippleScale;
  vec2 n = vec2(
    fbm(ruv + vec2(0.0, t * 0.18)),
    fbm(ruv + vec2(37.2, -t * 0.15))
  );

  return swell * uDistortion + n * 0.035 * uDistortion * uTurbulence;
}

// full camera-space warped + mirrored uv (pre video-flip), evaluated
// once per pixel to seed a local linear (Jacobian) approximation so we
// don't have to re-run the noise for every Sobel tap
vec2 warpedCameraUv(vec2 canvasUv, float t){
  vec2 base = coverUv(canvasUv);
  vec2 duv = base + waveOffset(base, t);
  if (uMirror > 0.5) duv.x = 1.0 - duv.x;
  return duv;
}

float luminance(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 sampleCameraAt(vec2 cameraUv){
  vec2 c = clamp(cameraUv, 0.0015, 0.9985);
  return texture(uCamera, vec2(c.x, 1.0 - c.y)).rgb;
}

// small local wave field at several wavelengths/rates, ported from the
// AR reference project — moves the ghost samples only, cheap (plain
// sines, no noise calls), never static
vec2 ghostLocalWaves(vec2 uv, float t){
  vec2 size = max(uResolution, vec2(1.0));
  vec2 q = (uv - 0.5) * size / min(size.x, size.y);
  vec2 p = q * 1.6;
  float tt = t * 2.2;
  float broad = sin(dot(p, vec2(5.0, 8.0)) - tt * 0.43);
  float middle = sin(dot(p, vec2(-14.0, 9.0)) + tt * 0.79 + 1.3);
  float fine = sin(dot(p, vec2(26.0, 19.0)) - tt * 1.17 + 3.1);
  float crossing = sin(dot(p, vec2(11.0, -21.0)) + tt * 0.58 + 0.7);
  float left = 0.5 + 0.5 * sin(p.x * 3.2 + p.y * 1.1);
  float upper = 0.5 + 0.5 * cos(p.y * 4.0 - p.x * 0.8);
  vec2 field = vec2(
    broad * (0.35 + 0.20 * left) + fine * (0.12 + 0.10 * upper),
    middle * (0.22 + 0.18 * upper) + crossing * (0.10 + 0.10 * (1.0 - left))
  );
  return field * 3.0 / size;
}

// same tonal grade used for the live base sample (contrast/brightness,
// then saturation reconstruction), reused for the ghost samples so they
// read as part of the same image rather than raw unprocessed color
vec3 gradeSample(vec3 raw){
  float l = luminance(raw);
  float g = (l - 0.5) * uContrast + 0.5 + uBrightness;
  vec3 toned = raw + (g - l);
  vec3 outG = mix(vec3(g), clamp(toned, 0.0, 1.0), clamp(uSaturation, 0.0, 1.0));
  return clamp(outG, 0.0, 1.0);
}

void main(){
  vec2 uv = vUv;
  vec2 texel = 1.0 / uResolution;
  float t = uTime * uWaveSpeed;

  // one real warp evaluation per pixel, plus two tiny finite-difference
  // probes to build a local Jacobian; every Sobel tap below then just
  // does a cheap matrix-multiply + texture fetch instead of re-running
  // the noise functions (this is the main render-cost fix)
  vec2 fdStep = texel * 3.0;
  vec2 duv0 = warpedCameraUv(uv, t);
  vec2 duvX = warpedCameraUv(uv + vec2(fdStep.x, 0.0), t);
  vec2 duvY = warpedCameraUv(uv + vec2(0.0, fdStep.y), t);
  vec2 Jx = (duvX - duv0) / fdStep.x;
  vec2 Jy = (duvY - duv0) / fdStep.y;

  vec3 base = sampleCameraAt(duv0);
  float lum = luminance(base);

  // ---- tonal grade (B&W) ----
  float bw = (lum - 0.5) * uContrast + 0.5 + uBrightness;

  // ---- sobel edge detection via the Jacobian-approximated warp ----
  float px = texel.x * 1.6;
  float py = texel.y * 1.6;

  float tl = luminance(sampleCameraAt(duv0 + Jx*(-px) + Jy*( py)));
  float tt = luminance(sampleCameraAt(duv0 + Jy*( py)));
  float tr = luminance(sampleCameraAt(duv0 + Jx*( px) + Jy*( py)));
  float l  = luminance(sampleCameraAt(duv0 + Jx*(-px)));
  float r  = luminance(sampleCameraAt(duv0 + Jx*( px)));
  float bl = luminance(sampleCameraAt(duv0 + Jx*(-px) + Jy*(-py)));
  float bb = luminance(sampleCameraAt(duv0 + Jy*(-py)));
  float br = luminance(sampleCameraAt(duv0 + Jx*( px) + Jy*(-py)));

  float gx = (tr + 2.0*r + br) - (tl + 2.0*l + bl);
  float gy = (bl + 2.0*bb + br) - (tl + 2.0*tt + tr);
  float edge = sqrt(gx*gx + gy*gy);
  float edgeMask = smoothstep(uEdgeSensitivity, uEdgeSensitivity + 0.25, edge);
  float edgeGlow = smoothstep(uEdgeSensitivity * 0.5, uEdgeSensitivity + 0.6, edge);

  // ---- caustic focusing: real underwater caustics are bright exactly
  // where the water surface is locally converging/diverging light, i.e.
  // where the refraction mapping compresses or stretches area. We
  // already have that mapping's Jacobian (Jx, Jy) from the warp
  // approximation above, so we reuse it directly instead of an
  // unrelated noise pattern — this is what ties the glisten to the
  // actual ripples instead of floating independently over them ----
  float warpDet = Jx.x * Jy.y - Jy.x * Jx.y;
  float causticField = clamp(abs(warpDet - 1.0) * 5.0, 0.0, 1.0);

  // a little extra high-frequency shimmer so the caustic lines don't
  // look perfectly smooth, still derived from the same warp signal
  float shimmer = sin(duv0.x * 60.0 + duv0.y * 40.0 + uTime * 2.4) * 0.5 + 0.5;
  float glintField = mix(causticField, causticField * shimmer, 0.35);
  float glintGate = smoothstep(uGlintThreshold, uGlintThreshold + 0.35, glintField);

  // ---- directional reflection: the virtual light travels with the
  // same dominant direction as the wave field itself (uAngleA), so the
  // specular sweep and the visible ripple motion read as one connected
  // phenomenon instead of two independent animations ----
  vec2 edgeDir = normalize(vec2(gx, gy) + vec2(1e-5));
  vec2 lightDir = vec2(cos(uAngleA), sin(uAngleA));
  float align = dot(edgeDir, lightDir) * 0.5 + 0.5;
  float specular = pow(align, 3.0);
  float specularSoft = pow(align, 1.5);

  // high-frequency sparkle so the highlight reads as scattered flecks of
  // light rather than a uniform strip
  float sparkle = fract(sin(dot(floor(uv * uResolution * 0.4), vec2(41.3, 289.1))
                   + uTime * 2.2) * 54321.0);
  float sparkleMod = mix(0.7, 1.4, sparkle);

  float core = edgeMask * glintGate * mix(0.3, 1.6, specular) * sparkleMod;
  float halo = edgeGlow * glintGate * specularSoft * 0.3;
  float glisten = (core + halo) * uGlistenIntensity;

  float result = bw + glisten;

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

  // ---- b&w saturation: all of the grading/glisten/vignette/grain/murk
  // above was computed as a single luminance delta from the base camera
  // sample; to reintroduce color we apply that exact same delta to every
  // channel (so highlights, grain etc. look identical) instead of
  // recomputing the whole pipeline in color ----
  vec3 toned = base + (result - lum);
  vec3 outColor = mix(vec3(result), clamp(toned, 0.0, 1.0), clamp(uSaturation, 0.0, 1.0));

  // ---- ghost double-exposure overlay (ported structure, see uGhostOffset
  // declaration above): offset direction follows the same dominant wave
  // direction as the water itself (uAngleA) so it reads as part of the
  // same current rather than an unrelated effect ----
  float op = clamp(uOverlayOpacity, 0.0, 0.5);
  if (op > 0.001) {
    vec2 dir = vec2(cos(uAngleA), sin(uAngleA));
    float breathing = 1.0 + (uTurbulence - 1.0) * 0.3;
    vec2 offset = dir * uGhostOffset * breathing / max(uResolution, vec2(1.0));
    offset += ghostLocalWaves(uv, uTime);

    vec3 ghost = gradeSample(sampleCameraAt(warpedCameraUv(uv + offset, t)));
    vec3 opposite = gradeSample(sampleCameraAt(warpedCameraUv(uv - offset, t)));

    outColor = outColor * (1.0 - 2.0 * op) + (ghost + opposite) * op;
  }

  fragColor = vec4(outColor, 1.0);
}
