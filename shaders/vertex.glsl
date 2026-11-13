precision highp float;

// WebGL1/GLSL ES 100 — no layout(location=...) qualifier; the
// attribute location is bound explicitly in JS via
// gl.bindAttribLocation() before the program is linked (see main.js)
attribute vec2 aPos;
varying vec2 vUv;

void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
