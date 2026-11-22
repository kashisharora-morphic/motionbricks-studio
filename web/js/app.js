import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { MannequinKit } from './mannequin.js';
import { Timeline } from './timeline.js';
import { icon } from './icons.js';
import { ViewCube, viewName } from './viewcube.js';

// ---------------------------------------------------------------- utilities
const $ = (id) => document.getElementById(id);
const f32 = (b64) => { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
const hash = (str) => { // cyrb53
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) { const c = str.charCodeAt(i); h1 = Math.imul(h1 ^ c, 2654435761); h2 = Math.imul(h2 ^ c, 1597334677); }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
};
let toastTimer;
function toast(msg, error = false) {
  const t = $('toast'); t.textContent = msg; t.className = 'toast' + (error ? ' error' : '');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), error ? 5000 : 2200);
}
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ---------------------------------------------------------------- state
const STORE = 'mbstudio.v2';
const S = {
  lib: null, poses: {}, order: [],
  length: 240, frame: 0, playing: false,
  keys: [],              // saved: {id, frame, pose, x, z, yaw, carry}
  draft: null,           // unkeyed pose at the current frame: {pose, x, z, yaw, dirty}
  cache: new Map(),      // segment signature -> generated segment (not saved)
  lastRun: [],           // signatures from the last generation, for the sync tag
  generating: null,      // {done, total}
  nextId: 1,
  keySel: new Set(),   // keyframes selected on the timeline (ids)
  spanSel: null,       // span whose path was clicked (its first keyframe's id): its settings show
  clip: null,          // copied keyframes (with frame offsets)
  props: [],           // shapes from the Add menu: {id, type, p:[x,y,z], q:[x,y,z,w], s:[x,y,z]}
  show: { actor: true, bones: false, jointPaths: false, feet: true, keys: true, spanHandles: true, bds: true, tweens: true, keyPath: true, genPath: true }, // view options (per viewer)
};

function save() {
  try { localStorage.setItem(STORE, JSON.stringify({ length: S.length, keys: S.keys, nextId: S.nextId, props: S.props })); } catch {}
}
function load() {
  try { Object.assign(S.show, JSON.parse(localStorage.getItem(STORE + '.view') || '{}')); } catch {}
  try {
    const d = JSON.parse(localStorage.getItem(STORE) || 'null');
    if (!d) return;
    // Poses that left the library: their keyframes switch to the closest one that's left.
    let moved = 0;
    for (const k of d.keys || []) {
      if (!k.custom && !S.poses[k.pose] && S.poses[RETIRED_POSES[k.pose]]) { k.pose = RETIRED_POSES[k.pose]; moved++; }
    }
    S.length = d.length || 240; S.keys = (d.keys || []).filter((k) => k.custom || S.poses[k.pose]); S.nextId = d.nextId || 1;
    S.props = Array.isArray(d.props) ? d.props.filter((p) => SHAPES[p.type]) : [];
    if (moved) toast(`${moved} keyframe${moved > 1 ? 's' : ''} used a pose that's been removed and now use${moved > 1 ? '' : 's'} the closest one left.`);
  } catch {}
}
const RETIRED_POSES = { walk: 'walk_l', tuck: 'jump', punch: 'stand', kick: 'stand', block: 'stand', cannonball: 'crouch' };

const sortedKeys = () => [...S.keys].sort((a, b) => a.frame - b.frame);
const keyAt = (f) => S.keys.find((k) => k.frame === f);
const posesToUse = (k) => (k.carry === 'stop' ? 1 : k.carry === 'carry' || k.custom ? 4 : S.poses[k.pose].posesToUse);
// A saved pose is either a library pose (k.pose) or one copied from generated motion
// (k.custom: frames stored relative to its own position and facing).
const poseLabelOf = (t) => (t.custom ? t.custom.label || 'In-between' : S.poses[t.pose].label);

// ---------------------------------------------------------------- spans + breakdowns
// The stretch between two keyframes is a span, annotated on its first keyframe (k.span):
//   walk ... (default) straight along the floor
//   jump ... a parabola whose peak height the apex handle sets
// Breakdowns are poses placed inside a span (computed, never saved): walking steps along a
// walk, takeoff / apex / landing along a jump. With breakdowns on, the AI in-betweens run
// through them, so the generated motion follows the path and the jump arc.
const SPAN_DEFAULT = { kind: 'walk', height: 0.6, count: 0, flip: false };
const spanOf = (k) => ({ ...SPAN_DEFAULT, ...(k.span || {}) });
const JUMP_MIN_H = 0.1, JUMP_MAX_H = 2.5;
const lerp = (a, b, t) => a + (b - a) * t;

// Frames (from the span's start) where a jump leaves and meets the floor: the whole span.
const jumpWindowOf = (a, gap) => [0, gap];

/**
 * A jump's height is how far its lowest point clears the floor at the top. The hips arc
 * up to where the jump pose's hips are at that clearance.
 */
function jumpApex(a, b, A, B) {
  const gap = b.frame - a.frame, [t0, t1] = jumpWindowOf(a, gap), t = (t0 + t1) / 2 / gap;
  const lift = spanOf(a).height + lerp(a.y || 0, b.y || 0, t);
  return { t, lift, hipsY: S.poses.jump.root[1] + lift, rise: Math.max(0.02, S.poses.jump.root[1] + lift - lerp(A.y, B.y, t)) };
}

/** Hips position along a span's path at frame f: straight, plus the jump arc. */
function spanPoint(a, b, f, A = keyFrameData(a).pos, B = keyFrameData(b).pos, out = new THREE.Vector3()) {
  const gap = b.frame - a.frame, l = f - a.frame;
  out.lerpVectors(A, B, l / gap);
  if (spanOf(a).kind === 'jump') {
    const [t0, t1] = jumpWindowOf(a, gap), u = (l - t0) / (t1 - t0);
    if (u > 0 && u < 1) out.y += jumpApex(a, b, A, B).rise * 4 * u * (1 - u);
  }
  return out;
}

/** Most breakdowns a span of `gap` frames can hold (about 5 frames apart at least). */
const maxBreakdowns = (gap) => Math.min(12, Math.max(0, Math.floor(gap / 5) - 1));

/**
 * The breakdown poses inside the span a → b: as many as its slider asks for.
 *   walk ... steps spread evenly along the path, alternating the leading foot
 *   jump ... jump poses spread along the arc
 * They face along the path, or back along it when the span's Flip is on.
 */
function breakdownsOf(a, b) {
  const sp = spanOf(a), gap = b.frame - a.frame;
  const n = Math.min(sp.count | 0, maxBreakdowns(gap));
  if (n <= 0) return [];
  const A = keyFrameData(a).pos, B = keyFrameData(b).pos;
  const dx = B.x - A.x, dz = B.z - A.z, dist = Math.hypot(dx, dz);
  const ahead = dist > 0.2 ? Math.atan2(dx, dz) : a.yaw; // face where it's going ...
  const travel = sp.flip ? wrapAngle(ahead + Math.PI) : ahead; // ... or back the way it came
  const out = [], used = new Set();
  const add = (l, pose, y, role) => {
    const f = a.frame + l, t = l / gap;
    if (l <= 0 || l >= gap || used.has(f)) return;
    used.add(f);
    out.push({ bd: true, id: `bd${a.id}.${role}.${f}`, owner: a.id, role, frame: f, pose,
      x: lerp(A.x, B.x, t), y: Math.max(0, y), z: lerp(A.z, B.z, t), yaw: travel, pitch: 0 });
  };
  if (sp.kind === 'walk') {
    const step = (i) => (i % 2 ? 'walk_l' : 'walk_r');
    for (let i = 1; i <= n; i++) add(Math.round((i * gap) / (n + 1)), step(i), 0, 'step');
  } else {
    const [t0, t1] = jumpWindowOf(a, gap), air = n;
    const p = new THREE.Vector3();
    for (let i = 1; i <= air; i++) { // on the arc: the jump pose's hips ride the path
      const l = Math.round(t0 + ((t1 - t0) * i) / (air + 1));
      add(l, 'jump', spanPoint(a, b, a.frame + l, A, B, p).y - S.poses.jump.root[1], 'air');
    }
    out.sort((x, y) => x.frame - y.frame);
  }
  return out;
}

/**
 * Chain signatures: segment i depends on every pose up to i + 1 (context flows forward).
 * `keys` are the poses the AI runs through: saved keyframes plus any breakdowns; `real`
 * are the saved keyframes alone.
 */
function chain() {
  const real = sortedKeys();
  const keys = [];
  real.forEach((k, i) => { keys.push(k); if (i + 1 < real.length) keys.push(...breakdownsOf(k, real[i + 1])); });
  const sig = (k) => [k.custom ? k.custom.uid : k.pose, k.x.toFixed(3), (k.y || 0).toFixed(3), k.z.toFixed(3), k.yaw.toFixed(4), (k.pitch || 0).toFixed(4), posesToUse(k)].join(',');
  const sigs = [];
  let prev = keys.length ? 'S' + sig(keys[0]) : '';
  for (let i = 0; i + 1 < keys.length; i++) {
    prev = hash(prev + '|' + (keys[i + 1].frame - keys[i].frame) + ':' + sig(keys[i + 1]));
    sigs.push(prev);
  }
  return { keys, real, sigs };
}

function syncState() {
  const { keys, real, sigs } = chain();
  const ok = sigs.map((s) => S.cache.has(s));
  const firstBad = ok.indexOf(false);
  let tag = null;
  if (S.lastRun.length && firstBad >= 0) tag = firstBad < S.lastRun.length ? 'stale' : 'more';
  return { keys, real, sigs, ok, firstBad, tag };
}

// ---------------------------------------------------------------- scene
const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.prepend(renderer.domElement);
renderer.domElement.tabIndex = 0;

const BG = new THREE.Color('#121418');
const scene = new THREE.Scene();
scene.background = BG;
const FOG = new THREE.Fog(BG, 18, 70);
scene.fog = FOG;
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 500);
camera.position.set(4.2, 2.6, 5.4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
// The perspective camera is always the one the controls drive. Snapped views render through an
// orthographic twin that copies its position and facing and frames the same height at the target.
const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, 1000);
let rcam = camera; // the camera we render and pick through
const MAX_POLAR = controls.maxPolarAngle;

renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;
scene.add(new THREE.HemisphereLight('#dfe8ff', '#3a3530', 0.8));
const sun = new THREE.DirectionalLight('#fff3e2', 2.2);
sun.position.set(3, 7, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 25 });
sun.shadow.bias = -0.0004;
scene.add(sun, sun.target);

// Infinite floor: a large plane with a distance-faded procedural grid.
const floorMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  uniforms: { uCam: { value: new THREE.Vector3() }, uBg: { value: BG } },
  vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `
    varying vec3 vW; uniform vec3 uCam; uniform vec3 uBg;
    float grid(vec2 p, float s){ vec2 g = abs(fract(p/s - 0.5) - 0.5) / fwidth(p/s); return 1.0 - min(min(g.x, g.y), 1.0); }
    void main(){
      float d = length(vW.xz - uCam.xz);
      float minor = grid(vW.xz, 0.5) * 0.25, major = grid(vW.xz, 2.0) * 0.6;
      float axis = (1.0 - min(abs(vW.x)/fwidth(vW.x), 1.0)) + (1.0 - min(abs(vW.z)/fwidth(vW.z), 1.0));
      float fade = 1.0 - smoothstep(6.0, 40.0, d);
      vec3 base = uBg * 1.6;
      vec3 col = base + vec3(minor + major) * 0.05 * fade + vec3(0.06,0.06,0.08) * clamp(axis,0.0,1.0) * fade;
      gl_FragColor = vec4(mix(uBg, col, fade * 0.9 + 0.1), 1.0);
      #include <colorspace_fragment>
    }`,
});
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), floorMat);
floor.rotation.x = -Math.PI / 2;
floor.renderOrder = -1;
const shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.ShadowMaterial({ opacity: 0.35 }));
shadowCatcher.rotation.x = -Math.PI / 2;
shadowCatcher.position.y = 0.001;
shadowCatcher.receiveShadow = true;
scene.add(floor, shadowCatcher);
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

// ---------------------------------------------------------------- mannequins
let kit, actor;
const keyGhosts = new Map(); // key id -> mannequin (orange onion skin)
const genGhosts = [];        // pool of blue onion skins
const ONION_OFFSETS = [-15, -10, -5, 5, 10, 15];
const KEY_GHOST_HIDE = 3; // frames either side of a key where its orange ghost is hidden
// Actor: light glossy surface with dark joints. Saved poses: solid glossy orange.
// In-betweens: flat, unlit, see-through blue silhouettes.
const glossy = (color, extra = {}) => new THREE.MeshPhysicalMaterial({ color, roughness: 0.32, clearcoat: 0.8, clearcoatRoughness: 0.18, ...extra });
const woodMat = glossy('#cfd4dc');
const jointMat = glossy('#2d3239', { roughness: 0.4, metalness: 0.25 });
const hoverMat = glossy('#e4e9f0', { emissive: '#18202c' });
// Saved poses: one flat, unlit orange, like the blue in-betweens. 50% opaque at rest
// (hover/selected brighter), fading to 25% while the timeline moves.
const flat = (color, opacity = 1) => new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity });
const keyGhostMat = flat('#ff7a1a', 0.5);
const keyJointMat = keyGhostMat;
const keyGhostHover = flat('#ffa45c', 0.75);
const keyGhostFade = flat('#ff7a1a', 0.25);
const keyJointFade = keyGhostFade;
const genGhostMats = ONION_OFFSETS.map((o) => new THREE.MeshBasicMaterial({
  color: '#4f9dff', transparent: true, depthWrite: true, opacity: 0.34 * (1 - (Math.abs(o) - 5) / 15) + 0.08 }));
// Breakdowns: flat mint, like the saved poses (50% at rest, 25% while the timeline moves).
const BD_COLOR = '#35c48d';
const bdGhostMat = flat(BD_COLOR, 0.5);
const bdGhostFade = flat(BD_COLOR, 0.25);
const bdGhostDim = flat(BD_COLOR, 0.1);
const bdGhosts = []; // pool

// ---------------------------------------------------------------- motion paths (root bone)
// Orange: the saved poses' hips along each span (solid, 50%), with a timing dot per frame.
// Blue: the hips' generated trajectory (solid), with a dot every PATH_TICK frames.
const PATH_TICK = 5;
const PLAY_WINDOW = 10; // while playing, joint and in-between paths show only this many frames either side
const playWindow = () => (S.playing ? [S.frame - PLAY_WINDOW, S.frame + PLAY_WINDOW] : null);

// Paused, paths are solid near the playhead and fade to FADE_MIN further away.
const FADE_NEAR = 10, FADE_FAR = 45, FADE_MIN = 0.5;
const fadeOf = (f) => {
  if (S.playing) return 1;
  const d = Math.abs(f - S.frame);
  const t = Math.min(1, Math.max(0, (d - FADE_NEAR) / (FADE_FAR - FADE_NEAR)));
  return 1 - (1 - FADE_MIN) * t * t * (3 - 2 * t);
};
/** A path line whose per-vertex colour's red channel is its opacity (see fadeLine). */
function fadeLineMat(params) {
  const m = new LineMaterial({ ...params, vertexColors: true, transparent: true });
  m.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <color_fragment>', 'alpha *= vColor.x;'); };
  m.customProgramCacheKey = () => 'line-fade';
  return m;
}
/** Dots whose per-instance colour's red channel is their opacity. */
function fadeDotMat(color, opacity = 1) {
  const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
  // Until a mesh's per-dot colours exist it has no vColor, so the line only applies once they do.
  m.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <color_fragment>', '#ifdef USE_COLOR\n  diffuseColor.a *= vColor.x;\n#endif'); };
  m.customProgramCacheKey = () => 'dot-fade';
  return m;
}
const keyPathMat = fadeLineMat({ color: 0xff7a1a, linewidth: 2.5, opacity: 0.5 }); // solid, half see-through: the ticks carry the timing
const genPathMat = fadeLineMat({ color: 0x4f9dff, linewidth: 3, opacity: 0.95 });
const keyPath = new LineSegments2(new LineSegmentsGeometry(), keyPathMat);
const genPath = new LineSegments2(new LineSegmentsGeometry(), genPathMat);
const dotGeo = new THREE.SphereGeometry(1, 12, 8);
const keyDots = new THREE.InstancedMesh(dotGeo, fadeDotMat('#ff7a1a'), 256);
const genDots = new THREE.InstancedMesh(dotGeo, fadeDotMat('#8fbcff'), 4096);
// Timing dots: a small solid dot on the orange path at every frame, like an animator's
// timing chart (close together = slow, spread out = fast).
const TIMING_DOT_R = 0.016;
const keyTicks = new THREE.InstancedMesh(dotGeo, fadeDotMat('#ff7a1a'), 8192);
for (const o of [keyPath, keyTicks, genPath, keyDots, genDots]) { o.frustumCulled = false; o.visible = false; scene.add(o); }
const paths = { keySig: '', genSig: '' };
const _dot = new THREE.Matrix4();
const _fadeCol = new THREE.Color();

/**
 * Give a path line new segments. three.js caches a line geometry's segment count the
 * first time it draws it, so reusing the geometry with a different count draws stale or
 * stray segments. A fresh geometry per change avoids that.
 * `frames` (two per segment) are the animation frames its ends belong to, for the fade.
 */
function setLine(line, seg, frames = null) {
  line.geometry.dispose();
  line.geometry = new LineSegmentsGeometry();
  if (seg.length) line.geometry.setPositions(seg);
  if (line.material.vertexColors && seg.length) line.geometry.setColors(new Float32Array(seg.length).fill(1));
  line.userData.frames = frames;
  line.userData.fadeAt = null;
}

/** Re-fade a path for the current frame (cheap: rewrites its colour buffer only on change). */
function fadeLine(line) {
  const fr = line.userData.frames, attr = line.geometry.attributes.instanceColorStart;
  if (!fr || !attr) return;
  const at = S.playing ? 'play' : S.frame;
  if (line.userData.fadeAt === at) return;
  line.userData.fadeAt = at;
  const arr = attr.data.array;
  for (let i = 0; i < fr.length / 2; i++) { arr[i * 6] = fadeOf(fr[i * 2]); arr[i * 6 + 3] = fadeOf(fr[i * 2 + 1]); }
  attr.data.needsUpdate = true;
}

/** Place one small sphere per point (x, y, z triples), up to the mesh's capacity. */
function setDots(mesh, points, radius, frames = null) {
  const n = Math.min(points.length / 3, mesh.instanceMatrix.count);
  for (let i = 0; i < n; i++) {
    _dot.makeScale(radius, radius, radius).setPosition(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    mesh.setMatrixAt(i, _dot);
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.userData.frames = frames;
  mesh.userData.fadeAt = null;
}
function fadeDots(mesh) {
  const fr = mesh.userData.frames;
  if (!fr) return;
  const at = S.playing ? 'play' : S.frame;
  if (mesh.userData.fadeAt === at) return;
  mesh.userData.fadeAt = at;
  for (let i = 0; i < mesh.count; i++) mesh.setColorAt(i, _fadeCol.setRGB(fadeOf(fr[i]), 1, 1));
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

const spanHandles = []; // jump apex handles, one per jump span (pooled)

function updatePaths(fs) {
  // Orange: the saved poses' hips, frame by frame along each span (straight for a walk,
  // arcing for a jump).
  const real = fs.real;
  const keySig = real.map((k) => [k.frame, k.pose, k.custom?.uid, k.x, k.y, k.z, k.yaw, k.pitch, JSON.stringify(k.span || '')].join(',')).join('|');
  if (keySig !== paths.keySig) {
    paths.keySig = keySig;
    const seg = [], frames = [], dots = [], dotFrames = [], p = new THREE.Vector3(), q = new THREE.Vector3();
    const ticks = [], tickFrames = [];
    real.forEach((a, i) => {
      const A = keyFrameData(a).pos;
      dots.push(A.x, A.y, A.z); dotFrames.push(a.frame);
      const b = real[i + 1];
      if (!b) return;
      const B = keyFrameData(b).pos;
      for (let f = a.frame; f < b.frame; f++) {
        spanPoint(a, b, f, A, B, p); spanPoint(a, b, f + 1, A, B, q);
        seg.push(p.x, p.y, p.z, q.x, q.y, q.z); frames.push(f, f + 1);
        if (f === a.frame) continue; // the keyframe has its own, bigger dot
        ticks.push(p.x, p.y, p.z); tickFrames.push(f);
      }
    });
    setLine(keyPath, seg, frames);
    setDots(keyTicks, ticks, TIMING_DOT_R, tickFrames);
    setDots(keyDots, dots, 0.035, dotFrames);
    paths.keySegs = seg.length;
  }
  keyPath.visible = S.show.keyPath && paths.keySegs > 0;
  keyTicks.visible = keyPath.visible; // timing dots always come with the path
  fadeDots(keyTicks);
  keyDots.visible = S.show.keyPath && real.length > 0;
  fadeLine(keyPath); fadeDots(keyDots);

  // Blue: every generated frame's hip position, segment by segment. While playing, only
  // the PLAY_WINDOW frames either side of the playhead.
  const win = playWindow();
  const genSig = fs.sigs.filter((sg, i) => fs.ok[i]).join('|') + (win ? `@${S.frame}` : '');
  if (genSig !== paths.genSig) {
    paths.genSig = genSig;
    const seg = [], frames = [], dots = [], dotFrames = [];
    fs.sigs.forEach((sg, i) => {
      if (!fs.ok[i]) return;
      const r = S.cache.get(sg).root, n = S.cache.get(sg).frames, base = fs.keys[i].frame;
      for (let f = 0; f < n; f++) {
        const F = base + f;
        if (win && (F < win[0] || F > win[1])) continue;
        if (f + 1 < n && (!win || F + 1 <= win[1])) { seg.push(r[f * 3], r[f * 3 + 1], r[f * 3 + 2], r[f * 3 + 3], r[f * 3 + 4], r[f * 3 + 5]); frames.push(F, F + 1); }
        if (F % PATH_TICK === 0 && f > 0) { dots.push(r[f * 3], r[f * 3 + 1], r[f * 3 + 2]); dotFrames.push(F); }
      }
    });
    setLine(genPath, seg, frames);
    setDots(genDots, dots, 0.016, dotFrames);
    paths.genSegs = seg.length;
  }
  genPath.visible = S.show.genPath && paths.genSegs > 0;
  genDots.visible = genPath.visible;
  fadeLine(genPath); fadeDots(genDots);
}

// ---------------------------------------------------------------- bones (white mannequin)
// Yellow lines over the body (drawn on top), with a dot per joint. Works with the body
// hidden too, for a skeleton-only view.
const boneMat = new LineMaterial({ color: 0xf2c230, linewidth: 2, depthTest: false, transparent: true, opacity: 0.95 });
const boneLine = new LineSegments2(new LineSegmentsGeometry(), boneMat);
const boneDots = new THREE.InstancedMesh(dotGeo, new THREE.MeshBasicMaterial({ color: '#f2c230', depthTest: false, transparent: true }), 64);
for (const o of [boneLine, boneDots]) { o.frustumCulled = false; o.visible = false; o.renderOrder = 998; scene.add(o); }

function updateBones() {
  const on = S.show.bones;
  boneLine.visible = boneDots.visible = on;
  if (!on) return;
  const { seg, joints } = actor.boneLines();
  if (boneLine.geometry.attributes.instanceStart?.count !== seg.length / 6) setLine(boneLine, seg); // count changed: fresh geometry
  else boneLine.geometry.setPositions(seg);
  setDots(boneDots, joints, 0.018);
}

// ---------------------------------------------------------------- joint paths (white mannequin)
// White trails for the head, hands and feet over the whole animation, built from what the
// mannequin shows on each frame (keyframe, AI in-between or blend), with a dot every
// PATH_TICK frames. Rebuilt when the keyframes or generated motion change (not mid-drag).
const JOINT_PATH_BONES = ['Head', 'LeftHand', 'RightHand', 'LeftToeBase', 'RightToeBase'];
const jointPathMat = fadeLineMat({ color: 0xffffff, linewidth: 1.6, opacity: 0.7 });
const jointPath = new LineSegments2(new LineSegmentsGeometry(), jointPathMat);
const jointDots = new THREE.InstancedMesh(dotGeo, fadeDotMat('#ffffff', 0.8), 4096);
for (const o of [jointPath, jointDots]) { o.frustumCulled = false; o.visible = false; scene.add(o); }
let jointScratch = null; // hidden mannequin used to sample poses
const jointPaths = { sig: '' };

/** Pose `m` as the white mannequin appears at frame f (ignoring an unsaved draft). */
function poseAtFrame(m, f, fs) {
  const { keys, sigs, ok } = fs;
  const key = keys.find((k) => k.frame === f);
  if (key) { applyPoseOf(m, key); return; }
  for (let i = 0; i + 1 < keys.length; i++) {
    const a = keys[i], b = keys[i + 1];
    if (a.frame < f && f < b.frame) {
      if (ok[i]) {
        const seg = S.cache.get(sigs[i]), local = f - a.frame, r = seg.root;
        m.setPose(r[local * 3], r[local * 3 + 1], r[local * 3 + 2], seg.rot, local * kit.names.length * 4);
      } else {
        const bl = blendSeg(a, b, f, 0);
        m.setPose(bl.root[0], bl.root[1], bl.root[2], bl.rot, 0);
      }
      return;
    }
  }
}

function updateJointPaths(fs) {
  const on = S.show.jointPaths && fs.keys.length >= 2;
  jointPath.visible = jointDots.visible = on;
  if (!on || drag) return;
  // Sample the whole animation once per change to the keyframes / generated motion ...
  const sig = fs.sigs.map((sg, i) => sg + (fs.ok[i] ? '+' : '-')).join('|') + '#' + paths.keySig;
  if (sig !== jointPaths.sig) {
    jointPaths.sig = sig;
    if (!jointScratch) { jointScratch = kit.create(keyGhostMat); jointScratch.visible = false; scene.add(jointScratch.group); }
    const first = fs.keys[0].frame, last = fs.keys[fs.keys.length - 1].frame, track = [];
    for (let f = first; f <= last; f++) {
      poseAtFrame(jointScratch, f, fs);
      jointScratch.jointPositions(JOINT_PATH_BONES, track);
    }
    Object.assign(jointPaths, { track, first, built: null });
  }
  // ... then draw all of it, or (while playing) just the window around the playhead.
  const win = playWindow(), built = win ? `w${S.frame}` : 'all';
  if (built === jointPaths.built) { fadeLine(jointPath); fadeDots(jointDots); return; }
  jointPaths.built = built;
  const { track, first } = jointPaths, J = JOINT_PATH_BONES.length, n = track.length / (3 * J);
  const i0 = win ? Math.max(0, win[0] - first) : 0, i1 = win ? Math.min(n - 1, win[1] - first) : n - 1;
  const seg = [], frames = [], dots = [], dotFrames = [];
  for (let i = i0; i <= i1; i++) {
    for (let j = 0; j < J; j++) {
      const k = (i * J + j) * 3;
      if (i + 1 <= i1) { const k2 = ((i + 1) * J + j) * 3; seg.push(track[k], track[k + 1], track[k + 2], track[k2], track[k2 + 1], track[k2 + 2]); frames.push(first + i, first + i + 1); }
      if ((first + i) % PATH_TICK === 0) { dots.push(track[k], track[k + 1], track[k + 2]); dotFrames.push(first + i); }
    }
  }
  setLine(jointPath, seg, frames);
  setDots(jointDots, dots, 0.011, dotFrames);
  fadeLine(jointPath); fadeDots(jointDots);
}

// ---------------------------------------------------------------- foot contacts
// A red outline on the floor round the sole of each of the white mannequin's feet, shown
// while that foot touches the floor (its heel, ball or toe at floor level).
const FOOT_BONES = ['LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightFoot', 'RightToeBase', 'RightToe_End'];
const CONTACT_H = 0.025;     // heel, ball or toe this close to the floor touches it (m)
const HEEL_BACK = 0.07;      // the back of the heel, behind the ankle (m)
const TOE_TIP = 0.03;        // the tip of the shoe, past the toe joint (m)
const SOLE_W = 1.15;         // the outline sits a little outside the shoe's sides
/** Sole outline, 1 long (heel at -0.5, toes at +0.5) in x (across) and z (along); left foot's outside is +x. */
function soleOutline(mirror) {
  const pts = [[0, -0.5], [0.12, -0.46], [0.155, -0.34], [0.15, -0.12], [0.18, 0.08], [0.2, 0.26], [0.17, 0.4], [0.09, 0.485],
    [-0.03, 0.5], [-0.14, 0.44], [-0.19, 0.27], [-0.13, 0.08], [-0.09, -0.1], [-0.12, -0.32], [-0.11, -0.45]]
    .map(([x, y]) => new THREE.Vector2(mirror ? -x : x, y));
  return new THREE.SplineCurve([...pts, pts[0]]).getSpacedPoints(64);
}
const footLineMat = new LineMaterial({ color: 0xff4d4f, linewidth: 2, transparent: true, opacity: 0.95 });
const footLines = [false, true].map((mirror) => { // [left, right]
  const outline = soleOutline(mirror), seg = [];
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i + 1) % outline.length];
    seg.push(a.x * SOLE_W, 0, a.y, b.x * SOLE_W, 0, b.y);
  }
  const g = new LineSegmentsGeometry(); g.setPositions(seg);
  const line = new LineSegments2(g, footLineMat);
  line.frustumCulled = false; line.visible = false; line.renderOrder = 3;
  scene.add(line);
  return line;
});
const _feetAt = [];
function updateFootContacts() {
  const on = S.show.feet && S.show.actor;
  if (!on) { for (const l of footLines) l.visible = false; return; }
  const probe = kit.groundProbes.find((g) => kit.bones[g.i].name === 'LeftFoot');
  const ankleH = probe ? probe.h : 0.088; // the ankle's height above the floor with the foot flat
  _feetAt.length = 0;
  actor.jointPositions(FOOT_BONES, _feetAt);
  footLines.forEach((line, side) => {
    const at = (j) => { const k = (side * 3 + j) * 3; return [_feetAt[k], _feetAt[k + 1], _feetAt[k + 2]]; };
    const ank = at(0), ball = at(1), toe = at(2);
    line.visible = Math.min(ank[1] - ankleH, ball[1], toe[1]) < CONTACT_H;
    if (!line.visible) return;
    const reach = Math.hypot(toe[0] - ank[0], toe[2] - ank[2]) || 0.2;
    const dx = (toe[0] - ank[0]) / reach, dz = (toe[2] - ank[2]) / reach;
    const len = reach + HEEL_BACK + TOE_TIP, mid = (reach + TOE_TIP - HEEL_BACK) / 2;
    line.position.set(ank[0] + dx * mid, 0.004, ank[2] + dz * mid);
    line.rotation.set(0, Math.atan2(dx, dz), 0);
    line.scale.set(len, 1, len);
  });
}

// ---------------------------------------------------------------- gizmo
const gizmo = new THREE.Group();
gizmo.visible = false;
const gizmoPick = [];
const RING_R = 0.55;          // both rotation rings share one size
const TURN_R = RING_R;
const TURN_PER_PX = 0.012; // radians of turn per pixel dragged on the turn ring
const ARROW_START = RING_R + 0.07, ARROW_LEN = 0.42, HEAD_LEN = 0.14; // arrows start outside the rings
function gizmoMat(color) { return new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }); }
function makeArrow(axis, color) {
  const g = new THREE.Group();
  const m = gizmoMat(color);
  // Arrows start outside the turn ring so the two never overlap.
  // The cone's base sits exactly on the shaft's end, so the arrow reads as one piece.
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, ARROW_LEN, 10), m);
  shaft.position.y = ARROW_START + ARROW_LEN / 2;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.05, HEAD_LEN, 18), m);
  head.position.y = ARROW_START + ARROW_LEN + HEAD_LEN / 2;
  const hit = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, ARROW_LEN + HEAD_LEN + 0.08, 8), new THREE.MeshBasicMaterial({ visible: false }));
  hit.position.y = ARROW_START + (ARROW_LEN + HEAD_LEN) / 2;
  g.add(shaft, head, hit);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  hit.userData.handle = { type: 'axis', axis: axis.clone(), mat: m, color };
  gizmoPick.push(hit);
  return g;
}
gizmo.add(makeArrow(new THREE.Vector3(1, 0, 0), '#ff5a5a'), makeArrow(new THREE.Vector3(0, 1, 0), '#5ad16a'), makeArrow(new THREE.Vector3(0, 0, 1), '#4f9dff'));
{
  const m = gizmoMat('#ffd040');
  const ring = new THREE.Mesh(new THREE.TorusGeometry(TURN_R, 0.014, 8, 96), m);
  ring.rotation.x = Math.PI / 2;
  const hit = new THREE.Mesh(new THREE.TorusGeometry(TURN_R, 0.07, 6, 64), new THREE.MeshBasicMaterial({ visible: false }));
  hit.rotation.x = Math.PI / 2;
  hit.userData.handle = { type: 'ring', mat: m, color: '#ffd040' };
  const facing = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.1, 12), m);
  facing.rotation.x = Math.PI / 2;
  facing.position.z = TURN_R + 0.06;
  gizmo.userData.facing = facing;
  gizmo.add(ring, hit, facing);
  gizmoPick.push(hit);
}
gizmo.traverse((o) => { o.renderOrder = 999; });
gizmo.position.y = 0.015;
scene.add(gizmo);

// ---------------------------------------------------------------- span labels + jump handles
// Each span gets a small label on the floor halfway along it (Walk / Jump, click to change).
// A jump also gets a handle at the top of its arc: drag it up or down to set the height.
const SPAN_HANDLE = '#5ad16a'; // the gizmo's vertical-axis green
function makeSpanHandle() {
  const mat = gizmoMat(SPAN_HANDLE);
  const group = new THREE.Group();
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 10), mat);
  const up = new THREE.Mesh(new THREE.ConeGeometry(0.034, 0.075, 14), mat);
  up.position.y = 0.095;
  const down = up.clone();
  down.position.y = -0.095; down.rotation.x = Math.PI;
  const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, -1, 0)]),
    new THREE.LineBasicMaterial({ color: SPAN_HANDLE, transparent: true, opacity: 0.45, depthTest: false }));
  const hit = new THREE.Mesh(new THREE.SphereGeometry(0.15, 8, 6), new THREE.MeshBasicMaterial({ visible: false }));
  group.add(ball, up, down, stem, hit);
  group.traverse((o) => { o.renderOrder = 997; });
  group.visible = false;
  scene.add(group);
  const sh = { group, hit, stem, mat, key: null };
  hit.userData.span = sh;
  return sh;
}

const chipLayer = $('span-chips');
const chips = new Map(); // span (its first keyframe's id) -> label element
let chipOpen = null;     // span whose Walk / Jump list is open
const SPAN_KINDS = [['walk', 'Walk'], ['jump', 'Jump']];

function setSpan(k, patch) {
  k.span = { ...spanOf(k), ...patch };
  if (!keyAt(S.frame)) { S.draft = null; resetDraft(); }
  save();
}
// Each label: the Walk / Jump selector, and a slider for how many breakdowns the span gets.
function makeChip(id) {
  const el = document.createElement('div');
  el.className = 'span-chip';
  el.dataset.key = id;
  el.innerHTML = `<div class="sc-row">
      <button class="sc-name" title="What happens between these two keyframes"><span class="sc-kind"></span><small></small><span class="chev">▾</span></button>
      <label class="sc-bd" title="Breakdowns: extra poses along this path for the AI to follow (0 = none)">
        <span class="ic"></span><input type="range" min="0" max="12" step="1" value="0" aria-label="Number of breakdowns"><output>0</output>
      </label>
      <label class="sc-flip" title="Turn the breakdowns to face back along the path"><input type="checkbox"><span class="ic"></span>Flip direction</label>
    </div>
    <div class="sc-list hidden">${SPAN_KINDS.map(([k, label]) => `<button data-kind="${k}">${label}</button>`).join('')}</div>`;
  setIcon(el.querySelector('.sc-bd .ic'), 'footprints');
  setIcon(el.querySelector('.sc-flip .ic'), 'arrow-left-right');
  el.querySelector('input[type=range]').addEventListener('input', (e) => {
    const k = S.keys.find((x) => x.id === id);
    if (k) setSpan(k, { count: +e.target.value });
  });
  el.querySelector('input[type=checkbox]').addEventListener('change', (e) => {
    const k = S.keys.find((x) => x.id === id);
    if (k) setSpan(k, { flip: e.target.checked });
  });
  chipLayer.append(el);
  return el;
}
chipLayer.addEventListener('click', (e) => {
  const el = e.target.closest('.span-chip');
  const k = el && S.keys.find((x) => x.id === +el.dataset.key);
  if (!k) return;
  const kind = e.target.closest('[data-kind]')?.dataset.kind;
  if (kind) { if (kind !== spanOf(k).kind) setSpan(k, { kind }); chipOpen = null; return; }
  if (e.target.closest('.sc-name')) chipOpen = chipOpen === k.id ? null : k.id;
});
window.addEventListener('pointerdown', (e) => { if (!e.target.closest?.('.span-chip')) chipOpen = null; }, true);

// A span's settings show only once you click its path. Hovering a path lights it up green.
let hoverSpan = null; // span (first keyframe id) under the cursor
const spanLineMat = new LineMaterial({ color: 0x35c48d, linewidth: 4.5, transparent: true, opacity: 0.95 });
const spanLine = new LineSegments2(new LineSegmentsGeometry(), spanLineMat);
spanLine.frustumCulled = false; spanLine.visible = false; spanLine.renderOrder = 2;
scene.add(spanLine);
const spanLineState = { sig: '' };
const SPAN_PICK_PX = 9;

/** The span whose path passes within a few pixels of the cursor (closest wins), or null. */
function pickSpanPath() {
  if (!S.show.keyPath || S.playing || S.scrubbing) return null;
  const real = sortedKeys(), r = canvas.getBoundingClientRect();
  const mx = (pointer.x * 0.5 + 0.5) * r.width, my = (-pointer.y * 0.5 + 0.5) * r.height;
  const p = new THREE.Vector3(), toPx = (v) => { v.project(rcam); return v.z < 1 ? [(v.x * 0.5 + 0.5) * r.width, (-v.y * 0.5 + 0.5) * r.height] : null; };
  let best = null, bestD = SPAN_PICK_PX;
  for (let i = 0; i + 1 < real.length; i++) {
    const a = real[i], b = real[i + 1], A = keyFrameData(a).pos, B = keyFrameData(b).pos;
    let prev = toPx(spanPoint(a, b, a.frame, A, B, p));
    for (let f = a.frame + 1; f <= b.frame; f++) {
      const cur = toPx(spanPoint(a, b, f, A, B, p));
      if (prev && cur) {
        const dx = cur[0] - prev[0], dy = cur[1] - prev[1], len2 = dx * dx + dy * dy || 1;
        const t = Math.max(0, Math.min(1, ((mx - prev[0]) * dx + (my - prev[1]) * dy) / len2));
        const d = Math.hypot(mx - (prev[0] + t * dx), my - (prev[1] + t * dy));
        if (d < bestD) { bestD = d; best = a; }
      }
      prev = cur;
    }
  }
  return best;
}

/** Green overlay on the selected span's path (solid) or the hovered one (fainter). */
function updateSpanLine(real) {
  const id = S.spanSel ?? hoverSpan;
  const i = real.findIndex((k) => k.id === id);
  const on = i >= 0 && i + 1 < real.length && S.show.keyPath && !S.playing && !S.scrubbing && !S.poseMode;
  spanLine.visible = on;
  if (!on) return;
  spanLineMat.opacity = id === S.spanSel ? 0.95 : 0.5;
  const sig = id + '|' + paths.keySig;
  if (sig === spanLineState.sig) return;
  spanLineState.sig = sig;
  const a = real[i], b = real[i + 1], A = keyFrameData(a).pos, B = keyFrameData(b).pos, seg = [], p = new THREE.Vector3(), q = new THREE.Vector3();
  for (let f = a.frame; f < b.frame; f++) { spanPoint(a, b, f, A, B, p); spanPoint(a, b, f + 1, A, B, q); seg.push(p.x, p.y, p.z, q.x, q.y, q.z); }
  setLine(spanLine, seg);
}

const _apex = new THREE.Vector3();
function updateSpans(fs) {
  const real = fs.real;
  const busy = S.playing || S.scrubbing || !!S.poseMode;
  const showChips = !busy;
  const r = canvas.getBoundingClientRect();
  const seen = new Set();
  let used = 0;
  for (let i = 0; i + 1 < real.length; i++) {
    const a = real[i], b = real[i + 1], sp = spanOf(a), gap = b.frame - a.frame;
    const A = keyFrameData(a).pos, B = keyFrameData(b).pos;
    seen.add(a.id);
    let el = chips.get(a.id);
    if (!el) { el = makeChip(a.id); chips.set(a.id, el); }
    tmpV.lerpVectors(A, B, 0.5).setY(0.02).project(rcam);
    const vis = showChips && S.spanSel === a.id && tmpV.z < 1;
    el.classList.toggle('hidden', !vis);
    if (vis) {
      el.style.left = `${(tmpV.x * 0.5 + 0.5) * r.width}px`;
      el.style.top = `${(-tmpV.y * 0.5 + 0.5) * r.height}px`;
      const open = chipOpen === a.id, max = maxBreakdowns(gap), n = Math.min(sp.count | 0, max);
      const kindText = sp.kind === 'jump' ? 'Jump' : 'Walk', heightText = sp.kind === 'jump' ? `${sp.height.toFixed(2)} m` : '';
      const kindEl = el.querySelector('.sc-kind'), small = el.querySelector('.sc-name small');
      if (kindEl.textContent !== kindText) kindEl.textContent = kindText;
      if (small.textContent !== heightText) small.textContent = heightText;
      const input = el.querySelector('input[type=range]'), outEl = el.querySelector('output');
      const cb = el.querySelector('input[type=checkbox]');
      if (cb.checked !== !!sp.flip) cb.checked = !!sp.flip;
      if (cb.disabled !== (n === 0)) cb.disabled = n === 0; // nothing to turn without breakdowns
      if (input.max !== String(max)) { input.max = max; input.disabled = max === 0; }
      if (+input.value !== n) input.value = n;
      input.style.setProperty('--fill', `${max ? (n / max) * 100 : 0}%`);
      if (outEl.textContent !== String(n)) outEl.textContent = n;
      el.querySelector('.sc-list').classList.toggle('hidden', !open);
      for (const btn of el.querySelectorAll('[data-kind]')) btn.classList.toggle('on', btn.dataset.kind === sp.kind);
      el.classList.toggle('jump', sp.kind === 'jump');
      el.classList.toggle('has-bd', n > 0);
      el.classList.toggle('flip-on', !!sp.flip && n > 0);
      el.classList.toggle('open', open);
    }
    if (sp.kind !== 'jump') continue;
    const sh = spanHandles[used] || (spanHandles[used] = makeSpanHandle());
    used++;
    const ja = jumpApex(a, b, A, B);
    _apex.lerpVectors(A, B, ja.t);
    _apex.y = ja.hipsY; // top of the arc
    sh.key = a;
    sh.group.position.copy(_apex);
    sh.stem.scale.y = ja.hipsY - 0.02; // a guide down to the floor
    // three.js hides only on `false` exactly, so this must be a real boolean (not null).
    sh.group.visible = (!busy && S.show.spanHandles) || !!(drag && drag.type === 'height' && drag.key === a);
    sh.mat.color.set((hover && hover.handle === sh) || (drag && drag.key === a) ? '#ffffff' : SPAN_HANDLE);
  }
  for (let i = used; i < spanHandles.length; i++) spanHandles[i].group.visible = false;
  for (const [id, el] of chips) if (!seen.has(id)) { el.remove(); chips.delete(id); }
  if (S.spanSel != null && !seen.has(S.spanSel)) S.spanSel = null; // its keyframe went away
  if (hoverSpan != null && !seen.has(hoverSpan)) hoverSpan = null;
  updateSpanLine(real);
}


// ---------------------------------------------------------------- what the actor shows
/** Resolve what the mannequin shows at the current frame. */
function frameState() {
  const st = syncState();
  const { keys, sigs, ok } = st;
  const f = S.frame;
  const key = keyAt(f);
  if (key) return { ...st, mode: 'key', key };
  if (S.draft && S.draft.dirty) return { ...st, mode: 'draft' };
  for (let i = 0; i + 1 < keys.length; i++) {
    if (keys[i].frame < f && f < keys[i + 1].frame) {
      if (ok[i]) return { ...st, mode: 'gen', seg: S.cache.get(sigs[i]), local: f - keys[i].frame, segIndex: i };
      return { ...st, mode: 'draft', unsynced: S.lastRun.length > 0 };
    }
  }
  return { ...st, mode: 'draft' };
}

/** Draft for an unkeyed frame: where the character is, with the pose of the key before it. */
/**
 * Draft for an unsaved frame. On a generated in-between it sits where the motion is.
 * Between two keyframes it is a blend of them: hips position lerped, root rotation and
 * every joint slerped, by time (shown as "Blend"). Before the first or after the last
 * keyframe it holds that keyframe.
 */
function resetDraft() {
  const keys = chain().keys; // breakdowns count as poses to blend between
  const here = keys.find((k) => k.bd && k.frame === S.frame);
  if (here && frameStateNoDraft().mode !== 'gen') { // on a breakdown: show it as it is
    S.draft = { pose: here.pose, x: here.x, y: here.y, z: here.z, yaw: here.yaw, pitch: 0, dirty: false, bd: true };
    return;
  }
  const prev = [...keys].reverse().find((k) => k.frame < S.frame);
  const next = keys.find((k) => k.frame > S.frame);
  const near = !prev ? next : !next ? prev : S.frame - prev.frame <= next.frame - S.frame ? prev : next;
  let x = 0, z = 0, yaw = 0, y = 0;
  if (prev && next) {
    const t = (S.frame - prev.frame) / (next.frame - prev.frame);
    x = prev.x + (next.x - prev.x) * t;
    z = prev.z + (next.z - prev.z) * t;
    y = (prev.y || 0) + ((next.y || 0) - (prev.y || 0)) * t;
    yaw = near.yaw; // facing comes with the nearest keyframe's pose, not blended
  } else if (near) ({ x, z, yaw } = near), (y = near.y || 0);
  const fs = frameStateNoDraft();
  if (fs.mode !== 'gen' && prev && next) {
    const ib = inBetweenPose(blendSeg(prev, next, S.frame), CUSTOM_SPAN);
    S.draft = { pose: 'custom', custom: { ...ib.custom, label: 'Blend' }, x: ib.x, y: 0, z: ib.z, yaw: ib.yaw, pitch: 0, dirty: false };
    return;
  }
  if (fs.mode === 'gen') {
    const r = fs.seg.root, k = fs.local * 3;
    x = r[k]; z = r[k + 2];
    actor.setPose(r[k], r[k + 1], r[k + 2], fs.seg.rot, fs.local * kit.names.length * 4);
    yaw = actor.facing();
    y = 0;
  }
  S.draft = { pose: near ? near.pose : 'stand', custom: near?.custom, x, y, z, yaw, pitch: near?.pitch || 0, dirty: false };
}

const CUSTOM_SPAN = 3; // frames kept either side of an in-between pose (its momentum)
const _Y = new THREE.Vector3(0, 1, 0);

const _Xa = new THREE.Vector3(1, 0, 0);

/** A keyframe's pose in world terms: hips position and per-joint rotations (root turned). */
function keyFrameData(k) {
  const src = k.custom ? k.custom.frames[k.custom.anchor] : null;
  const rootY = src ? src.root[1] : S.poses[k.pose].root[1];
  const rots = Float32Array.from(src ? src.rot : S.poses[k.pose].rot);
  const turn = new THREE.Quaternion().setFromAxisAngle(_Y, k.yaw);
  if (k.pitch) turn.multiply(new THREE.Quaternion().setFromAxisAngle(_Xa, k.pitch));
  const q0 = new THREE.Quaternion().fromArray(rots, 0).premultiply(turn);
  q0.toArray(rots, 0);
  return { pos: new THREE.Vector3(k.x, rootY + (k.y || 0), k.z), rots };
}

/**
 * Blend two keyframes into a short run of frames around `frame` (same layout as a
 * generated segment): hips lerped, every joint rotation (root included) slerped.
 */
function blendSeg(a, b, frame, span = CUSTOM_SPAN) {
  const A = keyFrameData(a), B = keyFrameData(b), nJ = kit.names.length;
  const n = 2 * span + 1;
  const root = new Float32Array(n * 3), rot = new Float32Array(n * nJ * 4);
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), q = new THREE.Quaternion(), p = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const t = Math.min(1, Math.max(0, (frame + i - span - a.frame) / (b.frame - a.frame)));
    p.lerpVectors(A.pos, B.pos, t).toArray(root, i * 3);
    for (let j = 0; j < nJ; j++) {
      qa.fromArray(A.rots, j * 4); qb.fromArray(B.rots, j * 4);
      q.slerpQuaternions(qa, qb, t).toArray(rot, (i * nJ + j) * 4);
    }
  }
  return { root, rot, frames: n };
}

/** Copy the generated in-between at `local` into a saved pose of its own. */
function inBetweenPose(seg, local) {
  const nJ = kit.names.length, r = seg.root, n = seg.frames;
  const ax = r[local * 3], az = r[local * 3 + 2];
  actor.setPose(ax, r[local * 3 + 1], az, seg.rot, local * nJ * 4);
  const yaw = actor.facing();
  const inv = new THREE.Quaternion().setFromAxisAngle(_Y, -yaw);
  const frames = [];
  for (let d = -CUSTOM_SPAN; d <= CUSTOM_SPAN; d++) {
    const f = Math.min(n - 1, Math.max(0, local + d));
    const p = new THREE.Vector3(r[f * 3] - ax, r[f * 3 + 1], r[f * 3 + 2] - az).applyQuaternion(inv);
    const rot = Array.from(seg.rot.subarray(f * nJ * 4, (f + 1) * nJ * 4));
    const q = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]).premultiply(inv);
    rot[0] = q.x; rot[1] = q.y; rot[2] = q.z; rot[3] = q.w;
    frames.push({ root: [p.x, p.y, p.z].map((v) => +v.toFixed(4)), rot: rot.map((v) => +v.toFixed(5)) });
  }
  const uid = 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  return { custom: { uid, anchor: CUSTOM_SPAN, frames }, x: ax, z: az, yaw };
}
function frameStateNoDraft() { const d = S.draft; S.draft = null; const fs = frameState(); S.draft = d; return fs; }

/** Pose a mannequin from a saved pose, draft or key: library pose or copied in-between. */
function applyPoseOf(m, t) {
  if (t.custom) {
    const fr = t.custom.frames[t.custom.anchor];
    return m.setPose(t.x, fr.root[1] + (t.y || 0), t.z, fr.rot, 0, t.yaw);
  }
  applyLibraryPose(m, t.pose, t.x, t.z, t.yaw, t.y, t.pitch);
}

function applyLibraryPose(m, poseId, x, z, yaw, y = 0, pitch = 0) {
  const p = S.poses[poseId];
  if (!pitch) return m.setPose(x, p.root[1] + y, z, p.rot, 0, yaw);
  // Flipped: pivot at the hips, then drop the body so its lowest joint is on the floor
  // (the server grounds it the same way, so the model gets what you see).
  m.setPose(x, p.root[1], z, p.rot, 0, yaw, pitch);
  m.setPose(x, p.root[1] + y + (S.floorY - m.rawMinY), z, p.rot, 0, yaw, pitch);
}

/** The thing edits apply to: the key at this frame, or the draft (marked dirty). */
function editable() {
  const key = keyAt(S.frame);
  if (key) return key;
  if (!S.draft) resetDraft();
  S.draft.dirty = true;
  return S.draft;
}

function setFrame(f) {
  f = Math.max(0, Math.min(S.length - 1, Math.round(f)));
  if (f === S.frame && S.draft) return;
  S.frame = f;
  picker.target = null;
  resetDraft();
}

// ---------------------------------------------------------------- key operations
function keyPoseButton() {
  const key = keyAt(S.frame);
  if (key) {
    S.keys = S.keys.filter((k) => k !== key);
    if (selected === key) selected = null;
    resetDraft();
    Object.assign(S.draft, { pose: key.pose, custom: key.custom, x: key.x, y: key.y || 0, z: key.z, yaw: key.yaw, pitch: key.pitch || 0, dirty: true });
    toast(`Pose removed from frame ${S.frame}.`);
  } else {
    if (!S.draft) resetDraft();
    const fs = frameState();
    if (fs.mode === 'gen') { // on a generated in-between: key exactly that pose and position
      const ib = inBetweenPose(fs.seg, fs.local);
      S.keys.push({ id: S.nextId++, frame: S.frame, pose: 'custom', custom: ib.custom, x: ib.x, y: 0, z: ib.z, yaw: ib.yaw, pitch: 0, carry: 'auto' });
    } else {
      const d = S.draft;
      S.keys.push({ id: S.nextId++, frame: S.frame, pose: d.pose, custom: d.custom, x: d.x, y: d.y || 0, z: d.z, yaw: d.yaw, pitch: d.pitch || 0, carry: 'auto' });
    }
    S.draft = null;
    resetDraft();
    if (S.keys.length === 1) toast('Keyframe added! Now pick a later moment on the timeline and set the next pose.');
    else if (S.keys.length === 2 && !S.lastRun.length) toast('Nice — two keyframes added. Press AI In-betweens to fill in the motion between them.');
  }
  save();
}

// While scrolling, the pose label turns into the full pose list beside the body.
const picker = { target: null, mannequin: null, until: 0, hold: false, html: '' };
const PICKER_MS = 1600;

function setPose(target, id) {
  target.pose = id;
  delete target.custom; // a library pose replaces a copied in-between
  if (target.id) save(); else target.dirty = true;
  picker.until = performance.now() + PICKER_MS;
}

function cyclePose(target, dir, mannequin = actor) {
  const i = S.order.indexOf(target.pose);
  Object.assign(picker, { target, mannequin });
  setPose(target, S.order[(i + dir + S.order.length) % S.order.length]);
}

// ---------------------------------------------------------------- generation
async function generate() {
  if (S.generating) return;
  const { keys, sigs, ok } = syncState();
  if (keys.length < 2) return toast('Add at least two keyframes at different moments, then press AI In-betweens.');
  const first = ok.indexOf(false);
  if (first < 0) { S.lastRun = sigs; return toast("Everything's already generated. Press ▶ to watch it."); }
  S.generating = { done: 0, total: sigs.length - first };
  const payload = (k) => ({ pose: k.pose, x: k.x, y: k.y || 0, z: k.z, yaw: k.yaw, pitch: k.pitch || 0, frame: k.frame, posesToUse: posesToUse(k),
    custom: k.custom ? { anchor: k.custom.anchor, frames: k.custom.frames } : undefined });
  const started = performance.now();
  let ms = 0;
  try {
    for (let i = first; i < sigs.length; i++) {
      const ctx = i === 0 ? null : S.cache.get(sigs[i - 1]).ctx;
      const res = await fetch('/api/segment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from: payload(keys[i]), to: payload(keys[i + 1]), ctx }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.statusText);
      S.cache.set(sigs[i], { ...j, root: f32(j.root), rot: f32(j.rot) });
      ms += j.timings.inference_ms;
      S.generating.done++;
    }
    S.lastRun = sigs;
    toast(`Generated ${sigs.length - first} part${sigs.length - first > 1 ? 's' : ''} in ${Math.round(performance.now() - started)} ms. Playing it back now.`);
    if (S.cache.size > 400) S.cache = new Map([...S.cache].slice(-200));
    setFrame(keys[first].frame);
    S.playing = true;
  } catch (e) {
    toast("Couldn't generate that: " + e.message, true);
  } finally {
    S.generating = null;
  }
}

/** Throw away the generated motion. Keyframes stay; AI In-betweens rebuilds it. */
function clearGenerated() {
  if (S.generating || !S.cache.size) return;
  S.cache = new Map();
  S.lastRun = [];
  S.playing = false;
  resetDraft();
  toast('Generated motion cleared. Your saved poses are untouched.');
}

/** The active mannequin and the pose it holds (null pose = a generated in-between). */
function activeOf(fs) {
  if (selected === 'actor' && S.show.actor) return { m: actor, t: fs.mode === 'key' ? fs.key : fs.mode === 'gen' ? null : S.draft };
  if (selected && selected !== 'actor' && S.keys.includes(selected) && S.show.keys) {
    const g = keyGhosts.get(selected.id);
    return { m: g && g.visible ? g : actor, t: selected }; // hidden ghost: the mannequin stands in it
  }
  return null;
}
const OUTLINE = '#f2c230';

// ---------------------------------------------------------------- pose mode
// "Custom…" in the pose menu: the white mannequin's armature becomes draggable joints.
//   Hands / feet ........ exact two-bone IK (elbows bend back, knees forward, no stretch)
//   Elbows / knees ...... swing the upper arm / thigh toward the cursor
//   Head / chest ........ lean, spread over the spine and neck, capped so it can't fold
//   Hips ................ move the body with both feet planted (stops at reach and floor)
// Clicking a joint selects it: its gizmo arrows move it along X/Y/Z (same rules) and its
// rings rotate the bone. Nothing may go under the floor. Undo / Redo / Reset in a toolbar.
// Everything else dims to 10% and can't be clicked. Add Keyframe keeps the pose; Cancel
// (or Esc) puts it back.
const LIMB = {
  LeftHand: ['LeftArm', 'LeftForeArm', 'LeftHand', -1], RightHand: ['RightArm', 'RightForeArm', 'RightHand', -1],
  LeftFoot: ['LeftLeg', 'LeftShin', 'LeftFoot', 1], RightFoot: ['RightLeg', 'RightShin', 'RightFoot', 1],
}; // [root, mid, end, natural bend: +1 forward (knee), -1 back (elbow)]
const AIM = { LeftForeArm: 'LeftArm', RightForeArm: 'RightArm', LeftShin: 'LeftLeg', RightShin: 'RightLeg' };
const LEAN = {
  Head: [['Spine1', 0.22], ['Spine2', 0.2], ['Chest', 0.2], ['Neck1', 0.16], ['Neck2', 0.12], ['Head', 0.1]],
  Chest: [['Spine1', 0.4], ['Spine2', 0.33], ['Chest', 0.27]],
};
const LEAN_MAX = THREE.MathUtils.degToRad(70); // total lean per drag
const POSE_NODES = ['Hips', 'Chest', 'Head', 'LeftForeArm', 'LeftHand', 'RightForeArm', 'RightHand', 'LeftShin', 'LeftFoot', 'RightShin', 'RightFoot'];
const NODE_PICK_PX = 16;
const FLOOR_MIN = 0.03; // IK targets stay this far above the floor
const poseNodeMat = new THREE.MeshBasicMaterial({ color: '#f2c230', depthTest: false, transparent: true });
const poseNodeHot = new THREE.MeshBasicMaterial({ color: '#ffffff', depthTest: false, transparent: true });
const poseNodes = new THREE.Group();
const poseNodeMeshes = POSE_NODES.map((name) => {
  const m = new THREE.Mesh(dotGeo, poseNodeMat);
  m.scale.setScalar(name === 'Hips' ? 0.045 : 0.032);
  m.renderOrder = 1001; m.userData.joint = name;
  poseNodes.add(m);
  return m;
});
const poseBoneMat = new LineMaterial({ color: 0xffffff, linewidth: 1.5, depthTest: false, transparent: true, opacity: 0.85 });
const poseBones = new LineSegments2(new LineSegmentsGeometry(), poseBoneMat);
poseNodes.visible = poseBones.visible = false;
poseBones.renderOrder = 1000; poseBones.frustumCulled = false;
scene.add(poseNodes, poseBones);
const keyGhostDim = flat('#ff7a1a', 0.1);
const genGhostDim = new THREE.MeshBasicMaterial({ color: '#4f9dff', transparent: true, opacity: 0.1, depthWrite: true });
let poseDrag = null, poseHover = null, poseSel = null, jgHover = null;

/** Forward kinematics on the model's skeleton: hips position + local rotations -> world. */
function fkSoma(root, rots) {
  const { parents, offsets } = kit.soma;
  const pos = [], quat = [];
  for (let j = 0; j < parents.length; j++) {
    const q = new THREE.Quaternion().fromArray(rots, j * 4), p = parents[j];
    if (p < 0) { pos[j] = root.clone(); quat[j] = q; }
    else { quat[j] = quat[p].clone().multiply(q); pos[j] = offsets[j].clone().applyQuaternion(quat[p]).add(pos[p]); }
  }
  return { pos, quat };
}
const J = (name) => kit.index[name];
const lowestJoint = (pm) => Math.min(...fkSoma(pm.root, pm.rots).pos.map((p) => p.y));

/** Set joint j's world rotation to W (its parent's world rotation `parentW`). */
function setWorldRot(pm, j, W, parentW) {
  parentW.clone().invert().multiply(W).toArray(pm.rots, j * 4);
}

/** The body's forward direction, measured across the hips like the model does. */
function bodyForward(fk) {
  const r = fk.pos[J('RightLeg')], l = fk.pos[J('LeftLeg')];
  return new THREE.Vector3(r.z - l.z, 0, -(r.x - l.x)).normalize();
}

/**
 * Exact two-bone IK: place the hand/foot at `target` (or as close as the limb reaches),
 * bending the elbow/knee to its natural side. Bone lengths never change; the hand/foot
 * keeps its world orientation.
 */
function twoBone(pm, limbName, target) {
  const [rn, mn, en, sign] = LIMB[limbName];
  const ja = J(rn), jb = J(mn), jc = J(en);
  const fk = fkSoma(pm.root, pm.rots);
  const a = fk.pos[ja], b = fk.pos[jb], c = fk.pos[jc];
  const lab = a.distanceTo(b), lbc = b.distanceTo(c);
  const goal = target.clone(); goal.y = Math.max(goal.y, FLOOR_MIN);
  const toGoal = goal.clone().sub(a);
  const d = THREE.MathUtils.clamp(toGoal.length(), Math.abs(lab - lbc) + 1e-3, lab + lbc - 1e-4);
  const axis = toGoal.lengthSq() > 1e-10 ? toGoal.normalize() : c.clone().sub(a).normalize();
  // bend direction: keep the current bend if it's on the natural side, else the natural one
  const natural = bodyForward(fk).multiplyScalar(sign);
  if (sign < 0) natural.add(new THREE.Vector3(0, -0.35, 0)); // elbows: back and a little down
  const perp = (v) => v.clone().addScaledVector(axis, -v.dot(axis));
  let bend = perp(b.clone().sub(a.clone().addScaledVector(c.clone().sub(a).normalize(), b.clone().sub(a).dot(c.clone().sub(a).normalize()))));
  const nat = perp(natural);
  if (bend.lengthSq() < 1e-6 || bend.dot(nat) <= 0) bend = nat;
  if (bend.lengthSq() < 1e-10) bend = perp(new THREE.Vector3(0, 0, 1));
  bend.normalize();
  const cosA = THREE.MathUtils.clamp((lab * lab + d * d - lbc * lbc) / (2 * lab * d), -1, 1);
  const mid = a.clone().addScaledVector(axis, lab * cosA).addScaledVector(bend, lab * Math.sqrt(1 - cosA * cosA));
  const end = a.clone().addScaledVector(axis, d);
  // positions -> rotations (the end keeps its world rotation)
  const swing = (from, to) => new THREE.Quaternion().setFromUnitVectors(from.clone().normalize(), to.clone().normalize());
  const Wa = swing(b.clone().sub(a), mid.clone().sub(a)).multiply(fk.quat[ja]);
  const pa = kit.soma.parents[ja];
  setWorldRot(pm, ja, Wa, fk.quat[pa]);
  const fk2 = fkSoma(pm.root, pm.rots);
  const Wb = swing(fk2.pos[jc].clone().sub(fk2.pos[jb]), end.clone().sub(mid)).multiply(fk2.quat[jb]);
  setWorldRot(pm, jb, Wb, fk2.quat[ja]);
  setWorldRot(pm, jc, fk.quat[jc], Wb); // hand/foot keeps its orientation
}

/** Swing a bone (e.g. the upper arm) so its child joint (the elbow) heads toward `target`. */
function aimBone(pm, childName, target) {
  const jb = J(AIM[childName]), jc = J(childName), fk = fkSoma(pm.root, pm.rots);
  const from = fk.pos[jc].clone().sub(fk.pos[jb]), to = target.clone().sub(fk.pos[jb]);
  if (to.lengthSq() < 1e-8) return;
  const W = new THREE.Quaternion().setFromUnitVectors(from.normalize(), to.normalize()).multiply(fk.quat[jb]);
  setWorldRot(pm, jb, W, fk.quat[kit.soma.parents[jb]]);
}

/** Lean the spine so the head/chest heads toward `target`, spread over several joints. */
function spineLean(pm, nodeName, target) {
  const fk = fkSoma(pm.root, pm.rots), pivot = fk.pos[J('Spine1')];
  const from = fk.pos[J(nodeName)].clone().sub(pivot), to = target.clone().sub(pivot);
  if (to.lengthSq() < 1e-8) return;
  const total = new THREE.Quaternion().setFromUnitVectors(from.normalize(), to.normalize());
  const ang = 2 * Math.acos(Math.min(1, Math.abs(total.w)));
  if (ang > LEAN_MAX) total.slerp(new THREE.Quaternion(), 1 - LEAN_MAX / ang);
  const id = new THREE.Quaternion();
  for (const [name, w] of LEAN[nodeName]) {
    const f = fkSoma(pm.root, pm.rots), j = J(name);
    const part = id.clone().slerp(total, w / LEAN[nodeName].reduce((s, [, x]) => s + x, 0) * LEAN[nodeName].length / LEAN[nodeName].length);
    setWorldRot(pm, j, part.multiply(f.quat[j]), f.quat[kit.soma.parents[j]]);
  }
}

/** Re-plant both feet at `feet` (after the hips moved or turned). */
function plantFeet(pm, feet) { twoBone(pm, 'LeftFoot', feet[0]); twoBone(pm, 'RightFoot', feet[1]); }

/** Are both planted feet still within reach of their hips? */
function feetReachable(pm, feet) {
  const fk = fkSoma(pm.root, pm.rots);
  return ['LeftFoot', 'RightFoot'].every((limb, i) => {
    const [rn, mn, en] = LIMB[limb];
    const reach = fk.pos[J(rn)].distanceTo(fk.pos[J(mn)]) + fk.pos[J(mn)].distanceTo(fk.pos[J(en)]);
    return fk.pos[J(rn)].distanceTo(feet[i]) <= reach - 1e-3;
  });
}

/**
 * Apply one drag step at fraction k (0 = drag start, 1 = where the cursor is). If that
 * would break the pose (a joint under the floor, or the hips pulled off planted feet),
 * back off to the largest fraction that is still valid.
 */
function applySafely(d, step) {
  const pm = S.poseMode;
  const valid = (k) => {
    pm.rots.set(d.startRots); pm.root.copy(d.root0);
    step(k);
    if (d.planted && !feetReachable(pm, d.feet)) return false;
    return lowestJoint(pm) >= Math.min(-0.005, d.floor0 - 1e-4);
  };
  if (valid(1)) return;
  let lo = 0, hi = 1;
  for (let i = 0; i < 14; i++) { const m = (lo + hi) / 2; if (valid(m)) lo = m; else hi = m; }
  valid(lo);
}

// ---- history (undo / redo / reset)
function poseSnapshot() { const pm = S.poseMode; return { root: pm.root.clone(), rots: Float32Array.from(pm.rots) }; }
function poseRestore(snap) { const pm = S.poseMode; pm.root.copy(snap.root); pm.rots.set(snap.rots); }
function pushUndo() { const pm = S.poseMode; pm.undo.push(poseSnapshot()); if (pm.undo.length > 100) pm.undo.shift(); pm.redo.length = 0; }
function poseUndo() { const pm = S.poseMode; if (!pm?.undo.length) return; pm.redo.push(poseSnapshot()); poseRestore(pm.undo.pop()); }
function poseRedo() { const pm = S.poseMode; if (!pm?.redo.length) return; pm.undo.push(poseSnapshot()); poseRestore(pm.redo.pop()); }
function poseReset() { const pm = S.poseMode; if (!pm) return; pushUndo(); poseRestore(pm.start); }

/** Enter pose mode on the white mannequin, starting from what it shows (or from `target`). */
function enterPoseMode(target) {
  if (target && target.id && target.frame !== S.frame) { S.frame = target.frame; picker.target = null; resetDraft(); }
  S.playing = false; selected = 'actor'; picker.target = null; picker.until = 0; picker.hold = false;
  const fs = frameState();
  let root, rots;
  if (fs.mode === 'gen') {
    const r = fs.seg.root, l = fs.local, nJ = kit.names.length;
    root = new THREE.Vector3(r[l * 3], r[l * 3 + 1], r[l * 3 + 2]);
    rots = Float32Array.from(fs.seg.rot.subarray(l * nJ * 4, (l + 1) * nJ * 4));
  } else {
    if (!S.draft) resetDraft();
    const d = keyFrameData(fs.mode === 'key' ? fs.key : S.draft);
    root = d.pos; rots = d.rots;
  }
  S.poseMode = { root, rots, frame: S.frame, undo: [], redo: [], start: { root: root.clone(), rots: Float32Array.from(rots) } };
  poseSel = null;
  document.body.classList.add('pose-mode');
  frameCameraOn(root);
  toast('Pose mode: drag the yellow joints, or click one for its gizmo. Add Keyframe keeps the pose, Cancel puts it back.');
}

/**
 * Glide the camera to frame the posed mannequin: same viewing direction as now, a gentle
 * elevation, 2.8 m away. Orbit/pan/zoom work normally once it lands (or right away).
 */
let camTween = null;
function frameCameraOn(hips) {
  const toTarget = new THREE.Vector3(hips.x, Math.max(0.55, hips.y - 0.05), hips.z);
  const dir = camera.position.clone().sub(controls.target);
  const flatLen = Math.hypot(dir.x, dir.z) || 1;
  const azimuth = Math.atan2(dir.x / flatLen, dir.z / flatLen);
  const elev = Math.min(Math.max(Math.atan2(dir.y, flatLen), THREE.MathUtils.degToRad(8)), THREE.MathUtils.degToRad(30));
  const toPos = toTarget.clone().add(new THREE.Vector3(Math.sin(azimuth) * Math.cos(elev), Math.sin(elev), Math.cos(azimuth) * Math.cos(elev)).multiplyScalar(2.8));
  camTween = { fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos, toTarget, t0: performance.now(), dur: 550 };
}
/**
 * Glide the camera to frame a span: its middle (where the settings card sits) centred, the
 * whole path in view, keeping the viewing angle.
 */
function focusSpan(a, b) {
  const A = keyFrameData(a).pos, B = keyFrameData(b).pos;
  const mid = new THREE.Vector3().lerpVectors(A, B, 0.5);
  const dir = camera.position.clone().sub(controls.target);
  const hfov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * camera.aspect);
  const need = (A.distanceTo(B) / 2 + 1) / Math.tan(hfov / 2); // fit the span with a margin
  const dist = Math.min(30, Math.max(5, need * 1.15, Math.min(dir.length(), 10)));
  // Aim a little below the card so it sits above the dock.
  const toTarget = new THREE.Vector3(mid.x, 0.3, mid.z);
  const toPos = toTarget.clone().add(dir.normalize().multiplyScalar(dist));
  camTween = { fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos, toTarget, t0: performance.now(), dur: 550 };
}
function stepCamTween(now) {
  if (!camTween) return;
  const k = Math.min(1, (now - camTween.t0) / camTween.dur), e = k * k * (3 - 2 * k); // smoothstep
  camera.position.lerpVectors(camTween.fromPos, camTween.toPos, e);
  controls.target.lerpVectors(camTween.fromTarget, camTween.toTarget, e);
  if (k >= 1) camTween = null;
}
// ---------------------------------------------------------------- view cube
// Snapping to a face, edge or corner glides round the target and lands in an orthographic
// view; orbiting out of it (in the viewport or on the cube) goes back to perspective.
const VIEW_EPS = 1e-4;  // keeps straight up/down views off the controls' pole
let ortho = null;       // { dir, ijk } while in an orthographic view
let viewTween = null;
const viewCube = new ViewCube($('view-cube-host'), {
  onSnap: (dir, ijk) => snapView(dir, ijk),
  onOrbit: (dx, dy) => orbitBy(dx, dy),
});
function sphericalOf(dir, ijk) {
  const s = new THREE.Spherical().setFromVector3(dir);
  if (ijk && !ijk.x && !ijk.z) s.theta = 0; // top/bottom: Back at the top of the screen
  s.phi = THREE.MathUtils.clamp(s.phi, VIEW_EPS, Math.PI - VIEW_EPS);
  return s;
}
function snapView(dir, ijk) {
  camTween = null;
  if (controls._sphericalDelta) controls._sphericalDelta.set(0, 0, 0); // drop leftover orbit drift
  if (controls._panOffset) controls._panOffset.set(0, 0, 0);
  controls.maxPolarAngle = Math.PI;
  const off = camera.position.clone().sub(controls.target);
  const from = new THREE.Spherical().setFromVector3(off), to = sphericalOf(dir, ijk);
  let dTheta = wrapAngle(to.theta - from.theta);
  viewTween = { from, dTheta, dPhi: to.phi - from.phi, r: from.radius, ijk, t0: performance.now(), dur: 420 };
}
function stepViewTween(now) {
  if (!viewTween) return;
  const t = viewTween, k = Math.min(1, (now - t.t0) / t.dur), e = k * k * (3 - 2 * k);
  const s = new THREE.Spherical(t.r, t.from.phi + t.dPhi * e, t.from.theta + t.dTheta * e);
  camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(s));
  if (ortho) ortho.dir = new THREE.Vector3().setFromSpherical(s).normalize(); // stay flat between flat views
  if (k >= 1) {
    viewTween = null;
    ortho = { dir: new THREE.Vector3().setFromSpherical(s).normalize(), ijk: t.ijk };
  }
}
function orbitBy(dx, dy) {
  viewTween = null; camTween = null;
  const off = camera.position.clone().sub(controls.target);
  const s = new THREE.Spherical().setFromVector3(off);
  s.theta -= dx * 0.012;
  s.phi = THREE.MathUtils.clamp(s.phi - dy * 0.012, VIEW_EPS, Math.PI - VIEW_EPS);
  camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(s));
}
function toggleProjection() {
  if (ortho) { ortho = null; viewTween = null; controls.maxPolarAngle = Math.max(MAX_POLAR, new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target)).phi); return; }
  ortho = { dir: camera.position.clone().sub(controls.target).normalize(), ijk: null };
}
/** After the controls update: leave ortho if the view was orbited, then pick the camera to draw with. */
function syncViewCamera() {
  const off = camera.position.clone().sub(controls.target), dist = off.length();
  if (ortho && !viewTween && off.clone().divideScalar(dist).angleTo(ortho.dir) > 1e-3) {
    ortho = null; // maxPolarAngle eases back below
  }
  if (!ortho && !viewTween && controls.maxPolarAngle !== MAX_POLAR) {
    // A cube drag or a persp toggle may leave us below the usual limit; ease back over the next orbit.
    const phi = new THREE.Spherical().setFromVector3(off).phi;
    controls.maxPolarAngle = Math.max(MAX_POLAR, Math.min(controls.maxPolarAngle, phi));
  }
  if (ortho) {
    const h = dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), back = Math.max(dist, 60);
    Object.assign(orthoCam, { left: -h * camera.aspect, right: h * camera.aspect, top: h, bottom: -h, near: 0.05, far: back + 300 });
    orthoCam.position.copy(controls.target).addScaledVector(off, back / dist);
    orthoCam.quaternion.copy(camera.quaternion);
    orthoCam.updateProjectionMatrix();
    orthoCam.updateMatrixWorld();
    rcam = orthoCam;
  } else rcam = camera;
  scene.fog = ortho ? null : FOG; // the ortho camera sits far back; fog would grey everything
  const name = ortho && ortho.ijk ? viewName(ortho.ijk) : '';
  if (vcName.textContent !== name) vcName.textContent = name;
  const mode = ortho ? 'ortho' : 'persp';
  if (vcProj.dataset.mode !== mode) {
    vcProj.dataset.mode = mode;
    vcProj.querySelector('.ic').innerHTML = icon(ortho ? 'view-ortho' : 'view-persp');
    vcProj.title = ortho ? 'Orthographic · click for perspective' : 'Perspective · click for orthographic';
    vcProj.setAttribute('aria-label', ortho ? 'Switch to perspective' : 'Switch to orthographic');
    vcProj.classList.toggle('on', !!ortho);
  }
}
const vcName = $('vc-name'), vcProj = $('vc-proj');
vcProj.addEventListener('click', toggleProjection);

// Any camera input during the glide hands control straight back to you.
renderer.domElement.addEventListener('pointerdown', () => { if (!poseDrag) { camTween = null; viewTween = null; } }, true);
renderer.domElement.addEventListener('wheel', () => { camTween = null; }, { capture: true, passive: true });

function exitPoseMode() {
  S.poseMode = null; poseDrag = null; poseHover = null; poseSel = null;
  document.body.classList.remove('pose-mode');
}

/** Add Keyframe in pose mode: keep the pose as a keyframe here (replacing one already here). */
function commitPose() {
  const pm = S.poseMode, nJ = kit.names.length, n = 2 * CUSTOM_SPAN + 1;
  const seg = { root: new Float32Array(n * 3), rot: new Float32Array(n * nJ * 4), frames: n };
  for (let i = 0; i < n; i++) { pm.root.toArray(seg.root, i * 3); seg.rot.set(pm.rots, i * nJ * 4); }
  const ib = inBetweenPose(seg, CUSTOM_SPAN);
  const data = { pose: 'custom', custom: { ...ib.custom, label: 'Custom' }, x: ib.x, y: 0, z: ib.z, yaw: ib.yaw, pitch: 0, carry: 'stop' };
  const existing = keyAt(pm.frame);
  if (existing) Object.assign(existing, data);
  else S.keys.push({ id: S.nextId++, frame: pm.frame, ...data });
  exitPoseMode();
  S.draft = null; resetDraft(); save();
  toast(`Custom pose keyed at frame ${pm.frame}.`);
}

function cancelPose() {
  exitPoseMode();
  S.draft = null; resetDraft();
  toast('Pose put back.');
}

/** Node positions on screen; the mesh may be lifted off the floor, so nodes follow it. */
function poseNodePositions() {
  const fk = fkSoma(S.poseMode.root, S.poseMode.rots), lift = actor.lift || 0;
  return { fk, lift, at: (name) => fk.pos[J(name)].clone().setY(fk.pos[J(name)].y + lift) };
}

function pickPoseNode(e) {
  const { at } = poseNodePositions(), r = canvas.getBoundingClientRect();
  let best = null, bd = NODE_PICK_PX;
  for (const name of POSE_NODES) {
    const p = at(name).project(rcam);
    if (p.z > 1) continue;
    const d = Math.hypot((p.x * 0.5 + 0.5) * r.width + r.left - e.clientX, (-p.y * 0.5 + 0.5) * r.height + r.top - e.clientY);
    if (d < bd) { bd = d; best = name; }
  }
  return best;
}

// Gizmo on the selected joint: arrows move it along X/Y/Z, rings rotate it about X/Y/Z.
const JG_AXES = [[new THREE.Vector3(1, 0, 0), '#ff5a5a'], [new THREE.Vector3(0, 1, 0), '#5ad16a'], [new THREE.Vector3(0, 0, 1), '#4f9dff']];
const JG_ARROW = { start: 0.05, len: 0.15, head: 0.05 }, JG_RING = 0.27;
const jointGizmo = new THREE.Group();
const jgPick = [];
jointGizmo.visible = false;
for (const [axis, color] of JG_AXES) {
  const m = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
  const arrow = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, JG_ARROW.len, 8), m);
  shaft.position.y = JG_ARROW.start + JG_ARROW.len / 2;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.02, JG_ARROW.head, 14), m);
  head.position.y = JG_ARROW.start + JG_ARROW.len + JG_ARROW.head / 2;
  const hit = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, JG_ARROW.len + JG_ARROW.head, 8), new THREE.MeshBasicMaterial({ visible: false }));
  hit.position.y = JG_ARROW.start + (JG_ARROW.len + JG_ARROW.head) / 2;
  hit.userData.handle = { type: 'jaxis', axis, mat: m, color };
  arrow.add(shaft, head, hit);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  const rm = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(JG_RING, 0.006, 6, 72), rm);
  const rhit = new THREE.Mesh(new THREE.TorusGeometry(JG_RING, 0.028, 6, 48), new THREE.MeshBasicMaterial({ visible: false }));
  for (const o of [ring, rhit]) o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis); // ring around its axis
  rhit.userData.handle = { type: 'jring', axis, mat: rm, color };
  jointGizmo.add(arrow, ring, rhit);
  jgPick.push(hit, rhit);
}
jointGizmo.traverse((o) => { o.renderOrder = 1002; });
scene.add(jointGizmo);

/** Move one joint toward `target` by its rule (IK, swing, lean or body move). */
function moveJoint(pm, d, target) {
  if (d.name === 'Hips') { pm.root.copy(d.root0).add(target.clone().sub(d.start)); plantFeet(pm, d.feet); return; }
  if (LIMB[d.name]) return twoBone(pm, d.name, target);
  if (AIM[d.name]) return aimBone(pm, d.name, target);
  if (LEAN[d.name]) return spineLean(pm, d.name, target);
}

function poseDown(e) {
  setPointer(e);
  const pm = S.poseMode;
  // a joint node under the cursor wins (free drag); otherwise the selected joint's gizmo
  let handle = null, name = pickPoseNode(e);
  if (!name && jointGizmo.visible) {
    const h = raycaster.intersectObjects(jgPick, false)[0];
    if (h) { handle = h.object.userData.handle; name = poseSel; }
  }
  if (!name) return; // empty space: orbit the camera
  poseSel = name;
  controls.enabled = false;
  canvas.setPointerCapture(e.pointerId);
  const { fk, lift, at } = poseNodePositions();
  const hit = at(name), j = J(name);
  const camDir = camera.getWorldDirection(new THREE.Vector3());
  let plane;
  if (handle && handle.type === 'jaxis') { // a plane that holds the axis and faces the camera
    const n = handle.axis.clone().cross(camDir).cross(handle.axis).normalize();
    plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n.lengthSq() > 1e-6 ? n : camDir, hit);
  } else if (handle && handle.type === 'jring') {
    plane = new THREE.Plane().setFromNormalAndCoplanarPoint(handle.axis, hit);
  } else plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camDir, hit);
  const p0 = raycaster.ray.intersectPlane(plane, new THREE.Vector3()) || hit.clone();
  const p = kit.soma.parents[j];
  pushUndo();
  poseDrag = { name, handle, plane, lift, hit, p0, root0: pm.root.clone(), startRots: Float32Array.from(pm.rots),
    floor0: lowestJoint(pm), start: fk.pos[j].clone(), W0: fk.quat[j].clone(), parentW: p < 0 ? new THREE.Quaternion() : fk.quat[p].clone(),
    feet: ['LeftFoot', 'RightFoot'].map((f) => fk.pos[J(f)].clone()), planted: name === 'Hips', moved: false };
}

function poseMove(e) {
  setPointer(e);
  if (!poseDrag) {
    poseHover = pickPoseNode(e);
    jgHover = null;
    if (!poseHover && jointGizmo.visible) { const h = raycaster.intersectObjects(jgPick, false)[0]; jgHover = h ? h.object.userData.handle : null; }
    canvas.style.cursor = poseHover || jgHover ? 'grab' : '';
    return;
  }
  const d = poseDrag, pm = S.poseMode;
  const q = raycaster.ray.intersectPlane(d.plane, new THREE.Vector3());
  if (!q) return;
  d.moved = true;
  canvas.style.cursor = 'grabbing';
  if (d.handle && d.handle.type === 'jring') { // rotate the joint's bone about the axis
    const v0 = d.p0.clone().sub(d.hit), v1 = q.clone().sub(d.hit);
    let ang = Math.atan2(v0.clone().cross(v1).dot(d.handle.axis), v0.dot(v1));
    if (e.shiftKey) ang = Math.round(ang / (Math.PI / 12)) * (Math.PI / 12);
    applySafely(d, (k) => {
      setWorldRot(pm, J(d.name), new THREE.Quaternion().setFromAxisAngle(d.handle.axis, ang * k).multiply(d.W0), d.parentW);
      if (d.name === 'Hips') plantFeet(pm, d.feet);
    });
    return;
  }
  // move: free in the view plane, or along one axis
  let delta = q.clone().sub(d.p0);
  if (d.handle && d.handle.type === 'jaxis') delta = d.handle.axis.clone().multiplyScalar(delta.dot(d.handle.axis));
  applySafely(d, (k) => moveJoint(pm, d, d.start.clone().addScaledVector(delta, k)));
}

function poseUp() {
  controls.enabled = true;
  if (poseDrag && !poseDrag.moved) S.poseMode.undo.pop(); // a plain click changed nothing
  poseDrag = null;
  canvas.style.cursor = '';
}

function updatePoseNodes() {
  const on = !!S.poseMode;
  poseNodes.visible = poseBones.visible = on;
  jointGizmo.visible = on && !!poseSel;
  $('pose-tools').classList.toggle('hidden', !on);
  if (!on) return;
  $('pose-undo').disabled = !S.poseMode.undo.length;
  $('pose-redo').disabled = !S.poseMode.redo.length;
  const { fk, lift, at } = poseNodePositions();
  poseNodeMeshes.forEach((m) => {
    const name = m.userData.joint;
    m.position.copy(at(name));
    m.material = name === poseSel || (poseDrag && poseDrag.name === name) || poseHover === name ? poseNodeHot : poseNodeMat;
  });
  if (poseSel) {
    jointGizmo.position.copy(at(poseSel));
    for (const o of jgPick) { const h = o.userData.handle; h.mat.color.set((poseDrag && poseDrag.handle === h) || jgHover === h ? '#ffffff' : h.color); }
  }
  const seg = [], { parents } = kit.soma, skip = new Set(['Jaw', 'LeftEye', 'RightEye']);
  for (let j = 1; j < parents.length; j++) {
    if (skip.has(kit.names[j])) continue;
    const a = fk.pos[parents[j]], b = fk.pos[j];
    seg.push(a.x, a.y + lift, a.z, b.x, b.y + lift, b.z);
  }
  if (poseBones.geometry.attributes.instanceStart?.count !== seg.length / 6) setLine(poseBones, seg);
  else poseBones.geometry.setPositions(seg);
}

// ---------------------------------------------------------------- picking + dragging
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let hover = null;        // {type:'actor'|'ghost'|'handle', ...}
let drag = null;
let selected = 'actor';  // the active mannequin: 'actor', a saved key (its orange ghost), or null

function setPointer(e) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, rcam);
}
function groundPoint() { const p = new THREE.Vector3(); return raycaster.ray.intersectPlane(groundPlane, p) ? p : null; }

function pick() {
  if (gizmo.visible) {
    const h = raycaster.intersectObjects(gizmoPick, false)[0];
    if (h) return { type: 'handle', handle: h.object.userData.handle };
  }
  const live = spanHandles.filter((sh) => sh.group.visible).map((sh) => sh.hit);
  const sh = live.length && raycaster.intersectObjects(live, false)[0];
  if (sh) return { type: 'span', handle: sh.object.userData.span };
  const ray = raycaster.ray;
  let best = { d: actor.hitTest(ray), hit: { type: 'actor' } };
  const point = () => ray.at(best.d, new THREE.Vector3());
  for (const k of S.keys) {
    const g = keyGhosts.get(k.id);
    if (!g) continue;
    const d = g.hitTest(ray);
    if (d < best.d) best = { d, hit: { type: 'ghost', key: k } };
  }
  const ph = propMeshes.size && raycaster.intersectObjects([...propMeshes.values()], false)[0];
  if (ph && ph.distance < best.d) return { type: 'prop', id: ph.object.userData.id, point: ph.point };
  if (best.d === Infinity) { // nothing solid under the cursor: maybe a path between keyframes
    const span = pickSpanPath();
    return span ? { type: 'path', key: span } : null;
  }
  best.hit.point = point(); // where on the body the ray landed
  return best.hit;
}

const canvas = renderer.domElement;
canvas.addEventListener('pointermove', (e) => {
  if (S.poseMode) return poseMove(e);
  setPointer(e);
  if (drag) return onDrag(e);
  if (tc.dragging || (tc.object && tc.axis !== null)) { hover = null; hoverSpan = null; return; } // the shape gizmo has it
  hover = pick();
  hoverSpan = hover && hover.type === 'path' ? hover.key.id : null;
  canvas.style.cursor = hover ? (hover.type === 'span' ? 'ns-resize' : hover.type === 'handle' ? 'grab' : 'pointer') : '';
});

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  if (S.poseMode) return poseDown(e);
  picker.hold = false; picker.until = 0; // clicking the scene closes the pose list
  if (tc.object && tc.axis !== null) return; // a shape gizmo handle: the gizmo drags it
  setPointer(e);
  const h = pick();
  if (h && h.type === 'prop') { selectProp(h.id); selected = null; S.spanSel = null; return; }
  if (propSel) selectProp(null);
  if (!h) { selected = null; S.spanSel = null; return; }
  if (h.type === 'path') { S.spanSel = h.key.id; selected = null; chipOpen = null; return; } // show that span's settings
  if (h.type === 'ghost' || h.type === 'actor') S.spanSel = null;
  controls.enabled = false;
  canvas.setPointerCapture(e.pointerId);
  // Bodies are dragged on a level plane at the height where they were grabbed, so the
  // body follows the cursor 1:1 whether you grab the feet or the head.
  const grab = h.point ? new THREE.Plane(new THREE.Vector3(0, 1, 0), -h.point.y) : groundPlane;
  const p0 = h.point ? h.point.clone() : groundPoint();
  if (h.type === 'span') { // jump height: drag on an upright plane facing the camera
    const sh = h.handle, n = camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, sh.group.position);
    drag = { type: 'height', key: sh.key, plane, p0: raycaster.ray.intersectPlane(plane, new THREE.Vector3()), h0: spanOf(sh.key).height };
    canvas.style.cursor = 'ns-resize';
    return;
  }
  // Saved poses are edited where they stand: no jump to their frame.
  if (h.type === 'ghost') { drag = { type: 'click-ghost', start: [e.clientX, e.clientY], key: h.key, p0, grab }; return; }
  if (h.type === 'handle') {
    const t = selected && selected !== 'actor' ? selected : editable();
    const vertical = h.handle.type === 'axis' && h.handle.axis.y > 0.5;
    let plane = null, q0 = p0;
    if (vertical) {
      const n = camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
      plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, new THREE.Vector3(t.x, t.y || 0, t.z));
      q0 = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    }
    drag = { type: vertical ? 'lift' : h.handle.type, handle: h.handle, target: t, p0: q0, plane, x0: t.x, y0: t.y || 0, z0: t.z, yaw0: t.yaw,
      sx: e.clientX, sy: e.clientY };
    if (h.handle.type === 'ring') startRingDrag(drag, e);
    return;
  }
  drag = { type: 'press', start: [e.clientX, e.clientY], p0, grab };
});

/**
 * Turn ring: seen from above, it works like a dial (the point you grabbed follows the
 * cursor around the ring). Seen nearly edge-on, where a dial can't be read, dragging
 * left / right turns it so the near side of the ring moves with the cursor. Up / down
 * motion alone never turns it.
 */
const DIAL_MIN_ELEV = THREE.MathUtils.degToRad(14);
function ringAngle(d) {
  const q = raycaster.ray.intersectPlane(d.ringPlane, new THREE.Vector3());
  if (!q) return null;
  const dx = q.x - d.cx, dz = q.z - d.cz;
  if (Math.hypot(dx, dz) < RING_R * 0.15) return null;
  return Math.atan2(dx, dz); // same convention as yaw (0 = +Z)
}
function startRingDrag(d, e) {
  const t = d.target;
  d.cx = t.x; d.cz = t.z;
  d.ringPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -((t.y || 0) + 0.015));
  const view = camera.getWorldDirection(new THREE.Vector3());
  d.dial = Math.asin(Math.min(1, Math.abs(view.y))) > DIAL_MIN_ELEV;
  if (d.dial) {
    d.prevA = ringAngle(d);
    d.acc = 0;
    if (d.prevA === null) d.dial = false;
  }
  if (!d.dial) {
    // Which way the near side of the ring moves on screen for a positive turn.
    const toCam = camera.position.clone().sub(new THREE.Vector3(t.x, 0, t.z)).setY(0).normalize();
    const vel = new THREE.Vector3(toCam.z, 0, -toCam.x); // d/dyaw of the near point
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    d.sign = vel.dot(right) >= 0 ? 1 : -1;
  }
}

/**
 * Collisions with shapes: a mannequin being moved keeps its body box (an AABB round its
 * bones) out of every shape's box. A move that would overlap slides along the shape's side
 * when one direction is free, and stops otherwise; the shape it hit flashes red. Shapes it
 * already overlaps when the drag starts are ignored, so it can always be pulled back out.
 */
const _bodyBox = new THREE.Box3(), _tryBox = new THREE.Box3(), _shapeBox = new THREE.Box3(), _shift = new THREE.Vector3();
const BLOCK_FLASH_MS = 350;
let propBlocked = { id: null, until: 0 };
function mannequinOf(t) {
  const g = t.id && keyGhosts.get(t.id);
  return g && g.visible ? g : actor;
}
function shapeHit(box, ignore) {
  for (const [id, mesh] of propMeshes) {
    if (ignore.has(id)) continue;
    if (_shapeBox.setFromObject(mesh).intersectsBox(box)) return id;
  }
  return null;
}
function moveClear(t, nx, ny, nz) {
  const y0 = t.y || 0;
  if (!propMeshes.size) { t.x = nx; t.y = ny; t.z = nz; return; }
  if (!drag.body) { // the pose doesn't change during a drag: measure it once, where it stands now
    const box = mannequinOf(t).bounds(new THREE.Box3());
    drag.body = { box, at: new THREE.Vector3(t.x, y0, t.z) };
    drag.ignore = new Set();
    for (const [id, mesh] of propMeshes) if (_shapeBox.setFromObject(mesh).intersectsBox(box)) drag.ignore.add(id);
  }
  const { box, at } = drag.body;
  const hitAt = (x, y, z) => shapeHit(_tryBox.copy(box).translate(_shift.set(x - at.x, y - at.y, z - at.z)), drag.ignore);
  let x = nx, y = ny, z = nz;
  const blocker = hitAt(x, y, z);
  if (blocker) {
    // Go as far toward the cursor as fits (a fast drag can jump right into a shape) ...
    let lo = 0, hi = 1;
    for (let i = 0; i < 10; i++) {
      const f = (lo + hi) / 2;
      if (hitAt(lerp(t.x, nx, f), lerp(y0, ny, f), lerp(t.z, nz, f))) hi = f; else lo = f;
    }
    const fx = lerp(t.x, nx, lo), fy = lerp(y0, ny, lo), fz = lerp(t.z, nz, lo);
    // ... then slide along the shape's side with what's left.
    if (!hitAt(nx, fy, fz)) { x = nx; y = fy; z = fz; }
    else if (!hitAt(fx, fy, nz)) { x = fx; y = fy; z = nz; }
    else { x = fx; y = fy; z = fz; }
    propBlocked = { id: blocker, until: performance.now() + BLOCK_FLASH_MS };
  }
  t.x = x; t.y = y; t.z = z;
}

function onDrag(e) {
  if (drag.type === 'height') {
    const q = raycaster.ray.intersectPlane(drag.plane, new THREE.Vector3());
    if (q && drag.p0) {
      const h = Math.max(JUMP_MIN_H, Math.min(JUMP_MAX_H, drag.h0 + q.y - drag.p0.y));
      drag.key.span = { ...spanOf(drag.key), height: Math.round(h * 100) / 100 };
      if (!keyAt(S.frame)) { S.draft = null; resetDraft(); }
    }
    return;
  }
  if (drag.type === 'lift') { // never below the floor
    const q = raycaster.ray.intersectPlane(drag.plane, new THREE.Vector3());
    if (q && drag.p0) { const t = drag.target; moveClear(t, t.x, Math.max(0, Math.min(4, drag.y0 + q.y - drag.p0.y)), t.z); }
    return;
  }
  if (drag.type === 'ring') {
    let turn;
    if (drag.dial) { // the grabbed point on the ring follows the cursor around the centre
      const a = ringAngle(drag);
      if (a === null) return; // cursor right over the centre: hold
      drag.acc += wrapAngle(a - drag.prevA);
      drag.prevA = a;
      turn = drag.acc;
    } else turn = (e.clientX - drag.sx) * TURN_PER_PX * drag.sign; // ring seen edge-on: side to side only
    let yaw = drag.yaw0 + turn;
    if (e.shiftKey) yaw = Math.round(yaw / (Math.PI / 12)) * (Math.PI / 12);
    drag.target.yaw = wrapAngle(yaw);
    return;
  }
  const p = groundPoint();
  if (drag.type === 'press' || drag.type === 'click-ghost') {
    const moved = Math.hypot(e.clientX - drag.start[0], e.clientY - drag.start[1]) > 5;
    if (!moved) return;
    // Dragging the body jumps straight into freeform placement (gizmo hides while dragging).
    const t = drag.type === 'click-ghost' ? drag.key : editable();
    drag = { type: 'free', target: t, p0: drag.p0, grab: drag.grab, x0: t.x, z0: t.z };
    canvas.style.cursor = 'grabbing';
  }
  if (!p || !drag.p0) return;
  const t = drag.target;
  if (drag.type === 'free') {
    const q = raycaster.ray.intersectPlane(drag.grab, new THREE.Vector3());
    if (q) moveClear(t, drag.x0 + q.x - drag.p0.x, t.y || 0, drag.z0 + q.z - drag.p0.z);
  }
  else if (drag.type === 'axis') { const d = p.clone().sub(drag.p0).dot(drag.handle.axis); moveClear(t, drag.x0 + drag.handle.axis.x * d, t.y || 0, drag.z0 + drag.handle.axis.z * d); }
}

canvas.addEventListener('pointerup', (e) => {
  if (S.poseMode) return poseUp(e);
  controls.enabled = true;
  if (!drag) return;
  if (drag.type === 'press') {
    selected = 'actor';
    // A generated in-between has no editable pose; clicking it starts a draft there.
    if (selected && !keyAt(S.frame) && !(S.draft && S.draft.dirty)) editable();
  } else if (drag.type === 'click-ghost') {
    selected = drag.key; // that saved pose becomes active
  }
  if ((drag.target && drag.target.id) || drag.type === 'height') save();
  drag = null;
  canvas.style.cursor = '';
});

let wheelAcc = 0; // poses change by scrolling over the pose menu, not over bodies

// ---------------------------------------------------------------- shapes
// Basic shapes from the Add menu, saved with the scene. Click one to select it: a gizmo
// appears on it, and a small toolbar above it switches between move, rotate and scale
// (W / E / R). Delete removes it.
const SHAPES = {
  cube: { label: 'Cube', icon: 'shape-cube', geo: () => new THREE.BoxGeometry(0.5, 0.5, 0.5), base: 0.25 },
  sphere: { label: 'Sphere', icon: 'shape-sphere', geo: () => new THREE.SphereGeometry(0.25, 40, 20), base: 0.25 },
  cylinder: { label: 'Cylinder', icon: 'shape-cylinder', geo: () => new THREE.CylinderGeometry(0.25, 0.25, 0.5, 40), base: 0.25 },
  cone: { label: 'Cone', icon: 'shape-cone', geo: () => new THREE.ConeGeometry(0.25, 0.5, 40), base: 0.25 },
  torus: { label: 'Torus', icon: 'shape-torus', geo: () => new THREE.TorusGeometry(0.2, 0.07, 18, 48), base: 0.27 },
  plane: { label: 'Plane', icon: 'shape-plane', geo: () => new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), base: 0.002 },
};
const PROP_COLOR = '#8d95a1', PROP_SEL = '#22324c', PROP_HOVER = '#141b26', PROP_BLOCKED = '#6b1416';
const propMeshes = new Map(); // prop id -> mesh
let propSel = null;           // selected prop id
const r4 = (v) => +v.toFixed(4);
const PROP_TOOLS_CLEAR = 112; // px from the gizmo's centre to above its rings

const tc = new TransformControls(camera, canvas); // after the canvas handlers, so they see its hover first
tc.setSize(0.85);
scene.add(tc.getHelper());
tc.addEventListener('dragging-changed', (e) => { controls.enabled = !e.value; });
tc.addEventListener('objectChange', () => {
  const m = tc.object, p = m && S.props.find((q) => q.id === m.userData.id);
  if (!p) return;
  p.p = m.position.toArray().map(r4); p.q = m.quaternion.toArray().map(r4); p.s = m.scale.toArray().map(r4);
});
tc.addEventListener('mouseUp', save);

/** Make meshes match S.props (added, removed, moved by undo or reload). */
function syncProps() {
  const alive = new Set();
  for (const p of S.props) {
    alive.add(p.id);
    let m = propMeshes.get(p.id);
    if (!m) {
      m = new THREE.Mesh(SHAPES[p.type].geo(), new THREE.MeshStandardMaterial({ color: PROP_COLOR, roughness: 0.55, metalness: 0.05 }));
      if (p.type === 'plane') m.material.side = THREE.DoubleSide;
      m.castShadow = m.receiveShadow = true;
      m.userData.id = p.id;
      scene.add(m);
      propMeshes.set(p.id, m);
    }
    if (!(tc.dragging && tc.object === m)) { m.position.fromArray(p.p); m.quaternion.fromArray(p.q); m.scale.fromArray(p.s); }
    const blocked = propBlocked.id === p.id && performance.now() < propBlocked.until;
    m.material.emissive.set(blocked ? PROP_BLOCKED : p.id === propSel ? PROP_SEL : hover && hover.type === 'prop' && hover.id === p.id ? PROP_HOVER : '#000000');
  }
  for (const [id, m] of propMeshes) {
    if (alive.has(id)) continue;
    if (tc.object === m) tc.detach();
    scene.remove(m); m.geometry.dispose(); m.material.dispose();
    propMeshes.delete(id);
  }
  if (propSel && !alive.has(propSel)) selectProp(null);
}

function selectProp(id) {
  propSel = id;
  if (id) { selected = null; S.spanSel = null; } // one gizmo at a time
  syncProps();
  const m = id && propMeshes.get(id);
  if (m) tc.attach(m); else tc.detach();
  updatePropTools();
}

function setPropMode(mode) { tc.setMode(mode); updatePropTools(); }

function addProp(type, at = controls.target.clone().setY(0)) {
  // On the floor where you right-clicked, or beside the mannequin if it stands there.
  at = at.clone().setY(0);
  const hip = actor.wpos[0];
  if (Math.hypot(at.x - hip.x, at.z - hip.z) < 0.7) at.add(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).setY(0).normalize());
  const id = 'p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
  S.props.push({ id, type, p: [r4(at.x), SHAPES[type].base, r4(at.z)], q: [0, 0, 0, 1], s: [1, 1, 1] });
  save();
  setPropMode('translate');
  selectProp(id);
}

function deleteProp() {
  if (!propSel) return;
  S.props = S.props.filter((p) => p.id !== propSel);
  selectProp(null);
  save();
}

// Add menu: right-click the viewport. A right-drag still pans the view, so the menu only
// opens when the button comes up close to where it went down.
const ADD_CLICK_PX = 5;
const addList = $('add-list');
addList.innerHTML = '<div class="add-title">Add</div>' + Object.entries(SHAPES).map(([type, sh]) =>
  `<button data-type="${type}" role="menuitem"><span class="ic">${icon(sh.icon)}</span>${sh.label}</button>`).join('');
let addAt = null;     // floor point under the right-click
let rightDown = null; // where the right button went down on the viewport
function openAddMenu(cx, cy) {
  setPointer({ clientX: cx, clientY: cy });
  addAt = groundPoint() || controls.target.clone().setY(0);
  addList.classList.remove('hidden');
  // Opens down and right of the cursor, flipping up / left where the timeline or the edge is in the way.
  const r = viewport.getBoundingClientRect(), w = addList.offsetWidth, h = addList.offsetHeight;
  const floor = Math.min(r.bottom, $('bar').getBoundingClientRect().top) - 8;
  const x = cx + w > r.right - 8 ? cx - w : cx, y = cy + h > floor ? cy - h : cy;
  addList.style.left = `${Math.max(8, x - r.left)}px`;
  addList.style.top = `${Math.max(8, y - r.top)}px`;
}
const closeAddMenu = () => addList.classList.add('hidden');
canvas.addEventListener('pointerdown', (e) => { if (e.button === 2) rightDown = { x: e.clientX, y: e.clientY }; });
window.addEventListener('pointerup', (e) => {
  if (e.button !== 2 || !rightDown) return;
  const click = Math.hypot(e.clientX - rightDown.x, e.clientY - rightDown.y) <= ADD_CLICK_PX;
  rightDown = null;
  if (click && !S.poseMode) openAddMenu(e.clientX, e.clientY);
});
canvas.addEventListener('contextmenu', (e) => e.preventDefault()); // ours, not the browser's
addList.addEventListener('contextmenu', (e) => e.preventDefault());
addList.addEventListener('click', (e) => {
  const b = e.target.closest('[data-type]');
  if (!b) return;
  closeAddMenu();
  addProp(b.dataset.type, addAt);
});
window.addEventListener('pointerdown', (e) => { if (!addList.contains(e.target)) closeAddMenu(); }, true);
canvas.addEventListener('wheel', closeAddMenu, { passive: true });

// Toolbar over the selected shape
const propTools = $('prop-tools');
for (const [mode, ico] of [['translate', 'move'], ['rotate', 'rotate-cw'], ['scale', 'scale-3d']]) setIcon(propTools.querySelector(`[data-mode="${mode}"] .ic`), ico);
setIcon(propTools.querySelector('[data-act="delete"] .ic'), 'trash');
propTools.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.mode) setPropMode(b.dataset.mode);
  else if (b.dataset.act === 'delete') deleteProp();
});
const _propBox = new THREE.Box3(), _propTop = new THREE.Vector3();
function updatePropTools() {
  const m = propSel && propMeshes.get(propSel);
  if (!m || S.poseMode) { propTools.classList.add('hidden'); return; }
  for (const b of propTools.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === tc.mode);
  // Above the shape and clear of the gizmo, whose rings keep the same size on screen.
  _propBox.setFromObject(m);
  _propBox.getCenter(_propTop).setY(_propBox.max.y).project(rcam);
  const r = canvas.getBoundingClientRect();
  const topY = (-_propTop.y * 0.5 + 0.5) * r.height;
  _propTop.copy(m.position).project(rcam);
  const midY = (-_propTop.y * 0.5 + 0.5) * r.height;
  propTools.style.left = `${(_propTop.x * 0.5 + 0.5) * r.width}px`;
  propTools.style.top = `${Math.min(topY, midY - PROP_TOOLS_CLEAR)}px`;
  propTools.classList.toggle('hidden', _propTop.z > 1);
}

// ---------------------------------------------------------------- timeline + buttons
const timeline = new Timeline($('timeline'), {
  getState: () => {
    const ss = syncState();
    return { length: S.length, frame: S.frame, ...ss, keys: ss.real, poses: ss.keys, spanOf, cache: S.cache, generating: S.generating, lastRunLength: S.lastRun.length, selected: S.keySel, spanSel: S.spanSel };
  },
  onPointerDown: () => closeTimelineMenu(),
  onScrub: (f) => { if (S.poseMode) return; S.playing = false; S.scrubbing = true; setFrame(f); },
  onScrubEnd: () => { S.scrubbing = false; },
  onSelect: (ids) => {
    if (S.poseMode) return;
    S.keySel = ids;
    if (ids.size === 1) { const k = S.keys.find((x) => ids.has(x.id)); if (k) selected = k; } // one key: it becomes active
  },
  onMoveStart: () => { if (S.poseMode) return; S.moveSnap = new Map(S.keys.map((k) => [k.id, k.frame])); },
  onMoveKeys: (delta) => moveSelectedKeys(delta),
  onMoveEnd: (moved) => { S.moveSnap = null; S.scrubbing = false; if (moved) save(); },
  onContext: (cx, cy, frame) => { if (!S.poseMode) openTimelineMenu(cx, cy, frame); },
  onHoverSpan: (id) => { hoverSpan = id; }, // the 3D path lights up too
  onSpanClick: (frame) => { // a gap between two keyframes: show that span's settings
    if (S.poseMode) return;
    const real = sortedKeys(), i = real.findIndex((k, j) => real[j + 1] && k.frame < frame && frame < real[j + 1].frame);
    S.spanSel = i >= 0 ? real[i].id : null;
    chipOpen = null;
    if (i >= 0) { selected = null; focusSpan(real[i], real[i + 1]); } // bring its card on screen
  },
  label: (k) => poseLabelOf(k),
});

// ---- timeline selection: move, copy / cut / paste / delete
const selectedKeys = () => S.keys.filter((k) => S.keySel.has(k.id)).sort((a, b) => a.frame - b.frame);

/** Retime the selected keyframes together by `delta` frames (from where the drag started). */
function moveSelectedKeys(delta) {
  if (S.poseMode || !S.moveSnap) return;
  const moving = S.keys.filter((k) => S.keySel.has(k.id));
  const target = new Map(moving.map((k) => [k.id, S.moveSnap.get(k.id) + delta]));
  const others = new Set(S.keys.filter((k) => !S.keySel.has(k.id)).map((k) => k.frame));
  const ok = [...target.values()].every((f) => f >= 0 && f < S.length && !others.has(f));
  if (!ok) return; // would land on another keyframe or off the timeline: stay put
  for (const k of moving) k.frame = target.get(k.id);
  S.scrubbing = true; // fade saved poses like a scrub while things move
  resetDraft();
}

function copySelected() {
  const ks = selectedKeys();
  if (!ks.length) return false;
  const f0 = ks[0].frame;
  S.clip = ks.map((k) => ({ ...JSON.parse(JSON.stringify(k)), offset: k.frame - f0 }));
  toast(`Copied ${ks.length} keyframe${ks.length > 1 ? 's' : ''}.`);
  return true;
}

function deleteSelected(quiet = false) {
  const n = S.keySel.size;
  if (!n) return;
  S.keys = S.keys.filter((k) => !S.keySel.has(k.id));
  if (selected && selected !== 'actor' && !S.keys.includes(selected)) selected = 'actor';
  S.keySel = new Set();
  S.draft = null; resetDraft(); save();
  if (!quiet) toast(`Deleted ${n} keyframe${n > 1 ? 's' : ''}.`);
}

function cutSelected() {
  const n = S.keySel.size;
  if (copySelected()) { deleteSelected(true); toast(`Cut ${n} keyframe${n > 1 ? 's' : ''}.`); }
}

/** Paste the copied keyframes starting at `frame`, keeping their spacing; replaces keys already there. */
function pasteAt(frame) {
  if (!S.clip?.length) return;
  const ids = new Set();
  let skipped = 0;
  for (const c of S.clip) {
    const f = frame + c.offset;
    if (f >= S.length) { skipped++; continue; }
    S.keys = S.keys.filter((k) => k.frame !== f);
    const k = JSON.parse(JSON.stringify(c));
    delete k.offset;
    Object.assign(k, { id: S.nextId++, frame: f });
    S.keys.push(k);
    ids.add(k.id);
  }
  if (selected && selected !== 'actor' && !S.keys.includes(selected)) selected = 'actor';
  S.keySel = ids;
  S.draft = null; resetDraft(); save();
  toast(`Pasted ${ids.size} keyframe${ids.size === 1 ? '' : 's'} at frame ${frame}` + (skipped ? ` (${skipped} past the end skipped).` : '.'));
}

// Right-click menu
const tlMenu = $('tl-menu');
function openTimelineMenu(cx, cy, frame) {
  tlMenu.dataset.frame = frame;
  const has = S.keySel.size > 0;
  for (const [act, on] of [['copy', has], ['cut', has], ['delete', has], ['paste', !!S.clip?.length]]) tlMenu.querySelector(`[data-act="${act}"]`).disabled = !on;
  tlMenu.querySelector('[data-act="paste"] .where').textContent = `at ${frame}`;
  tlMenu.classList.remove('hidden');
  const r = tlMenu.getBoundingClientRect();
  tlMenu.style.left = `${Math.min(cx, innerWidth - r.width - 8)}px`;
  tlMenu.style.top = `${Math.max(8, cy - r.height - 6)}px`; // opens upward, above the timeline
}
function closeTimelineMenu() { tlMenu.classList.add('hidden'); }
tlMenu.addEventListener('click', (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act;
  if (!act || e.target.closest('button').disabled) return;
  if (act === 'copy') copySelected();
  else if (act === 'cut') cutSelected();
  else if (act === 'paste') pasteAt(+tlMenu.dataset.frame);
  else if (act === 'delete') deleteSelected();
  closeTimelineMenu();
});
window.addEventListener('pointerdown', (e) => { if (!tlMenu.contains(e.target)) closeTimelineMenu(); }, true);

$('btn-key').onclick = () => (S.poseMode ? commitPose() : keyPoseButton());
$('btn-cancel-pose').onclick = cancelPose;
$('pose-undo').onclick = poseUndo;
$('pose-redo').onclick = poseRedo;
$('pose-reset').onclick = poseReset;
for (const [id, ico] of [['pose-undo', 'undo-2'], ['pose-redo', 'redo-2'], ['pose-reset', 'rotate-ccw']]) setIcon($(id).querySelector('.ic'), ico);
setIcon($('btn-cancel-pose').querySelector('.ic'), 'x');
// Each mannequin toggle carries its motion path: turning the mannequins off hides the
// path too (and on brings it back). The path toggle can still be flipped on its own after.
const PATH_OF = { keys: 'keyPath', tweens: 'genPath' };
function toggleGhosts(which) {
  S.show[which] = !S.show[which];
  if (PATH_OF[which]) S.show[PATH_OF[which]] = S.show[which];
  if (which === 'bds') S.show.spanHandles = S.show.bds; // and the jump height handles
  if (which === 'actor' && !S.show.actor) S.show.bones = S.show.jointPaths = false; // they go with the body; back on only by choice
  try { localStorage.setItem(STORE + '.view', JSON.stringify(S.show)); } catch {}
}
$('tog-keys').onclick = () => toggleGhosts('keys');
$('tog-tweens').onclick = () => toggleGhosts('tweens');
$('tog-keypath').onclick = () => toggleGhosts('keyPath');
$('tog-handles').onclick = () => toggleGhosts('spanHandles');
$('tog-genpath').onclick = () => toggleGhosts('genPath');
$('tog-bones').onclick = () => toggleGhosts('bones');
$('tog-jointpath').onclick = () => toggleGhosts('jointPaths');
$('tog-feet').onclick = () => toggleGhosts('feet');
$('tog-actor').onclick = () => { toggleGhosts('actor'); if (!S.show.actor) selected = null; };
$('btn-generate').onclick = generate;
$('tog-bds').onclick = () => toggleGhosts('bds');
$('btn-clear-gen').onclick = clearGenerated;
setIcon($('btn-clear-gen').querySelector('.ic'), 'trash');
$('btn-play').onclick = () => { if (!S.poseMode) S.playing = !S.playing; };
$('len-input').onchange = (e) => {
  const minLen = Math.max(30, ...S.keys.map((k) => k.frame + 1));
  S.length = Math.max(minLen, Math.min(3600, Math.round(+e.target.value || 240)));
  e.target.value = S.length; setFrame(Math.min(S.frame, S.length - 1)); save();
};

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (S.poseMode) { // only save / cancel / history while posing
    const mod = e.metaKey || e.ctrlKey;
    if (e.code === 'Escape') cancelPose();
    else if (e.code === 'KeyK' || e.code === 'Enter') commitPose();
    else if (mod && e.code === 'KeyZ') { e.preventDefault(); if (e.shiftKey) poseRedo(); else poseUndo(); }
    else if (mod && e.code === 'KeyY') { e.preventDefault(); poseRedo(); }
    return;
  }
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.code === 'KeyC') { if (copySelected()) e.preventDefault(); return; }
  if (mod && e.code === 'KeyX') { e.preventDefault(); cutSelected(); return; }
  if (mod && e.code === 'KeyV') { e.preventDefault(); pasteAt(S.frame); return; }
  if (e.code === 'Escape' && !addList.classList.contains('hidden')) { closeAddMenu(); return; }
  if (propSel) { // a selected shape takes W / E / R, Delete and Esc
    const mode = { KeyW: 'translate', KeyE: 'rotate', KeyR: 'scale' }[e.code];
    if (mode && !mod) { setPropMode(mode); return; }
    if (e.code === 'Delete' || e.code === 'Backspace') { e.preventDefault(); deleteProp(); return; }
    if (e.code === 'Escape') { selectProp(null); return; }
  }
  if ((e.code === 'Delete' || e.code === 'Backspace') && S.keySel.size) { e.preventDefault(); deleteSelected(); return; }
  if (e.code === 'Escape' && S.keySel.size) { S.keySel = new Set(); closeTimelineMenu(); return; }
  if (e.code === 'Space') { e.preventDefault(); S.playing = !S.playing; }
  else if (e.code === 'ArrowRight') { S.playing = false; setFrame(S.frame + (e.shiftKey ? 10 : 1)); }
  else if (e.code === 'ArrowLeft') { S.playing = false; setFrame(S.frame - (e.shiftKey ? 10 : 1)); }
  else if (e.code === 'Home') setFrame(0);
  else if (e.code === 'KeyK' || e.code === 'Enter') keyPoseButton();
  else if (e.code === 'KeyG') generate();
  else if ((e.code === 'Delete' || e.code === 'Backspace') && keyAt(S.frame)) keyPoseButton();
  else if (e.code === 'Escape') { selected = null; S.spanSel = null; }
  else if (e.code === 'Digit1') toggleGhosts('keys');
  else if (e.code === 'Digit2') toggleGhosts('tweens');
  else if (e.code === 'Digit3') toggleGhosts('actor');
  else if (e.code === 'Digit4') toggleGhosts('keyPath');
  else if (e.code === 'Digit5') toggleGhosts('genPath');
  else if (e.code === 'Digit6') toggleGhosts('bones');
  else if (e.code === 'Digit7') toggleGhosts('jointPaths');
  else if (e.code === 'Digit8') toggleGhosts('bds');
  else if (e.code === 'Digit9') toggleGhosts('feet');
  else if (e.code === 'Digit0') toggleGhosts('spanHandles');
  else if (e.code === 'ArrowUp' || e.code === 'ArrowDown') { e.preventDefault(); { const m = menuTarget(); cyclePose(m.target, e.code === 'ArrowUp' ? -1 : 1, m.mannequin); } }
});

// ---------------------------------------------------------------- per-frame update
const poseLabel = $('pose-label');
const LABEL_Y = 1.2; // metres above the floor (plus any lift)
const _labelAt = new THREE.Vector3();
const pickerOpen = () => picker.target && (performance.now() < picker.until || picker.hold);
/** What the pose menu edits: the active saved pose, or the mannequin's pose at this frame. */
function menuTarget() {
  const act = activeOf(frameState());
  if (act && act.t && act.t !== S.draft) return { target: act.t, mannequin: act.m };
  return { target: editable(), mannequin: actor };
}
/** Delete the active saved pose (from the delete button beside its name). */
function deleteActivePose() {
  const act = activeOf(frameState());
  const key = act && act.t && act.t !== S.draft ? act.t : null;
  if (!key) return;
  S.keys = S.keys.filter((k) => k !== key);
  selected = 'actor';
  picker.target = null;
  resetDraft();
  save();
  toast(`Pose removed from frame ${key.frame}.`);
}

poseLabel.addEventListener('click', (e) => {
  if (e.target.closest('[data-del]')) return deleteActivePose();
  if (e.target.closest('[data-custom]')) return enterPoseMode(picker.target);
  const id = e.target.closest('[data-id]')?.dataset.id;
  if (id && picker.target) return setPose(picker.target, id);
  if (!pickerOpen()) { // clicking the pose name opens the list for the active mannequin
    Object.assign(picker, { ...menuTarget(), until: performance.now() + 2500, hold: true });
  }
});
poseLabel.addEventListener('pointerenter', () => { if (pickerOpen()) picker.hold = true; });
poseLabel.addEventListener('pointerleave', () => {
  if (!pickerOpen()) return;
  picker.hold = false; picker.until = performance.now() + 600;
});
poseLabel.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!pickerOpen()) Object.assign(picker, menuTarget());
  wheelAcc += e.deltaY;
  if (Math.abs(wheelAcc) < 40) return;
  const dir = Math.sign(wheelAcc); wheelAcc = 0;
  cyclePose(picker.target, dir, picker.mannequin);
}, { passive: false });
const tmpV = new THREE.Vector3();

function updateScene() {
  const fs = frameState();
  const nJ = kit.names.length;

  // Actor
  let label = '', sub = '';
  if (S.poseMode) {
    const pm = S.poseMode;
    actor.setPose(pm.root.x, pm.root.y, pm.root.z, pm.rots, 0, 0);
  } else if (fs.mode === 'key') {
    applyPoseOf(actor, fs.key);
    label = poseLabelOf(fs.key); sub = 'saved';
  } else if (fs.mode === 'gen') {
    const r = fs.seg.root, k = fs.local * 3;
    actor.setPose(r[k], r[k + 1], r[k + 2], fs.seg.rot, fs.local * nJ * 4);
  } else {
    if (!S.draft) resetDraft();
    const d = S.draft;
    applyPoseOf(actor, d);
    label = poseLabelOf(d); sub = d.bd && !d.dirty ? 'breakdown' : 'not saved yet';
  }
  actor.visible = S.show.actor;
  if (!S.show.actor) label = '';
  const hot = hover && hover.type === 'actor';
  actor.setMaterials(hot ? hoverMat : woodMat, jointMat);

  // Pose label: to the right of the body at a fixed height, so it doesn't jump with the pose.
  // While scrolling it expands into the list of every pose.
  // It belongs to the active mannequin (click one to make it active).
  const act = activeOf(fs);
  actor.setOutline(act && act.m === actor ? OUTLINE : null);
  for (const k of S.keys) keyGhosts.get(k.id)?.setOutline(act && act.t === k && act.m !== actor ? OUTLINE : null);
  const picking = picker.target && (performance.now() < picker.until || picker.hold);
  const anchorM = picking ? picker.mannequin : act?.m;
  let html = null;
  if (picking) {
    html = '<div class="plist-title">Pose</div>' + S.order.map((id) =>
      `<button data-id="${id}" class="${id === picker.target.pose && !picker.target.custom ? 'on' : ''}">${S.poses[id].label}</button>`).join('')
      + '<button data-custom class="pcustom" title="Pose the joints by hand">Custom…</button>';
  } else if (act) {
    const t = act.t;
    const name = t ? poseLabelOf(t) : 'In-between';
    const state = !t ? 'generated' : t === S.draft ? (t.bd && !t.dirty ? 'breakdown' : 'not saved yet') : 'saved';
    // A saved pose also gets a delete button beside its name.
    const del = t && t !== S.draft ? `<button class="pdel" data-del title="Delete this saved pose (Del)" aria-label="Delete this saved pose">${icon('trash')}</button>` : '';
    html = `<span class="pname">${name}<small>${state}</small><span class="chev">▾</span></span>${del}`;
  }
  if (html !== null && anchorM && !(S.playing || S.scrubbing) && !S.poseMode) {
    const lift = (picking ? picker.target : act?.t)?.y || 0;
    const hip = anchorM.wpos[0];
    tmpV.setFromMatrixColumn(camera.matrixWorld, 0).setY(0).normalize().multiplyScalar(0.42);
    tmpV.add(_labelAt.set(hip.x, LABEL_Y + lift, hip.z)).project(rcam);
    const r = canvas.getBoundingClientRect();
    poseLabel.style.left = `${(tmpV.x * 0.5 + 0.5) * r.width}px`;
    poseLabel.style.top = `${(-tmpV.y * 0.5 + 0.5) * r.height}px`;
    if (html !== picker.html) { poseLabel.innerHTML = html; picker.html = html; }
    poseLabel.classList.toggle('picking', !!picking);
    poseLabel.classList.toggle('active', hot);
    poseLabel.classList.toggle('hidden', tmpV.z > 1);
  } else poseLabel.classList.add('hidden');

  // Orange onion skins: every key, on every frame (except the one being shown live).
  const alive = new Set();
  for (const k of S.keys) {
    alive.add(k.id);
    let g = keyGhosts.get(k.id);
    if (!g) { g = kit.create(keyGhostMat, keyJointMat, { shadows: true, flatOrder: 8 + (k.id % 40) }); scene.add(g.group); keyGhosts.set(k.id, g); }
    applyPoseOf(g, k);
    // While playing or scrubbing every saved pose stays in place (faded), so you can see
    // where the motion is headed. At rest, the one under the playhead hides: the mannequin is in it.
    const moving = S.playing || S.scrubbing;
    g.visible = S.show.keys && (moving || Math.abs(k.frame - S.frame) > KEY_GHOST_HIDE);
    const hot = (hover && hover.type === 'ghost' && hover.key === k) || selected === k;
    if (S.poseMode) g.setMaterials(keyGhostDim); // 10% and not clickable while posing
    else if (S.playing || S.scrubbing) g.setMaterials(keyGhostFade, keyJointFade); // 25% while the timeline moves
    else g.setMaterials(hot ? keyGhostHover : keyGhostMat);
  }
  for (const [id, g] of keyGhosts) if (!alive.has(id)) { scene.remove(g.group); keyGhosts.delete(id); }

  // Green breakdowns: on every frame like the saved poses (hidden under the playhead).
  const bds = fs.keys.filter((k) => k.bd);
  bds.forEach((b, i) => {
    let g = bdGhosts[i];
    if (!g) { g = kit.create(bdGhostMat, bdGhostMat, { flatOrder: 60 + (i % 30) }); scene.add(g.group); bdGhosts.push(g); }
    applyPoseOf(g, b);
    const moving = S.playing || S.scrubbing;
    g.visible = S.show.bds && (moving || Math.abs(b.frame - S.frame) > KEY_GHOST_HIDE);
    g.setMaterials(S.poseMode ? bdGhostDim : moving ? bdGhostFade : bdGhostMat);
  });
  for (let i = bds.length; i < bdGhosts.length; i++) bdGhosts[i].visible = false;
  updateSpans(fs);
  updatePaths(fs);
  updateBones();
  updateJointPaths(fs);
  updateFootContacts();
  if (S.poseMode && propSel) selectProp(null);
  syncProps();
  updatePropTools();
  updatePoseNodes();

  // Blue onion skins: generated in-betweens around the current frame.
  const { keys, sigs, ok } = fs;
  ONION_OFFSETS.forEach((off, i) => {
    const g = genGhosts[i];
    g.visible = false;
    if (!S.show.tweens) return;
    const f = S.frame + off;
    for (let s = 0; s + 1 < keys.length; s++) {
      if (keys[s].frame < f && f < keys[s + 1].frame && ok[s] && !keys.some((k) => k.frame === f)) {
        const seg = S.cache.get(sigs[s]), local = f - keys[s].frame;
        g.setPose(seg.root[local * 3], seg.root[local * 3 + 1], seg.root[local * 3 + 2], seg.rot, local * nJ * 4);
        g.setMaterials(S.poseMode ? genGhostDim : genGhostMats[i]);
        g.visible = true;
        break;
      }
    }
  });

  // Gizmo follows the actor.
  const busy = S.playing || S.scrubbing; // no handles or pose list while the timeline moves
  if (busy) { picker.until = 0; picker.hold = false; }
  let t = null; // what the gizmo edits: the mannequin, or a selected saved pose
  if (!busy && !S.poseMode && !(drag && drag.type === 'free')) {
    if (selected === 'actor' && S.show.actor && fs.mode !== 'gen') t = fs.mode === 'key' ? fs.key : S.draft;
    else if (selected && selected !== 'actor' && S.keys.includes(selected) && S.show.keys) t = selected;
  }
  gizmo.visible = !!t;
  if (gizmo.visible) {
    gizmo.position.set(t.x, 0.015 + (t.y || 0), t.z);
    gizmo.userData.facing.position.set(Math.sin(t.yaw) * (TURN_R + 0.06), 0, Math.cos(t.yaw) * (TURN_R + 0.06));
    gizmo.userData.facing.rotation.set(Math.PI / 2, 0, -t.yaw);
    for (const o of gizmoPick) { const hd = o.userData.handle; hd.mat.color.set(hover && hover.handle === hd ? '#ffffff' : hd.color); }
  }

  // Shadow follows the actor around the infinite floor.
  sun.target.position.copy(actor.wpos[0]).setY(0);
  sun.position.copy(sun.target.position).add(new THREE.Vector3(3, 7, 4));

  // Buttons
  const keyBtn = $('btn-key');
  const setBtn = (btn, ico, title, badge = '', badgeClass = '') => {
    setIcon(btn.querySelector('.ic'), ico);
    btn.querySelector('.t').textContent = title;
    const s = btn.querySelector('.badge'); s.textContent = badge; s.className = 'badge ' + badgeClass;
  };
  $('btn-cancel-pose').classList.toggle('hidden', !S.poseMode);
  $('pose-mode-tag').classList.toggle('hidden', !S.poseMode);
  if (S.poseMode) { setBtn(keyBtn, 'diamond-plus', 'Add Keyframe'); keyBtn.className = 'big key armed'; }
  else if (fs.mode === 'key') { setBtn(keyBtn, 'diamond-minus', 'Remove Keyframe'); keyBtn.className = 'big key remove'; }
  else { setBtn(keyBtn, 'diamond-plus', 'Add Keyframe'); keyBtn.className = 'big key' + (S.draft && S.draft.dirty ? ' armed' : ''); }
  const genBtn = $('btn-generate');
  genBtn.classList.toggle('busy', !!S.generating);
  $('btn-clear-gen').disabled = !!S.generating || !S.cache.size;
  if (S.generating) setBtn(genBtn, 'loader-circle', 'Generating');
  else if (fs.tag === 'stale') setBtn(genBtn, 'refresh-cw', 'AI In-betweens', 'Out of sync', 'warn');
  else if (fs.tag === 'more') setBtn(genBtn, 'sparkles', 'AI In-betweens', 'New keyframes', 'more');
  else setBtn(genBtn, 'sparkles', 'AI In-betweens');
  for (const [id, on, ico] of [['tog-actor', S.show.actor, 'person'], ['tog-bones', S.show.bones, 'bone'], ['tog-jointpath', S.show.jointPaths, 'spline'], ['tog-feet', S.show.feet, 'footprints'], ['tog-keys', S.show.keys, 'person'], ['tog-bds', S.show.bds, 'person'], ['tog-tweens', S.show.tweens, 'person'],
    ['tog-keypath', S.show.keyPath, 'spline'], ['tog-handles', S.show.spanHandles, 'move-vertical'], ['tog-genpath', S.show.genPath, 'spline']]) {
    const el = $(id);
    el.classList.toggle('on', on);
    setIcon(el.querySelector('.ic'), ico);
  }
  for (const id of S.keySel) if (!S.keys.some((k) => k.id === id)) S.keySel.delete(id);
  document.body.classList.toggle('playing', !!S.playing); // the big buttons hide while playing
  $('frame-readout').textContent = S.frame;
  setIcon($('btn-play').querySelector('.ic'), S.playing ? 'pause' : 'play');
}

/** Swap an icon only when it changes, so the per-frame update stays cheap. */
function setIcon(el, name) {
  if (el.dataset.icon === name) return;
  el.innerHTML = icon(name);
  el.dataset.icon = name;
}

function resize() {
  const r = viewport.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  keyPathMat.resolution.set(r.width, r.height);
  genPathMat.resolution.set(r.width, r.height);
  boneMat.resolution.set(r.width, r.height);
  jointPathMat.resolution.set(r.width, r.height);
  spanLineMat.resolution.set(r.width, r.height);
  footLineMat.resolution.set(r.width, r.height);
  camera.updateProjectionMatrix();
  timeline.resize();
}
window.addEventListener('resize', resize);

let last = performance.now(), acc = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (S.playing) {
    acc += dt;
    while (acc >= 1 / 30) { acc -= 1 / 30; S.frame = (S.frame + 1) % S.length; S.draft = null; }
  } else acc = 0;
  stepCamTween(now);
  stepViewTween(now);
  controls.update();
  syncViewCamera();
  tc.camera = rcam;
  floorMat.uniforms.uCam.value.copy(camera.position);
  updateScene();
  timeline.draw();
  renderer.render(scene, rcam);
  viewCube.render(camera);
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    const lib = await (await fetch('/api/library')).json();
    S.lib = lib;
    window.__studio = { S, setFrame, keyPoseButton, generate, cyclePose, editable, syncState, get actor() { return actor; }, camera, controls, get rcam() { return rcam; }, viewCube, snapView, get drag() { return drag; }, get hover() { return hover; }, footLines, addProp, selectProp, setPropMode, tc, propMeshes, keyGhosts, genGhosts, bdGhosts, spanHandles, chain, keyPath, genPath, jointPath, step: () => { updateScene(); renderer.render(scene, rcam); }, fkSoma, kit: () => kit, poseOps: { twoBone, aimBone, spineLean, applySafely, poseUndo, poseRedo, poseReset }, landCamera: () => stepCamTween(performance.now() + 1e6), jointGizmo, jgPick };
    for (const p of lib.poses) { p.rot = new Float32Array(p.rot.flat()); S.poses[p.id] = p; S.order.push(p.id); }
    kit = await MannequinKit.load(lib.skeleton, S.poses.stand, '/models/mannequin.glb');
    actor = kit.create(woodMat, jointMat, { shadows: true });
    scene.add(actor.group);
    for (let i = 0; i < ONION_OFFSETS.length; i++) { const g = kit.create(genGhostMats[i], genGhostMats[i], { flatOrder: i }); g.visible = false; scene.add(g.group); genGhosts.push(g); }
    applyLibraryPose(actor, 'stand', 0, 0, 0);
    S.floorY = actor.rawMinY; // lowest joint of a grounded pose
    load();
    $('len-input').value = S.length;
    resetDraft();
    $('loading').classList.add('hidden');
    resize();
    if (!S.keys.length) toast('Welcome! Pick a moment on the timeline, click the pose name to choose a pose, then press Add Keyframe.');
    requestAnimationFrame(loop);
  } catch (e) {
    $('loading').textContent = 'Could not reach the MotionBricks server: ' + e.message;
  }
})();
