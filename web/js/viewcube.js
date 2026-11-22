// A 3ds Max style view cube: a small labelled cube in the corner that turns with the camera.
// Click a face, edge or corner to snap the view there; drag it to orbit.
import * as THREE from 'three';

const EDGE = 0.22;        // width of the edge and corner bands, as a fraction of the cube
const HILITE = '#5b94ff';
const FACES = [ // BoxGeometry material order: +x, -x, +y, -y, +z, -z
  { label: 'RIGHT', shade: 0.86 }, { label: 'LEFT', shade: 0.86 },
  { label: 'TOP', shade: 1.0 }, { label: 'BOTTOM', shade: 0.62 },
  { label: 'FRONT', shade: 0.94 }, { label: 'BACK', shade: 0.74 },
];
const AXIS_NAMES = [['Left', '', 'Right'], ['Bottom', '', 'Top'], ['Back', '', 'Front']];

/** Name of a view direction made of -1/0/1 components, e.g. "Top Front Right". */
export function viewName(v) {
  return [AXIS_NAMES[1][v.y + 1], AXIS_NAMES[2][v.z + 1], AXIS_NAMES[0][v.x + 1]].filter(Boolean).join(' ');
}

function faceTexture(label) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#262a32'; g.fillRect(0, 0, 256, 256);
  g.strokeStyle = '#3a404b'; g.lineWidth = 6; g.strokeRect(3, 3, 250, 250);
  g.fillStyle = '#d4d8df';
  g.font = `600 ${label.length > 5 ? 40 : 46}px Jost, Futura, "Avenir Next", "Century Gothic", sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  if ('letterSpacing' in g) g.letterSpacing = '4px';
  g.fillText(label, 128, 132);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

export class ViewCube {
  /**
   * @param {HTMLElement} host  element the cube's canvas goes into
   * @param {{onSnap:(dir:THREE.Vector3, ijk:THREE.Vector3)=>void, onOrbit:(dx:number, dy:number)=>void}} cb
   */
  constructor(host, cb) {
    this.cb = cb;
    this.size = 120;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(this.size, this.size);
    this.canvas = this.renderer.domElement;
    this.canvas.className = 'vc-canvas';
    host.appendChild(this.canvas);

    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
    const mats = FACES.map((f) => new THREE.MeshBasicMaterial({ map: faceTexture(f.label), color: new THREE.Color(f.shade, f.shade, f.shade) }));
    this.cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mats);
    this.scene.add(this.cube);
    this.scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(this.cube.geometry),
      new THREE.LineBasicMaterial({ color: '#59606c' })));
    // Soft floor shadow so you can tell up from down at a glance. It fades to nothing well
    // inside the canvas, so no view clips its edge, and hides when you look from below.
    const sc = document.createElement('canvas'); sc.width = sc.height = 128;
    const sg = sc.getContext('2d'), grad = sg.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(0,0,0,0.5)'); grad.addColorStop(0.55, 'rgba(0,0,0,0.25)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
    sg.fillStyle = grad; sg.fillRect(0, 0, 128, 128);
    this.shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.25, 1.25),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sc), transparent: true, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2; this.shadow.position.y = -0.56;
    this.scene.add(this.shadow);

    // 26 hit regions (6 faces, 12 edges, 8 corners) that light up under the pointer.
    this.parts = [];
    const span = (c) => (c === 0 ? 1 - 2 * EDGE : EDGE) + 0.006;
    const at = (c) => c * (0.5 - EDGE / 2 + 0.003);
    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
      if (!x && !y && !z) continue;
      const m = new THREE.Mesh(new THREE.BoxGeometry(span(x), span(y), span(z)),
        new THREE.MeshBasicMaterial({ color: HILITE, transparent: true, opacity: 0, depthWrite: false }));
      m.position.set(at(x), at(y), at(z));
      m.userData.ijk = new THREE.Vector3(x, y, z);
      this.parts.push(m);
      this.scene.add(m);
    }

    // In a straight-on view the neighbouring faces are edge-on, so arrows step to them.
    this.arrows = document.createElement('div');
    this.arrows.className = 'vc-arrows hidden';
    for (const side of ['up', 'down', 'left', 'right']) {
      const b = document.createElement('button');
      b.className = 'vc-arrow ' + side; b.title = 'Turn ' + side; b.setAttribute('aria-label', 'Turn the view ' + side);
      b.addEventListener('click', () => this.step(side));
      this.arrows.appendChild(b);
    }
    host.appendChild(this.arrows);
    this.basis = { up: new THREE.Vector3(), right: new THREE.Vector3() };

    this.ray = new THREE.Raycaster();
    this.hover = null;
    this.down = null;
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      this.down = { x: e.clientX, y: e.clientY, lx: e.clientX, ly: e.clientY, moved: false };
    });
    c.addEventListener('pointermove', (e) => {
      const d = this.down;
      if (d) {
        if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 3) { d.moved = true; this.setHover(null); c.style.cursor = 'grabbing'; }
        if (d.moved) { this.cb.onOrbit(e.clientX - d.lx, e.clientY - d.ly); d.lx = e.clientX; d.ly = e.clientY; }
        return;
      }
      this.setHover(this.pick(e));
    });
    const end = (e) => {
      const d = this.down; this.down = null;
      c.style.cursor = '';
      if (!d || d.moved || e.type === 'pointercancel') return;
      const p = this.pick(e);
      if (p) this.cb.onSnap(p.userData.ijk.clone().normalize(), p.userData.ijk.clone());
      this.setHover(this.pick(e));
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('pointerleave', () => { if (!this.down) this.setHover(null); });
  }

  pick(e) {
    const r = this.canvas.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(p, this.cam);
    const hit = this.ray.intersectObjects(this.parts, false)[0];
    return hit ? hit.object : null;
  }

  setHover(part) {
    if (part === this.hover) return;
    if (this.hover) this.hover.material.opacity = 0;
    this.hover = part;
    if (part) part.material.opacity = 0.62;
    this.canvas.style.cursor = part ? 'pointer' : '';
    this.canvas.title = part ? viewName(part.userData.ijk) : 'Drag to orbit';
  }

  /** Snap to the face on that side of the screen. */
  step(side) {
    const v = (side === 'up' || side === 'down' ? this.basis.up : this.basis.right).clone();
    if (side === 'down' || side === 'left') v.negate();
    const ijk = new THREE.Vector3(Math.round(v.x), Math.round(v.y), Math.round(v.z));
    if (ijk.lengthSq() !== 1) return;
    this.cb.onSnap(ijk.clone(), ijk);
  }

  /** Turn the cube to match the main camera and draw it. */
  render(camera) {
    const e = camera.matrixWorld.elements;
    this.basis.right.set(e[0], e[1], e[2]); this.basis.up.set(e[4], e[5], e[6]);
    const back = new THREE.Vector3(e[8], e[9], e[10]);
    const faceOn = Math.max(Math.abs(back.x), Math.abs(back.y), Math.abs(back.z)) > 0.9995;
    this.arrows.classList.toggle('hidden', !faceOn);
    this.shadow.material.opacity = THREE.MathUtils.clamp(back.y * 6 + 0.6, 0, 1); // gone from below
    this.cam.quaternion.copy(camera.quaternion);
    this.cam.position.set(0, 0, 3.45).applyQuaternion(camera.quaternion);
    this.renderer.render(this.scene, this.cam);
  }
}
