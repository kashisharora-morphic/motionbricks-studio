// A rigged base mesh driven by SOMA poses.
//
// The model is "Male Base Mesh" by orange-juice-games (CC0 1.0, public domain), a Blender
// Rigify metarig in a T-pose: web/models/mannequin.glb. RIG maps its bones to the names the
// rest of the app uses (Hips, LeftUpLeg, ...), and the kit adds end bones for the toe tips
// and the top of the head, which the rig lacks.
//
// MotionBricks outputs SOMA local joint rotations. The SOMA rest pose is not a T-pose
// (every bone points down its local X axis), so the model is calibrated against a matching
// pose instead: SOMA "stand", and the model turned from its T-pose into the same stand
// (CAL_AIM points each bone where SOMA's does; the head and feet keep their upright T-pose
// rotation). For each mapped bone, K = W_soma_cal^-1 * M_model_cal, and at runtime
// M_model = W_soma * K. Limb bones then get a small swing so their direction matches the
// SOMA bone exactly, which keeps hands and feet where the model put them.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

// App bone name -> this model's bone (glTF names lose their dots: "spine.001" -> "spine001").
const RIG = {
  Hips: 'spine', Spine: 'spine001', Spine1: 'spine002', Spine2: 'spine003', Neck: 'spine004', Head: 'spine005',
  LeftShoulder: 'shoulderL', LeftArm: 'upper_armL', LeftForeArm: 'forearmL', LeftHand: 'handL',
  RightShoulder: 'shoulderR', RightArm: 'upper_armR', RightForeArm: 'forearmR', RightHand: 'handR',
  LeftUpLeg: 'thighL', LeftLeg: 'shinL', LeftFoot: 'footL', LeftToeBase: 'toeL', LeftHeel: 'heel02L',
  RightUpLeg: 'thighR', RightLeg: 'shinR', RightFoot: 'footR', RightToeBase: 'toeR', RightHeel: 'heel02R',
};
const TOE_LEN = 0.07; // toe joint to the tip of the toes, metres
// Calibration: bone -> [its child bone, SOMA joint, SOMA child]; the bone is turned so it
// points the way that SOMA bone points in the stand pose.
const CAL_AIM = {
  Hips: ['Spine', 'Hips', 'Spine1'], Spine: ['Spine1', 'Spine1', 'Spine2'], Spine1: ['Spine2', 'Spine2', 'Chest'],
  Spine2: ['Neck', 'Chest', 'Neck1'], Neck: ['Head', 'Neck1', 'Head'],
  LeftShoulder: ['LeftArm', 'LeftShoulder', 'LeftArm'], LeftArm: ['LeftForeArm', 'LeftArm', 'LeftForeArm'], LeftForeArm: ['LeftHand', 'LeftForeArm', 'LeftHand'],
  RightShoulder: ['RightArm', 'RightShoulder', 'RightArm'], RightArm: ['RightForeArm', 'RightArm', 'RightForeArm'], RightForeArm: ['RightHand', 'RightForeArm', 'RightHand'],
  LeftUpLeg: ['LeftLeg', 'LeftLeg', 'LeftShin'], LeftLeg: ['LeftFoot', 'LeftShin', 'LeftFoot'],
  RightUpLeg: ['RightLeg', 'RightLeg', 'RightShin'], RightLeg: ['RightFoot', 'RightShin', 'RightFoot'],
};
const CAL_UPRIGHT = ['Head', 'LeftFoot', 'RightFoot']; // keep their T-pose world rotation: head up, soles flat

const MAP = { // app bone -> SOMA joint
  Hips: 'Hips', Spine: 'Spine1', Spine1: 'Spine2', Spine2: 'Chest', Neck: 'Neck1', Head: 'Head',
  LeftShoulder: 'LeftShoulder', LeftArm: 'LeftArm', LeftForeArm: 'LeftForeArm', LeftHand: 'LeftHand',
  RightShoulder: 'RightShoulder', RightArm: 'RightArm', RightForeArm: 'RightForeArm', RightHand: 'RightHand',
  LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftShin', LeftFoot: 'LeftFoot', LeftToeBase: 'LeftToeBase',
  RightUpLeg: 'RightLeg', RightLeg: 'RightShin', RightFoot: 'RightFoot', RightToeBase: 'RightToeBase',
};
// Limb bones whose direction is matched exactly: model child bone, SOMA joint -> SOMA child.
const SWING = {
  LeftArm: ['LeftForeArm', 'LeftArm', 'LeftForeArm'], LeftForeArm: ['LeftHand', 'LeftForeArm', 'LeftHand'],
  RightArm: ['RightForeArm', 'RightArm', 'RightForeArm'], RightForeArm: ['RightHand', 'RightForeArm', 'RightHand'],
  LeftUpLeg: ['LeftLeg', 'LeftLeg', 'LeftShin'], LeftLeg: ['LeftFoot', 'LeftShin', 'LeftFoot'],
  RightUpLeg: ['RightLeg', 'RightLeg', 'RightShin'], RightLeg: ['RightFoot', 'RightShin', 'RightFoot'],
  // Feet are left out on purpose: the model's ankle can sit higher above the ball of the
  // foot than SOMA's, so matching the ankle->ball direction would tip the toes up and drop
  // the heel. The calibrated rotation alone keeps the sole as flat as SOMA's.
};
const BOX_PAD = 0.1;   // flesh around the joints, metres
const HEAD_TOP = 0.2;  // head joint to the top of the head
const Y = new THREE.Vector3(0, 1, 0);
const APP_NAME = Object.fromEntries(Object.entries(RIG).map(([app, model]) => [model, app]));
const appName = (name) => APP_NAME[name] || name;

function headingOf(right, left) { // same convention as the model: forward = Y x (right - left)
  const ax = right.x - left.x, az = right.z - left.z;
  return Math.atan2(az, -ax);
}

/** Forward kinematics on the SOMA skeleton into reusable arrays. */
const X = new THREE.Vector3(1, 0, 0);
function somaFK(soma, rx, ry, rz, rots, offset, yaw, pos, quat, pitch = 0) {
  // Root turn: yaw about up, after pitch about the body's side axis (a flip).
  const yq = new THREE.Quaternion().setFromAxisAngle(Y, yaw);
  if (pitch) yq.multiply(new THREE.Quaternion().setFromAxisAngle(X, pitch));
  for (let j = 0; j < soma.parents.length; j++) {
    const k = offset + j * 4;
    const q = quat[j].set(rots[k], rots[k + 1], rots[k + 2], rots[k + 3]);
    const p = soma.parents[j];
    if (p < 0) { q.premultiply(yq); pos[j].set(rx, ry, rz); }
    else { pos[j].copy(soma.offsets[j]).applyQuaternion(quat[p]).add(pos[p]); q.premultiply(quat[p]); }
  }
}

export class MannequinKit {
  static async load(skel, standPose, url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    return new MannequinKit(skel, standPose, gltf);
  }

  constructor(skel, stand, gltf) {
    this.names = skel.names;
    this.soma = { parents: skel.parents, offsets: skel.offsets.map((o) => new THREE.Vector3(...o)) };
    this.index = Object.fromEntries(skel.names.map((n, i) => [n, i]));
    const J = skel.names.length;
    const newArrays = () => ({ pos: Array.from({ length: J }, () => new THREE.Vector3()), quat: Array.from({ length: J }, () => new THREE.Quaternion()) });

    // SOMA calibration pose: "stand", heading 0 (faces +Z).
    const cal = newArrays();
    somaFK(this.soma, stand.root[0], stand.root[1], stand.root[2], stand.rot, 0, 0, cal.pos, cal.quat);
    // SOMA rest forward in root-local axes, used to read a facing back out of a pose.
    const rest = newArrays();
    const restRots = new Float32Array(J * 4); for (let j = 0; j < J; j++) restRots[j * 4 + 3] = 1;
    somaFK(this.soma, 0, 0, 0, restRots, 0, 0, rest.pos, rest.quat);
    const r = rest.pos[this.index.RightLeg], l = rest.pos[this.index.LeftLeg];
    this.restForward = new THREE.Vector3(r.z - l.z, 0, -(r.x - l.x)).normalize();

    // Model calibration pose: its T-pose turned into SOMA's stand (see CAL_AIM).
    this.template = gltf.scene;
    this.template.updateMatrixWorld(true);
    const byName = {};
    this.template.traverse((o) => { if (o.isBone) byName[appName(o.name)] = o; });
    const wp = (n) => byName[n].getWorldPosition(new THREE.Vector3());
    const setWorld = (b, q) => {
      b.quaternion.copy(b.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(q));
      b.updateMatrixWorld(true);
    };
    // End points the rig lacks, as child bones: toe tips along each toe, the top of the head.
    const addEnd = (name, parent, world) => {
      const e = new THREE.Bone(); e.name = name;
      e.position.copy(byName[parent].worldToLocal(world.clone()));
      byName[parent].add(e); byName[name] = e;
    };
    for (const side of ['Left', 'Right']) {
      const toe = byName[side + 'ToeBase'];
      const along = new THREE.Vector3(0, 1, 0).applyQuaternion(toe.getWorldQuaternion(new THREE.Quaternion()));
      addEnd(side + 'Toe_End', side + 'ToeBase', wp(side + 'ToeBase').addScaledVector(along, TOE_LEN));
    }
    const headTop = new THREE.Box3().setFromObject(this.template).max.y;
    addEnd('HeadTop_End', 'Head', wp('Head').setY(headTop));
    this.template.updateMatrixWorld(true);
    // Which way the T-pose faces; `fix` turns it to face +Z like SOMA's stand.
    const fix = new THREE.Quaternion().setFromAxisAngle(Y, -headingOf(wp('RightUpLeg'), wp('LeftUpLeg')));
    const unfix = fix.clone().invert();
    const tWorld = {};
    for (const n of CAL_UPRIGHT) tWorld[n] = byName[n].getWorldQuaternion(new THREE.Quaternion());
    this.template.traverse((b) => { // parents before children
      if (!b.isBone) return;
      const name = appName(b.name), aim = CAL_AIM[name];
      if (aim) {
        const from = wp(aim[0]).sub(b.getWorldPosition(new THREE.Vector3())).normalize();
        const to = cal.pos[this.index[aim[2]]].clone().sub(cal.pos[this.index[aim[1]]]).normalize().applyQuaternion(unfix);
        setWorld(b, b.getWorldQuaternion(new THREE.Quaternion()).premultiply(new THREE.Quaternion().setFromUnitVectors(from, to)));
      } else if (tWorld[name]) setWorld(b, tWorld[name]);
    });
    const bones = [];
    this.template.traverse((o) => { if (o.isBone) bones.push(o); });

    this.bones = bones.map((b) => {
      const name = appName(b.name);
      const parent = bones.indexOf(b.parent);
      const calWorld = b.getWorldQuaternion(new THREE.Quaternion()).premultiply(fix);
      const info = { name, objName: b.name, parent, calLocal: b.quaternion.clone(), soma: -1, K: null, swing: null };
      const sj = MAP[name] !== undefined ? this.index[MAP[name]] : undefined;
      if (sj !== undefined) {
        info.soma = sj;
        info.K = cal.quat[sj].clone().invert().multiply(calWorld);
      }
      const sw = SWING[name];
      if (sw && byName[sw[0]]) info.swing = { child: byName[sw[0]].position.clone().normalize(), a: this.index[sw[1]], b: this.index[sw[2]] };
      return info;
    });
    const hips = byName.Hips;
    this.rootParentQuat = hips.parent.getWorldQuaternion(new THREE.Quaternion());
    this.rootParentInv = hips.parent.matrixWorld.clone().invert();
    this.headIndex = bones.findIndex((b) => appName(b.name) === 'Head');
    // Ground probes: ankle, heel, ball and toe tip keep their standing height above the floor;
    // knees keep a small radius. Poses are lifted so no probe goes below y = 0.
    const soles = ['LeftFoot', 'LeftHeel', 'LeftToeBase', 'LeftToe_End', 'RightFoot', 'RightHeel', 'RightToeBase', 'RightToe_End'].filter((n) => byName[n]);
    const floorY = Math.min(...soles.map((n) => wp(n).y));
    this.heightScale = (wp('Hips').y - floorY) / cal.pos[0].y; // the model's size against SOMA's
    this.groundProbes = [
      ...soles.map((n) => ({ i: bones.indexOf(byName[n]), h: wp(n).y - floorY })),
      ...['LeftLeg', 'RightLeg'].map((n) => ({ i: bones.indexOf(byName[n]), h: 0.06 })),
      // Flips and crouches: keep the head and hands out of the floor too.
      ...[['HeadTop_End', 0.02], ['Head', 0.1], ['LeftHand', 0.04], ['RightHand', 0.04]]
        .filter(([n]) => byName[n]).map(([n, h]) => ({ i: bones.indexOf(byName[n]), h })),
    ];
    this._newArrays = newArrays;
  }

  create(bodyMat, jointMat = bodyMat, opts = {}) { return new Mannequin(this, bodyMat, jointMat, opts); }
}

const _m = new THREE.Quaternion(), _inv = new THREE.Quaternion(), _sw = new THREE.Quaternion();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _p = new THREE.Vector3(), _hit = new THREE.Vector3();
// Outline: a slightly inflated back-face copy of the body. Drawn late (after the
// see-through ghosts' depth pass), so only the rim around the silhouette shows.
const OUTLINE_WIDTH = 0.014; // metres
const outlineMats = new Map();
function outlineMaterial(color) {
  if (!outlineMats.has(color)) {
    const m = new THREE.MeshBasicMaterial({ color, side: THREE.BackSide, transparent: true, depthWrite: false });
    m.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        '#include <begin_vertex>', `#include <begin_vertex>\n  transformed += normalize(normal) * ${OUTLINE_WIDTH.toFixed(4)};`);
    };
    outlineMats.set(color, m);
  }
  return outlineMats.get(color);
}

const _box = new THREE.Box3(), _ray = new THREE.Ray(), _local = new THREE.Matrix4(), _yq = new THREE.Quaternion(), _one = new THREE.Vector3(1, 1, 1);

export class Mannequin {
  constructor(kit, bodyMat, jointMat, { shadows = false, flatOrder = null } = {}) {
    this.kit = kit;
    this.group = SkeletonUtils.clone(kit.template);
    const byObj = {};
    this.group.traverse((o) => { if (o.isBone) byObj[o.name] = o; });
    this.bones = kit.bones.map((b) => byObj[b.objName]);
    this.world = kit.bones.map(() => new THREE.Quaternion());
    const { pos, quat } = kit._newArrays();
    this.wpos = pos; this.wquat = quat;
    this.meshes = [];
    this.group.traverse((o) => {
      if (!o.isSkinnedMesh) return;
      o.frustumCulled = false;
      o.castShadow = shadows;
      o.userData.kind = /joint/i.test(o.name + (o.material && o.material.name)) ? 'joint' : 'body';
      this.meshes.push(o);
    });
    if (flatOrder !== null) this._makeFlat(flatOrder);
    this.setMaterials(bodyMat, jointMat);
  }

  /** Flat 2D look: a depth-only pass first so the ghost reads as one silhouette. */
  _makeFlat(order) {
    const depthMat = new THREE.MeshBasicMaterial({ colorWrite: false, transparent: true });
    for (const mesh of [...this.meshes]) {
      const d = mesh.clone();
      d.bind(mesh.skeleton, mesh.bindMatrix);
      d.material = depthMat;
      d.renderOrder = 10 + order * 2;
      mesh.renderOrder = 11 + order * 2;
      mesh.parent.add(d);
    }
  }

  /** Draw an outline in `color` around the body, or remove it with null. */
  setOutline(color) {
    if (color === (this._outline ?? null)) return;
    this._outline = color;
    if (!this.hull && color) {
      this.hull = this.meshes.map((mesh) => {
        const h = mesh.clone();
        h.bind(mesh.skeleton, mesh.bindMatrix);
        h.castShadow = false;
        h.frustumCulled = false;
        h.renderOrder = 900;
        mesh.parent.add(h);
        return h;
      });
    }
    for (const h of this.hull || []) { h.visible = !!color; if (color) h.material = outlineMaterial(color); }
  }

  setMaterials(body, joint = body) {
    for (const m of this.meshes) m.material = m.userData.kind === 'joint' ? joint : body;
  }

  /** root: world position; rots: flat SOMA local xyzw array from `offset`; yaw turns the root. */
  setPose(rx, ry, rz, rots, offset = 0, yaw = 0, pitch = 0) {
    const kit = this.kit;
    somaFK(kit.soma, rx, ry, rz, rots, offset, yaw, this.wpos, this.wquat, pitch);
    this.rawMinY = Infinity; // lowest SOMA joint before the floor clamp
    for (const p of this.wpos) this.rawMinY = Math.min(this.rawMinY, p.y);
    kit.bones.forEach((info, i) => {
      const parentWorld = info.parent >= 0 ? this.world[info.parent] : kit.rootParentQuat;
      const M = this.world[i];
      if (info.K) {
        M.copy(this.wquat[info.soma]).multiply(info.K);
        if (info.swing) {
          _a.copy(info.swing.child).applyQuaternion(M);
          _b.subVectors(this.wpos[info.swing.b], this.wpos[info.swing.a]).normalize();
          M.premultiply(_sw.setFromUnitVectors(_a, _b));
        }
      } else M.copy(parentWorld).multiply(info.calLocal);
      this.bones[i].quaternion.copy(_inv.copy(parentWorld).invert().multiply(M));
    });
    const hip = this.wpos[0];
    const hipY = hip.y * kit.heightScale;
    _p.set(hip.x, hipY, hip.z).applyMatrix4(kit.rootParentInv);
    this.bones[0].position.copy(_p);

    // Keep the legs above the floor: lift the whole body by the deepest penetration.
    this.group.updateMatrixWorld(true);
    let lift = 0;
    for (const g of kit.groundProbes) lift = Math.max(lift, g.h - this.bones[g.i].getWorldPosition(_hit).y);
    this.lift = lift;
    if (lift > 1e-4) {
      _p.set(hip.x, hipY + lift, hip.z).applyMatrix4(kit.rootParentInv);
      this.bones[0].position.copy(_p);
      for (const p of this.wpos) p.y += lift; // keep picking in sync
    }
  }

  /**
   * The body's bones as line segments (x,y,z pairs) plus joint points, from the model's
   * skeleton that moves the mesh. Fingers and eyes are left out to keep it readable.
   */
  boneLines() {
    if (!this._boneIdx) {
      const skip = /thumb|^f_|pelvis|Heel|Eye/; // fingers, the Rigify pelvis helpers, heels
      const keep = this.kit.bones.map((b) => !skip.test(b.name));
      this._boneIdx = this.kit.bones.map((b, i) => i).filter((i) => keep[i] && this.kit.bones[i].parent >= 0 && keep[this.kit.bones[i].parent]);
      this._jointIdx = this.kit.bones.map((b, i) => i).filter((i) => keep[i]);
    }
    this.group.updateMatrixWorld(true);
    const at = (i) => this.bones[i].getWorldPosition(_hit);
    const seg = [], joints = [];
    for (const i of this._boneIdx) {
      const c = at(i); const cx = c.x, cy = c.y, cz = c.z;
      const p = at(this.kit.bones[i].parent);
      seg.push(p.x, p.y, p.z, cx, cy, cz);
    }
    for (const i of this._jointIdx) { const c = at(i); joints.push(c.x, c.y, c.z); }
    return { seg, joints };
  }

  /** World positions (x,y,z triples) of named model bones (app names), appended to `out`. */
  jointPositions(names, out) {
    if (!this._named) this._named = {};
    const key = names.join();
    if (!this._named[key]) this._named[key] = names.map((n) => this.kit.bones.findIndex((b) => b.name === n));
    this.group.updateMatrixWorld(true);
    for (const i of this._named[key]) { const p = this.bones[i].getWorldPosition(_hit); out.push(p.x, p.y, p.z); }
    return out;
  }

  /**
   * Facing (yaw, 0 = +Z) of the current pose, measured across the hips the way the model
   * and the pose library measure it, so it doesn't tilt with the pelvis.
   */
  facing() {
    const { index } = this.kit;
    return headingOf(this.wpos[index.RightLeg], this.wpos[index.LeftLeg]);
  }

  headWorld(target) {
    this.group.updateMatrixWorld(true);
    return this.bones[this.kit.headIndex].getWorldPosition(target).add(_p.set(0, 0.28, 0));
  }

  /**
   * Distance along the ray to the skeleton's bounding box, or Infinity.
   * The box is turned to the pose's facing so it stays snug when the body rotates,
   * and padded to cover the flesh around the joints and the top of the head.
   */
  hitTest(ray) {
    if (!this.group.visible) return Infinity;
    const yaw = this.facing();
    const c = Math.cos(yaw), s = Math.sin(yaw), o = this.wpos[0];
    _box.makeEmpty();
    for (const p of this.wpos) { // into a frame centred on the hips, facing +Z
      const dx = p.x - o.x, dz = p.z - o.z;
      _box.expandByPoint(_p.set(c * dx - s * dz, p.y, s * dx + c * dz));
    }
    _box.min.x -= BOX_PAD; _box.max.x += BOX_PAD;
    _box.min.z -= BOX_PAD; _box.max.z += BOX_PAD;
    _box.min.y = Math.max(0, _box.min.y - 0.05); _box.max.y += HEAD_TOP;
    _local.compose(_p.set(o.x, 0, o.z), _yq.setFromAxisAngle(Y, yaw), _one).invert();
    _ray.copy(ray).applyMatrix4(_local);
    return _ray.intersectBox(_box, _hit) ? _hit.distanceTo(_ray.origin) : Infinity;
  }


  /**
   * World-space box round the body (the model's bones, padded for flesh and the top of the
   * head). It starts a little above the lowest bone, so a flat shape on the floor doesn't count.
   */
  bounds(target) {
    this.group.updateMatrixWorld(true);
    target.makeEmpty();
    for (const b of this.bones) target.expandByPoint(b.getWorldPosition(_hit));
    target.min.x -= BOX_PAD; target.max.x += BOX_PAD;
    target.min.z -= BOX_PAD; target.max.z += BOX_PAD;
    target.max.y += HEAD_TOP * 0.5;
    target.min.y += 0.03;
    return target;
  }

  set visible(v) { this.group.visible = v; }
  get visible() { return this.group.visible; }
}
