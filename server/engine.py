"""MotionBricks SOMA inference on CPU with ONNX Runtime.

A numpy port of motionbricks/model/soma/{engine,kernels}.py so the bundled ONNX
model runs without TensorRT/CUDA (e.g. on Apple Silicon). World space here is
SOMA model space: Y up, meters. That matches three.js, so model_to_runtime is the
identity.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from quat import (
    heading_from_hips,
    qinv,
    qmul,
    qnorm,
    qrot,
    qrot_inv,
    rotation_from_6d,
    slerp,
    tinv,
    tmul,
    yaw_q,
)
from bvh import forward_kinematics

POSES_PER_TOKEN = 4
SAMPLE_RATE = 30.0
MIN_TOKENS, MAX_TOKENS = 6, 16
BOUNDARY = POSES_PER_TOKEN - 1
FOOT_CONTACTS = 4


def valid_frames(tokens):
    return tokens * POSES_PER_TOKEN - 2 * BOUNDARY


class SomaEngine:
    def __init__(self, onnx_dir: Path, skeleton, precision="fp32"):
        manifest = json.loads((onnx_dir / "soma_n1.json").read_text())
        self.model_joints = manifest["joint_names"]
        fname = manifest["onnx_filename"] if precision == "fp32" else manifest["precision_variants"]["fp16"]["onnx_filename"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_dir / fname), opts, providers=["CPUExecutionProvider"])
        self.load_seconds = time.perf_counter() - t0
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.skel = skeleton
        self.model_to_pose = np.array([skeleton.index(n) for n in self.model_joints])
        assert self.model_to_pose[0] == 0, "Skeleton root must be Hips"
        self.pose_to_model = np.full(skeleton.num_joints, -1)
        for mj, pj in enumerate(self.model_to_pose):
            parent = skeleton.parents[pj]
            if parent == -1 or self.pose_to_model[parent] >= 0:
                self.pose_to_model[pj] = mj
        self.right_leg = skeleton.index("RightLeg")
        self.left_leg = skeleton.index("LeftLeg")

    # ------------------------------------------------------------------ encode
    def _alignment(self, wpos):
        last = POSES_PER_TOKEN - 1
        hips = wpos[last, 0]
        current = heading_from_hips(wpos[last, self.right_leg], wpos[last, self.left_leg])
        desired = np.arctan2(1.0, 0.0)  # heading_alignment (1, 0) -> face +X
        return np.array([hips[0], 0.0, hips[2]]), yaw_q(current - desired)

    def _encode(self, guide_pos, guide_rot, num_targets, allowed_tokens):
        wpos, wrot = forward_kinematics(self.skel.parents, guide_pos, guide_rot)  # [8, J, *]
        align_p, align_q = self._alignment(wpos)
        n_model = len(self.model_joints)

        def aligned(p):
            return qrot_inv(align_q, p - align_p)

        frames = np.arange(8)
        nxt = np.where(frames == POSES_PER_TOKEN - 1, frames, np.minimum(frames + 1, 7))

        hips = aligned(wpos[:, 0])
        right = aligned(wpos[:, self.right_leg])
        left = aligned(wpos[:, self.left_leg])
        heading = heading_from_hips(right, left)
        dh = heading[nxt] - heading
        dh = np.arctan2(np.sin(dh), np.cos(dh))

        g = np.zeros((1, 8, 5), np.float32)
        g[0, :, :3] = hips
        g[0, :, 3] = np.cos(heading)
        g[0, :, 4] = np.sin(heading)

        l = np.zeros((1, 8, 4), np.float32)
        l[0, :, 0] = dh * SAMPLE_RATE
        l[0, :, 1] = (hips[nxt, 0] - hips[:, 0]) * SAMPLE_RATE
        l[0, :, 2] = (hips[nxt, 2] - hips[:, 2]) * SAMPLE_RATE
        l[0, :, 3] = hips[:, 1]

        mp = wpos[:, self.model_to_pose]  # [8, M, 3]
        mq = wrot[:, self.model_to_pose]
        jp = aligned(mp)
        jq = qmul(np.broadcast_to(qinv(align_q), mq.shape), mq)
        root_ground = np.stack([hips[:, 0], np.zeros(8), hips[:, 2]], -1)[:, None]
        centered = (jp - root_ground)[:, 1:].reshape(8, -1)
        xa = qrot(jq, np.broadcast_to([1.0, 0.0, 0.0], jp.shape))
        ya = qrot(jq, np.broadcast_to([0.0, 1.0, 0.0], jp.shape))
        rot6 = np.concatenate([xa, ya], -1).reshape(8, -1)
        lp = np.concatenate([centered, rot6], -1)[None].astype(np.float32)
        assert lp.shape[-1] == (n_model - 1) * 3 + n_model * 6

        idx = np.arange(8)
        given = (idx < POSES_PER_TOKEN + num_targets).astype(np.int32)
        has_local_root = given.copy()
        has_local_root[POSES_PER_TOKEN - 1] = 0
        has_local_root[POSES_PER_TOKEN + num_targets - 1] = 0
        allowed = np.zeros((1, MAX_TOKENS - MIN_TOKENS + 1), np.int32)
        for t in allowed_tokens:
            allowed[0, t - MIN_TOKENS] = 1

        inputs = {
            "global_root_values": g,
            "has_global_root_values": given[None],
            "local_root_values": l,
            "has_local_root_values": has_local_root[None],
            "local_poses": lp,
            "has_local_poses": given[None],
            "allowed_pred_num_tokens": allowed,
        }
        return inputs, (align_p, align_q)

    # ------------------------------------------------------------------ decode
    def _decode(self, pred, tokens, align):
        n_model = len(self.model_joints)
        J = self.skel.num_joints
        count = valid_frames(tokens)
        raw = pred[0, BOUNDARY : BOUNDARY + count]  # [F, 366]
        contacts = np.clip(raw[:, -FOOT_CONTACTS:], 0.0, 1.0)
        rot_off = 5 + (n_model - 1) * 3
        r6 = raw[:, rot_off : rot_off + n_model * 6].reshape(count, n_model, 6).astype(np.float64)
        model_rot = rotation_from_6d(r6[..., :3], r6[..., 3:])  # [F, M, 4] in aligned space

        pos = np.broadcast_to(self.skel.ref_pos, (count, J, 3)).copy()
        rot = np.zeros((count, J, 4))
        rot[..., 3] = 1.0
        hp, hq = tmul(
            (np.broadcast_to(align[0], (count, 3)), np.broadcast_to(align[1], (count, 4))),
            (raw[:, :3].astype(np.float64), model_rot[:, 0]),
        )
        pos[:, 0] = hp
        rot[:, 0] = hq
        for j in range(1, J):
            mj = self.pose_to_model[j]
            if mj < 0:
                continue
            pmj = self.pose_to_model[self.skel.parents[j]]
            rot[:, j] = qnorm(qmul(qinv(model_rot[:, pmj]), model_rot[:, mj]))
        return pos, rot, contacts

    @staticmethod
    def _endpoint_correct(pos, rot, ctx_last, tgt_first):
        F = pos.shape[0]
        if F < 2:
            return pos, rot
        w_end = (np.arange(F) / (F - 1))[:, None]
        w_start = 1.0 - w_end
        cp, cq = ctx_last
        tp, tq = tgt_first

        # Root: world-space position offsets, rotation applied after.
        s_rot = qnorm(qmul(qinv(qnorm(rot[0, 0])), qnorm(cq[0])))
        e_rot = qnorm(qmul(qinv(qnorm(rot[-1, 0])), qnorm(tq[0])))
        s_pos = cp[0] - pos[0, 0]
        e_pos = tp[0] - pos[-1, 0]
        ident = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (F, 4))
        rq = qmul(qmul(rot[:, 0], slerp(ident, np.broadcast_to(s_rot, (F, 4)), w_start[:, 0])),
                  slerp(ident, np.broadcast_to(e_rot, (F, 4)), w_end[:, 0]))
        new_root_p = pos[:, 0] + s_pos * w_start + e_pos * w_end
        new_root_q = qnorm(rq)

        # Joints: local deltas, start fades out while end fades in.
        sd = tmul(tinv((pos[0, 1:], rot[0, 1:])), (cp[1:], cq[1:]))
        ed = tmul(tinv((pos[-1, 1:], rot[-1, 1:])), (tp[1:], tq[1:]))
        J1 = pos.shape[1] - 1
        idq = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (F, J1, 4))
        ws = np.broadcast_to(w_start, (F, J1))
        we = np.broadcast_to(w_end, (F, J1))
        d1 = (sd[0][None] * ws[..., None], slerp(idq, np.broadcast_to(sd[1], (F, J1, 4)), ws))
        p1, q1 = tmul((pos[:, 1:], rot[:, 1:]), d1)
        d2 = (ed[0][None] * we[..., None], slerp(idq, np.broadcast_to(ed[1], (F, J1, 4)), we))
        p2, q2 = tmul((p1, q1), d2)

        pos = pos.copy()
        rot = rot.copy()
        pos[:, 0], rot[:, 0] = new_root_p, new_root_q
        pos[:, 1:], rot[:, 1:] = p2, q2
        return pos, rot

    # ---------------------------------------------------------------- generate
    def generate(self, ctx_pos, ctx_rot, tgt_pos, tgt_rot, allowed_tokens):
        """Generate one transition.

        ctx_*: [4, J, *] local transforms (root in world). tgt_*: [1..4, J, *].
        Returns (pos [F,J,3], rot [F,J,4], contacts [F,4], tokens, timings).
        """
        nt = tgt_pos.shape[0]
        pad = np.minimum(np.arange(4), nt - 1)
        guide_pos = np.concatenate([ctx_pos, tgt_pos[pad]], 0)
        guide_rot = np.concatenate([ctx_rot, tgt_rot[pad]], 0)

        t0 = time.perf_counter()
        inputs, align = self._encode(guide_pos, guide_rot, nt, allowed_tokens)
        t1 = time.perf_counter()
        out = dict(zip(self.output_names, self.session.run(None, inputs)))
        t2 = time.perf_counter()
        tokens = int(out["pred_num_tokens"][0])
        pos, rot, contacts = self._decode(out["pred_global_motions"], tokens, align)
        pos, rot = self._endpoint_correct(
            pos, rot, (ctx_pos[-1], ctx_rot[-1]), (tgt_pos[0], tgt_rot[0])
        )
        t3 = time.perf_counter()
        timings = {"encode_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3, "decode_ms": (t3 - t2) * 1e3}
        return pos, rot, contacts, tokens, timings
