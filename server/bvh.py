# Written to match motionbricks.io.bvh from NVlabs/motionbricks,
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0
# (see licenses/Apache-2.0.txt). Modified. See NOTICE.
"""Minimal numpy BVH loader matching motionbricks.io.bvh.

The SOMA BVH files have a static `Root` joint above `Hips`. The MotionBricks apps
retarget clips onto a skeleton whose root is Hips, copying each joint's local
transform by name, so the Root joint is dropped here the same way.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quat import axis_angle, qmul, qnorm, slerp


@dataclass
class Skeleton:
    names: list
    parents: np.ndarray  # [J] int, -1 for root
    ref_pos: np.ndarray  # [J, 3] reference local offsets (m)

    @property
    def num_joints(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)


@dataclass
class Clip:
    path: Path
    skeleton: Skeleton
    rate: float
    pos: np.ndarray  # [F, J, 3] local positions
    rot: np.ndarray  # [F, J, 4] local rotations (xyzw)

    @property
    def num_frames(self):
        return self.pos.shape[0]

    @property
    def duration(self):
        return (self.num_frames - 1) / self.rate

    def sample(self, t):
        """Local pose at time t (seconds), clamped, as (pos [J,3], rot [J,4])."""
        f = t * self.rate
        last = self.num_frames - 1
        i0 = int(np.clip(np.floor(f), 0, last))
        blend = float(np.clip(f - i0, 0.0, 1.0)) if i0 < last else 0.0
        i1 = min(i0 + 1, last)
        p = self.pos[i0] * (1 - blend) + self.pos[i1] * blend
        q = slerp(self.rot[i0], self.rot[i1], np.full(self.rot.shape[1], blend))
        return p, q


def load_bvh(path):
    path = Path(path)
    names, parents, offsets, channels = [], [], [], []
    stack, in_end = [], False
    with path.open(encoding="utf-8") as src:
        for line in src:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "MOTION":
                break
            if tok[0] in ("ROOT", "JOINT"):
                names.append(tok[1].rsplit(":", 1)[-1])
                parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1)
                offsets.append([0.0, 0.0, 0.0])
                channels.append([])
            elif tok[:2] == ["End", "Site"]:
                in_end = True
            elif tok[0] == "OFFSET" and not in_end:
                offsets[stack[-1]] = [float(v) for v in tok[1:4]]
            elif tok[0] == "CHANNELS":
                channels[stack[-1]] = tok[2:]
            elif tok[0] == "}":
                if in_end:
                    in_end = False
                else:
                    stack.pop()
        rate = None
        for line in src:
            tok = line.split()
            if tok[:2] == ["Frame", "Time:"]:
                rate = round(1.0 / float(tok[2]), 2)
                break
        data = np.loadtxt(src, dtype=np.float64, ndmin=2)

    J = len(names)
    F = data.shape[0]
    ref = np.asarray(offsets, dtype=np.float64) * 0.01
    pos = np.broadcast_to(ref, (F, J, 3)).copy()
    rot = np.zeros((F, J, 4))
    rot[..., 3] = 1.0
    col = 0
    for j, chans in enumerate(channels):
        q = None
        for ch in chans:
            ch = ch.lower()
            axis = "xyz".index(ch[0])
            if ch.endswith("position"):
                pos[:, j, axis] = data[:, col] * 0.01
            elif ch.endswith("rotation"):
                ax = np.zeros(3)
                ax[axis] = 1.0
                r = qnorm(axis_angle(ax, np.radians(data[:, col])))
                q = r if q is None else qmul(q, r)
            col += 1
        if q is not None:
            rot[:, j] = qnorm(q)

    parents = np.asarray(parents, dtype=np.int64)
    if names[0] == "Root":
        names = names[1:]
        parents = parents[1:] - 1
        ref, pos, rot = ref[1:], pos[:, 1:], rot[:, 1:]
        parents[0] = -1
    skel = Skeleton(names, parents, ref)
    return Clip(path, skel, rate, pos, rot)


def remap_clip(clip, skeleton):
    """Remap a clip onto another skeleton by joint name (missing joints keep reference)."""
    F, J = clip.num_frames, skeleton.num_joints
    pos = np.broadcast_to(skeleton.ref_pos, (F, J, 3)).copy()
    rot = np.zeros((F, J, 4))
    rot[..., 3] = 1.0
    for src_j, name in enumerate(clip.skeleton.names):
        if name in skeleton.names:
            dst = skeleton.index(name)
            pos[:, dst] = clip.pos[:, src_j]
            rot[:, dst] = clip.rot[:, src_j]
    return Clip(clip.path, skeleton, clip.rate, pos, rot)


def forward_kinematics(parents, pos, rot):
    """Local -> world for [..., J, 3/4] arrays. Parents precede children."""
    from quat import qrot

    wp = np.empty_like(pos)
    wq = np.empty_like(rot)
    wp[..., 0, :] = pos[..., 0, :]
    wq[..., 0, :] = rot[..., 0, :]
    for j in range(1, len(parents)):
        p = parents[j]
        wp[..., j, :] = wp[..., p, :] + qrot(wq[..., p, :], pos[..., j, :])
        wq[..., j, :] = qmul(wq[..., p, :], rot[..., j, :])
    return wp, wq
