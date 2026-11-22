"""Pose library sampled from the SOMA sample clips, and key-pose windows for the model.

Every pose is grounded: its lowest joint sits at the same height as in the standing
pose, so all poses start on the floor and the editor's vertical gizmo sets any lift.

The walk comes as two poses, one per stride: Walk L (left foot ahead) and Walk R.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bvh import load_bvh, remap_clip, forward_kinematics
from quat import axis_angle, heading_from_hips, qmul, qrot, yaw_q

SAMPLE_RATE = 30.0


# id, label, clip, time (s), target poses to use[, tip (degrees), clearance (m)]
# poses_to_use: 1 = arrive and stop there; 4 = pass through with that motion's momentum.
# tip lays a clip's pose over about its side axis (+90 = face down); clearance raises it
# off the floor by the body's thickness, since grounding goes by joint centres.
IDLE_CLIP = "locomotion/idle/idle_loop_001__A268.bvh"
WALK_CLIP = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh"
WALK_TIME = 4.0  # the two stride poses are the extremes in the second after this
POSES = [
    ("stand", "Stand", IDLE_CLIP, 2.0, 1),
    ("walk", None, WALK_CLIP, WALK_TIME, 4),  # becomes walk_l / walk_r
    ("crouch", "Crouch", "locomotion/idle/crouch_idle_001__A529.bvh", 2.0, 1),
    ("jump", "Jump", "jump/jump_over_obstacle_0_5m_R_002__A437.bvh", 2.18, 4),
    ("sit", "Sit", "sitting/sit_on_chair_loop.bvh", 2.0, 1),
    ("lie_up", "Lie Face Up", IDLE_CLIP, 2.0, 1, -90.0, 0.067),
    ("lie_down", "Lie Face Down", IDLE_CLIP, 2.0, 1, 90.0, 0.024),
]

CANONICAL_CLIP = IDLE_CLIP
# Joints used for grounding: the body joints the editor draws (no fingers or end sites),
# so the server and the browser agree on where the floor is.
GROUND_JOINTS = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)


@dataclass
class PoseDef:
    id: str
    label: str
    group: str
    clip: object
    time: float
    poses_to_use: int
    heading: float  # facing angle of the clip at `time` (0 = +Z)
    lift: float = 0.0  # subtracted from the root height so the pose stands on the floor
    tip: float = 0.0  # radians about the side axis, laid down before any key's own pitch
    clear: float = 0.0  # extra height for a laid-down pose


class PoseLibrary:
    def __init__(self, motions_root: Path):
        self.root = Path(motions_root)
        canonical = load_bvh(self.root / CANONICAL_CLIP)
        self.skeleton = canonical.skeleton
        clips = {}
        self.poses = {}
        for pid, label, rel, t, n, *lay in POSES:
            if rel not in clips:
                clip = remap_clip(load_bvh(self.root / rel), self.skeleton)
                # Use canonical bone lengths everywhere so poses and generated motion agree.
                clip.pos[:, 1:] = self.skeleton.ref_pos[1:]
                clips[rel] = clip
            clip = clips[rel]
            p, q = clip.sample(t)
            wp, _ = forward_kinematics(self.skeleton.parents, p, q)
            heading = float(heading_from_hips(wp[self.skeleton.index("RightLeg")], wp[self.skeleton.index("LeftLeg")]))
            pd = PoseDef(pid, label, "", clip, t, n, heading)
            if lay:
                pd.tip, pd.clear = float(np.radians(lay[0])), float(lay[1])
            if pid == "walk":  # the walk's two stride extremes: left / right foot ahead
                self.poses["walk_l"] = self._contact(pd, "walk_l", "Walk L", 1.0)
                self.poses["walk_r"] = self._contact(pd, "walk_r", "Walk R", -1.0)
            else:
                self.poses[pid] = pd
        # Ground every pose to the standing pose's floor contact.
        self.ground_idx = [self.skeleton.index(n) for n in GROUND_JOINTS]
        self.floor = self._lowest("stand")
        for pd in self.poses.values():
            pd.lift = self._lowest(pd.id) - self.floor

    def _foot_lead(self, clip, t):
        """How far the left foot is ahead of the right along the facing, and that facing."""
        p, q = clip.sample(t)
        wp, _ = forward_kinematics(self.skeleton.parents, p, q)
        ix = self.skeleton.index
        h = float(heading_from_hips(wp[ix("RightLeg")], wp[ix("LeftLeg")]))
        fwd = np.array([np.sin(h), 0.0, np.cos(h)])
        return float(np.dot(wp[ix("LeftFoot")] - wp[ix("RightFoot")], fwd)), h

    def _contact(self, walk, pid, label, sign):
        """The walk clip's stride extreme after the walk pose: left (+1) or right (-1) foot ahead."""
        best = None
        for t in np.arange(walk.time, walk.time + 1.1, 1.0 / SAMPLE_RATE):
            lead, h = self._foot_lead(walk.clip, float(t))
            if best is None or sign * lead > best[0]:
                best = (sign * lead, float(t), h)
        _, t, h = best
        return PoseDef(pid, label, "", walk.clip, t, walk.poses_to_use, h)

    def _sample(self, pd, t):
        p, q = pd.clip.sample(t)
        return p.copy(), q.copy()

    def _lowest(self, pose_id):
        pd = self.poses[pose_id]
        p, q = self._sample(pd, pd.time)
        wp, _ = forward_kinematics(self.skeleton.parents, p, q)
        return float(wp[self.ground_idx, 1].min())

    def window(self, pose_id, placement, anchor, count=4):
        """Sample `count` poses at 30 Hz with the anchor pose at the placement.

        placement = (x, y, z, yaw[, pitch]): the anchor's root moves to (x, z) and turns
        to face `yaw` (0 = +Z). `pitch` tips the body about its side axis (a flip), with
        the hips as the pivot. The pose is then grounded so its lowest joint is on the
        floor, and raised by y. The other samples move rigidly with the anchor, so the
        motion between them is preserved.
        """
        pd = self.poses[pose_id]
        x, y, z, yaw = placement[:4]
        pitch = (float(placement[4]) if len(placement) > 4 else 0.0) + pd.tip
        ap, _ = self._sample(pd, pd.time)
        off_q = qmul(yaw_q(yaw), qmul(axis_angle([1.0, 0.0, 0.0], pitch), yaw_q(-pd.heading)))
        hips = ap[0]
        target = np.array([x, hips[1] - pd.lift, z])  # grounded hips, before any flip
        J = self.skeleton.num_joints
        pos = np.empty((count, J, 3))
        rot = np.empty((count, J, 4))
        for f in range(count):
            p, q = self._sample(pd, pd.time + (f - anchor) / SAMPLE_RATE)
            p[0] = target + qrot(off_q, p[0] - hips)
            q[0] = qmul(off_q, q[0])
            pos[f], rot[f] = p, q
        shift = 0.0
        if pitch:  # re-ground the flipped anchor pose
            wp, _ = forward_kinematics(self.skeleton.parents, pos[anchor], rot[anchor])
            shift = self.floor - float(wp[self.ground_idx, 1].min()) + pd.clear
        pos[:, 0, 1] += shift + y
        return pos, rot
