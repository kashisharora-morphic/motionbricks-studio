"""MotionBricks Studio: local HTTP server for the keyframe editor.

  .venv/bin/python server/server.py [--port 8765] [--motionbricks ../motionbricks]

Serves the web app and two JSON endpoints:
  GET  /api/library   skeleton + pose library (key poses for display)
  POST /api/segment   generate one transition between two keys
"""

import argparse
import base64
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MAX_TOKENS, MIN_TOKENS, SomaEngine  # noqa: E402
from library import PoseLibrary  # noqa: E402
from quat import qmul, qrot, slerp, yaw_q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
VENDOR = ROOT / "web" / "node_modules"

# Retiming: MotionBricks picks a duration of 17..57 frames per segment (6..16 tokens).
# We steer it toward the user's gap and then retime within these limits.
MAX_STRETCH = 1.35  # play up to 35% slower before holding the final pose
MAX_SQUASH = 0.75  # pick token counts no more than ~25% longer than the gap


def natural_frames(tokens):
    return tokens * 4 - 7  # valid frames - 1 (span between endpoints)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()


def unb64(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).astype(np.float64).reshape(shape)


class Studio:
    def __init__(self, mb_root: Path):
        t0 = time.perf_counter()
        self.lib = PoseLibrary(mb_root / "assets/motions/soma")
        self.engine = SomaEngine(mb_root / "motionbricks/assets/onnx/soma", self.lib.skeleton)
        self.lock = threading.Lock()
        skel = self.lib.skeleton
        # Joints the model drives through a connected hierarchy (no fingers).
        self.render_joints = [j for j in range(skel.num_joints) if self.engine.pose_to_model[j] >= 0]
        remap = {j: i for i, j in enumerate(self.render_joints)}
        self.render_parents = [remap.get(int(skel.parents[j]), -1) for j in self.render_joints]
        print(f"[studio] ready in {time.perf_counter() - t0:.1f}s, {len(self.lib.poses)} poses")

    def library_json(self):
        skel = self.lib.skeleton
        poses = []
        for pd in self.lib.poses.values():
            pos, rot = self.lib.window(pd.id, (0.0, 0.0, 0.0, 0.0), anchor=0, count=1)
            poses.append({
                "id": pd.id,
                "label": pd.label,
                "group": pd.group,
                "posesToUse": pd.poses_to_use,
                "root": pos[0, 0].tolist(),
                "rot": rot[0, self.render_joints].round(5).tolist(),
            })
        return {
            "fps": 30,
            "skeleton": {
                "names": [skel.names[j] for j in self.render_joints],
                "parents": self.render_parents,
                "offsets": skel.ref_pos[self.render_joints].round(5).tolist(),
            },
            "poses": poses,
            "limits": {"minFrames": natural_frames(MIN_TOKENS), "maxFrames": natural_frames(MAX_TOKENS),
                       "maxStretch": MAX_STRETCH},
        }

    def window(self, key, placement, anchor, count=4):
        """Model input poses for a key: a library pose, or one copied from generated motion."""
        if key.get("custom"):
            return self.custom_window(key["custom"], placement, anchor, count)
        return self.lib.window(key["pose"], placement, anchor=anchor, count=count)

    def custom_window(self, custom, placement, anchor, count):
        """Rebuild full-skeleton poses from a copied in-between.

        The browser stores the in-between frame and its neighbours for the joints the model
        drives, relative to the pose's own position and facing. Joints the model doesn't
        drive (fingers, end sites) keep their rest values, exactly as in generated motion.
        """
        frames, a = custom["frames"], int(custom["anchor"])
        skel, R = self.lib.skeleton, self.render_joints
        x, y, z, yaw = placement[:4]
        yq, base = yaw_q(yaw), np.array([x, y, z])
        J = skel.num_joints
        pos = np.broadcast_to(skel.ref_pos, (count, J, 3)).copy()
        rot = np.zeros((count, J, 4))
        rot[..., 3] = 1.0
        for f in range(count):
            src = frames[min(max(a + f - anchor, 0), len(frames) - 1)]
            rot[f, R] = np.asarray(src["rot"], dtype=np.float64).reshape(-1, 4)
            rot[f, 0] = qmul(yq, rot[f, 0])
            pos[f, 0] = base + qrot(yq, np.asarray(src["root"], dtype=np.float64))
        return pos, rot

    def segment(self, req):
        a, b = req["from"], req["to"]
        gap = int(b["frame"]) - int(a["frame"])
        if gap < 1:
            raise ValueError("Keys must be at least one frame apart.")
        J = self.lib.skeleton.num_joints

        def placement(k):
            return (float(k["x"]), float(k["y"]), float(k["z"]), float(k["yaw"]), float(k.get("pitch", 0.0)))

        if req.get("ctx"):
            ctx = unb64(req["ctx"], (4, J, 7))
            cp, cr = ctx[..., :3], ctx[..., 3:]
        else:
            cp, cr = self.window(a, placement(a), anchor=3)
        n_tgt = int(b.get("posesToUse") or self.lib.poses[b["pose"]].poses_to_use)
        tp, tr = self.window(b, placement(b), anchor=0, count=n_tgt)

        lo, hi = gap / MAX_STRETCH, gap / MAX_SQUASH
        allowed = [t for t in range(MIN_TOKENS, MAX_TOKENS + 1) if lo <= natural_frames(t) <= hi]
        if not allowed:
            allowed = [MIN_TOKENS] if gap < natural_frames(MIN_TOKENS) else [MAX_TOKENS]

        with self.lock:
            pos, rot, contacts, tokens, timings = self.engine.generate(cp, cr, tp, tr, allowed)

        natural = pos.shape[0] - 1
        span = min(gap, int(round(natural * MAX_STRETCH)))
        # Resample [0, natural] onto [0, span] then hold the last pose to fill the gap.
        s = np.arange(span + 1) * (natural / span)
        i0 = np.clip(np.floor(s).astype(int), 0, natural)
        i1 = np.minimum(i0 + 1, natural)
        w = (s - i0)
        out_pos = pos[i0] * (1 - w)[:, None, None] + pos[i1] * w[:, None, None]
        out_rot = slerp(rot[i0], rot[i1], np.broadcast_to(w[:, None], (len(w), rot.shape[1])))
        held = gap - span
        if held > 0:
            out_pos = np.concatenate([out_pos, np.repeat(out_pos[-1:], held, 0)], 0)
            out_rot = np.concatenate([out_rot, np.repeat(out_rot[-1:], held, 0)], 0)

        # The next segment's context is the last four poses that actually play.
        last4 = np.minimum(np.arange(gap - 3, gap + 1), gap).clip(0)
        ctx_out = np.concatenate([out_pos[last4], out_rot[last4]], -1)

        R = self.render_joints
        return {
            "frames": gap + 1,
            "tokens": tokens,
            "naturalFrames": natural,
            "heldFrames": held,
            "speed": natural / span,
            "root": b64(out_pos[:, 0]),
            "rot": b64(out_rot[:, R]),
            "ctx": b64(ctx_out),
            "timings": {k: round(v, 1) for k, v in timings.items()},
        }


class Handler(BaseHTTPRequestHandler):
    studio: Studio = None

    def log_message(self, fmt, *args):
        if "/api/" in (str(args[0]) if args else ""):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/library":
            return self._send(200, self.studio.library_json())
        if path == "/":
            path = "/index.html"
        if path.startswith("/vendor/"):
            base, rel = VENDOR, path[len("/vendor/"):]
        else:
            base, rel = WEB, path.lstrip("/")
        file = (base / rel).resolve()
        if not str(file).startswith(str(base.resolve())) or not file.is_file() or file.name.startswith("._"):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if file.suffix == ".js":
            ctype = "text/javascript"
        return self._send(200, file.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/segment":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            return self._send(200, self.studio.segment(req))
        except Exception as exc:  # report to the UI
            import traceback

            traceback.print_exc()
            return self._send(400, {"error": str(exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--motionbricks", type=Path, default=ROOT.parent / "motionbricks")
    args = ap.parse_args()
    Handler.studio = Studio(args.motionbricks.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[studio] http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
