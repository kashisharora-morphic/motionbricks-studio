# MotionBricks Studio

A minimal local keyframe editor for [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks):
key a few poses on a timeline, press **Generate**, and MotionBricks synthesizes the motion between them.

```bash
./setup.sh   # once: clones motionbricks next to this folder, fetches LFS assets, makes .venv
./run.sh     # then open http://127.0.0.1:8765
```

## How it works

- **Runs on any machine, Apple Silicon included.** The official runtime needs TensorRT and an NVIDIA GPU.
  `server/engine.py` is a numpy port of the SOMA engine's encode, decode and endpoint-correction
  kernels (`motionbricks/model/soma/kernels.py`) around the bundled ONNX model on ONNX Runtime CPU.
  One generation takes about 20 ms on an M5 Pro.
- **Pose library.** The poses are frames picked from the sample BVH clips (`server/library.py`): stand,
  look, guard, crouch, sit, walk, jog, run, crouch walk, skip, zombie, injured, jump takeoff/air/high/land, duck.
  Moving poses pass four target frames to the model, so it carries their momentum. Idle poses pass one, so it stops there.
- **Only keys are saved** (browser localStorage). Each segment between two keys is generated on demand and
  cached by a signature of every key up to its end, because each segment's context is the end of the previous one.
  Change a key and the downstream segments go **Out of sync**. Append keys and you get **Generate remaining**.
  Generate only reruns from the first invalid segment.
- **Mannequin.** `web/models/Xbot.glb` is the Mixamo X Bot as shipped in the three.js examples
  (`examples/models/gltf/Xbot.glb`). `web/js/mannequin.js` retargets SOMA poses onto it. It pairs SOMA's
  *stand* pose with the X Bot's *idle* clip to get one fixed offset per bone, then aligns limb directions exactly every frame.
  A ground clamp lifts the body whenever a heel, toe or knee would go below the floor. Mixamo characters
  can be used in your own projects but not redistributed as standalone assets, so swap in your own model before shipping this.
- **Timing.** The model picks 17–57 frames per segment. The server limits it to token counts near your key spacing,
  then retimes the result to fit exactly. If the gap is too long, it holds the final pose (shown darker on the timeline).

## Controls

| | |
|---|---|
| Pick a time | Click/drag the timeline |
| Pick a pose | Scroll while hovering the mannequin (or an orange key ghost) |
| Place | Drag the mannequin on the floor |
| Gizmo | Click the mannequin: red/blue arrows move, green arrow lifts, yellow ring turns (Shift snaps 15°) |
| Key | **Key Pose** (K / Enter). On a keyed frame the button removes the key |
| Retime | Drag a key diamond on the timeline |
| Play | Space · ←/→ step · ↑/↓ cycle pose · G generate |
