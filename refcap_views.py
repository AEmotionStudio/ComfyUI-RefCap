"""View selection for RefCap view sets (refcap_nodes.py). It imports nothing from ComfyUI, so scripts outside ComfyUI
can use it too. A yaw is the camera's azimuth around the subject in degrees; 0 is the front.
"""

import re

import numpy as np

# PIL-readable subset of TURNtable Viewer's filter (it also takes .exr)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp")
RELIABLE_YAW = 75.0  # insightface's measured yaw saturates beyond this (orbit frames asked for 120-135 read 78-89)


def natural_key(name):
    # TURNtable Viewer's natural_sort_key, so a folder orders the same in Blender and here
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def wrap(deg):
    # -> (-180, 180]
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def parse_list(text, cast):
    return [cast(t) for t in text.replace(";", ",").split(",") if t.strip()]


def turntable_yaws(n, frame_offset_degrees=0.0, reverse_direction=False):
    # TURNtable Viewer's driver shows frame i across [i, i+1) * 360/N past its offset; invert it
    sign = -1.0 if reverse_direction else 1.0
    return [wrap(sign * (i * 360.0 / n + frame_offset_degrees)) for i in range(n)]


def select_views(yaws, select, target_yaws="", count=0, indices=""):
    """-> sorted view indices. nearest: each target yaw in turn takes its nearest unused view, round after round, until
    `count` views (default one per target) are chosen."""
    n = len(yaws)
    if select == "all":
        return list(range(n))
    if select == "indices":
        chosen = sorted(set(parse_list(indices, int)))
        bad = [i for i in chosen if not 0 <= i < n]
        if bad:
            raise ValueError(f"view indices {bad} are outside 0..{n - 1}")
        return chosen
    targets = [wrap(t) for t in parse_list(target_yaws, float)]
    if not targets:
        raise ValueError("select=nearest needs target_yaws")
    want = min(n, count or len(targets))
    chosen = []
    while len(chosen) < want:
        for t in targets[:want - len(chosen)]:
            rest = [i for i in range(n) if i not in chosen]
            chosen.append(min(rest, key=lambda i: (abs(wrap(yaws[i] - t)), i)))
    return sorted(chosen)


def view_phrase(yaw):
    """Side-neutral words for a view's angle, for prompts."""
    a = abs(wrap(yaw))
    return ("front view" if a < 15 else "three-quarter view" if a < 60 else "side view" if a < 120
            else "three-quarter back view" if a < 165 else "back view")


def orbit_angles(measured):
    """measured: each frame's insightface yaw (None without a face) in an orbit clip that runs from the front (frame 0)
    to directly behind (the last frame) -> each frame's |yaw|. H3's orbits ease in, so frame index is a poor guide
    (asked 30 degrees, measured 5). Where insightface is reliable its measured yaw is used; beyond that a straight line
    runs from the last reliable frame to 180 at the final frame, which the orbit prompts end on (checked by eye: a
    clean back view with no face)."""
    xs, ys = [0], [0.0]
    for k, yaw in enumerate(measured[1:], 1):
        if yaw is not None and abs(yaw) <= RELIABLE_YAW:
            xs.append(k)
            ys.append(abs(yaw))
    xs.append(len(measured) - 1)
    ys.append(180.0)
    return np.interp(np.arange(len(measured)), xs, np.maximum.accumulate(ys))


def pan_angles(frames):
    """frames: [F, H, W, 3] uint8 of a camera pan that turns from 0 (frame 0) to 180 degrees (the last frame), as the
    location pan prompts ask -> each frame's heading. H3's pans ease in like its orbits, so the heading follows the
    image's sideways motion instead of the frame index: phase correlation between neighbouring frames, summed and
    scaled to 180 at the last frame."""
    gray = frames.astype(np.float32).mean(axis=3)
    gray -= gray.mean(axis=(1, 2), keepdims=True)
    spectra = np.fft.rfft2(gray * np.hanning(gray.shape[2]))
    shifts = [0.0]
    for a, b in zip(spectra[:-1], spectra[1:]):
        cross = a * np.conj(b)
        corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=gray.shape[1:])
        dx = int(np.unravel_index(np.argmax(corr), corr.shape)[1])
        shifts.append(float(abs(dx - gray.shape[2] if dx > gray.shape[2] // 2 else dx)))
    total = np.cumsum(shifts)
    if total[-1] == 0:
        raise ValueError("the pan doesn't move: no sideways motion between its frames")
    return 180.0 * total / total[-1]


def orbit_targets(text=""):
    """Comma-separated angles to take from a pair of orbits; empty: every 15 degrees round the head."""
    if text.strip():
        return parse_list(text, float)
    return [0.0] + [s * y for y in range(15, 181, 15) for s in (-1, 1) if not (y == 180 and s == 1)]


def pick_orbit_frames(angles, targets):
    """angles: {-1: |yaw| per frame of the orbit round the subject's left side, 1: round the right side}
    -> [(side, frame, target)] ordered by the frames' signed yaw. Each target takes the frame nearest it, and a frame
    is used once."""
    chosen = []
    for target in sorted(targets):
        side = 1 if target > 0 else -1
        frame = int(np.argmin(np.abs(angles[side] - abs(target))))
        if (side, frame) not in [c[:2] for c in chosen]:
            chosen.append((side, frame, target))
    return sorted(chosen, key=lambda c: c[0] * angles[c[0]][c[1]])
