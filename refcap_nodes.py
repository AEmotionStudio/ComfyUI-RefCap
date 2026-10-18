"""RefCap (reference capture) for MiniMax H3: reference photos captured once, replayed as native reference conditioning.

A reference image reaches the H3 DiT through two channels (comfy_extras/nodes_minimax_h3.py):
- the text encoder presents "<Picture i>: " plus the image's vision block ahead of the prompt. Those rows are
  5120-wide, and minimax_token_tags marks the vision block 0 (video modality);
- the VAE latent rides in minimax_refs as an image reference block the DiT attends to on every step.
Community RefMod files keep only the second channel, stacked or compressed. A RefCap records what
MiniMaxH3ReferenceToVideo produced, unmodified: the rows and tags of an empty-prompt encode, the latents, and the
pixels the text encoder saw when the images are connected at capture time.

Two ways to use a RefCap:
- H3RefCapReplay runs the text encoder over the stored pixels and the prompt, as the reference node does with the
  original images, so the prompt attends to the images and the conditioning is bit-identical to native. Only the
  image loading, resize and VAE encode are skipped.
- H3RefCapApply prepends the cached rows to a prompt encoded without references, so no image goes through the 32B
  encoder. The prompt rows are then not native: Qwen is causal, so natively they attend to the images, and in front of
  cached rows they were encoded without them (cosine median 0.97, min 0.31 against native). On the London Set faces
  this scored slightly lower ArcFace similarity than native (0.652 against 0.660).
Reference rows are not bit-identical between encodes either. With the nvfp4 encoder every row moved by about 0.03%
(median; 0.9% max) when the prompt changed, including row 0, which a causal model can't change by appending text.
So that shift is numeric, from sequence length.

Version-2 files are view sets (H3ViewSetBuild): many views of one subject, each with a yaw, and optionally an orbit video,
with no cached rows. H3 can't pick a reference by viewing angle (a reference block's RoPE is time/h/w only), so
H3ViewSetReplay picks views for the shot before encoding, the way a TURNtable billboard picks a frame by camera angle,
then replays them exactly as H3RefCapReplay does.

Version 3 extends a view set to what the native node takes: up to 3 clips, each with an optional soundtrack, and up to
3 standalone audio references. A clip stores the 2 fps frames the text encoder sees and its video latent; audio is
stored as the audio VAE's latent only, because the text encoder never sees audio (it gets a bare "<Audio j>: " label),
so the stored latent is exactly what the native node would hand the DiT. Version 2's orbit reads as clip 1.

File size: rows and pixels dominate, latents are 1-6%. Two switches cut that without touching what the model sees:
- `store_rows` (capture): rows and tags exist only for H3RefCapApply. Replay rebuilds them from the pixels, so a
  Replay-only RefCap can drop them. That is 101 MB of a 163 MB five-photo 1024 px capture. View sets never store rows.
- `compact` (capture and build): store each source image as a PNG instead of the resized float pixels, and redo the
  reference node's resize at replay. A PNG decodes to exactly the 8-bit source the resize saw, so the replayed pixels
  are bit-identical; a 1024 px photo costs about 0.7 MB instead of 12 MB. Sources that are not exactly 8-bit (a
  VAE-decoded batch, say) can't round-trip through a PNG, so those fall back to floats automatically.
"""

import io
import json
import math
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import comfy.utils
import folder_paths
import node_helpers
import nodes

from .refcap_views import (IMAGE_EXTENSIONS, natural_key, orbit_angles, orbit_targets, pan_angles, parse_list,
                           pick_orbit_frames, select_views, turntable_yaws, wrap)

REFCAP_FOLDER = "refcaps"
if REFCAP_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[REFCAP_FOLDER] = (
        [os.path.join(folder_paths.models_dir, REFCAP_FOLDER)], {".safetensors"})
REFCAP_FORMAT = "h3_refcap"
MAX_IMAGES = 9  # MiniMaxH3ReferenceToVideo's limits: ref_images, ref_videos (+ their soundtracks), ref_audios
MAX_CLIPS = 3
MAX_AUDIOS = 3
AUDIO_LATENT_FPS = 40  # comfy_extras.nodes_minimax_h3.AUDIO_LATENT_FPS
CATEGORY = "advanced/refcap"


def _safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
        raise ValueError(f"'{name}' must be a plain file name (letters, digits, _ . -)")
    return name


def _ref_tokens(block):
    # the DiT patches latents 2x2, so an image block costs (h/2)*(w/2) tokens, a video that per latent frame, and
    # audio 2 tokens per latent frame (comfy/ldm/minimax/model.py, PackedLayout)
    tokens = 2 * int(block.get("ref_audio_t") or block.get("audio_t") or 0)
    if "latent_h" in block:
        tokens += block.get("latent_t", 1) * (block["latent_h"] // 2) * (block["latent_w"] // 2)
    return tokens


def _read_refcap(refcap):
    path = folder_paths.get_full_path_or_raise(REFCAP_FOLDER, refcap)
    with safe_open(path, "pt") as f:
        meta = json.loads((f.metadata() or {}).get(REFCAP_FORMAT, "null"))
    if not meta or meta.get("version") not in (1, 2, 3):
        raise ValueError(f"{refcap} is not a {REFCAP_FORMAT} file (version 1, 2 or 3)")
    return meta, load_file(path)


def _clips(meta):
    """The clips of a view set, numbered from 1, with the tensor keys they are stored under. A version-2 orbit is
    clip 1; version 3 stores clip{k}_frames, clip{k}_latent and, with a soundtrack, clip{k}_audio_latent."""
    if meta.get("version", 1) < 3:
        o = meta.get("orbit")
        return [{**o, "audio_t": 0, "label": "orbit", "frames_key": "orbit_frames", "latent_key": "orbit_latent",
                 "audio_key": None}] if o else []
    return [{**c, "frames_key": f"clip{k}_frames", "latent_key": f"clip{k}_latent",
             "audio_key": f"clip{k}_audio_latent" if c.get("audio_t") else None}
            for k, c in enumerate(meta.get("clips") or [], 1)]


def _audios(meta):
    """The standalone audio references, numbered from 1, stored as audio{j}_latent."""
    return [{**a, "key": f"audio{j}_latent"} for j, a in enumerate(meta.get("audios") or [], 1)]


def _seconds(audio_t):
    return f"{audio_t / AUDIO_LATENT_FPS:.1f} s"


def _ref_blocks(meta, tensors):
    return [{"kind": "image", "latent_h": r["latent_h"], "latent_w": r["latent_w"], "latent": tensors[f"ref_{i}"]}
            for i, r in enumerate(meta["refs"])]


# --------------------------------------------------------------------------------------------------------------
# Pixel storage: the resized floats the text encoder saw, or the source PNG plus the reference node's own resize

def _resize(frames, latent_w, latent_h):
    """_resize in comfy_extras/nodes_minimax_h3.py, to the size a block's latent records (16 px per latent cell)."""
    return comfy.utils.common_upscale(frames[..., :3].movedim(-1, 1), latent_w * 16, latent_h * 16,
                                      "lanczos", "disabled").movedim(1, -1)


def _png_encode(frames):
    """Frames [F, H, W, 3] in 0..1 -> one PNG (frames stacked down the page) as a uint8 tensor, or None when the values
    are not exactly 8-bit and so cannot round-trip through it."""
    x = frames[..., :3]
    if float(x.min()) < 0.0 or float(x.max()) > 1.0:
        return None
    q = (x * 255.0).round()
    if float((q / 255.0 - x).abs().max()) > 1e-6:
        return None
    arr = q.to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr.reshape(-1, arr.shape[-2], 3)).save(buf, format="PNG", optimize=True)
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8).clone()


def _png_decode(data, count=1):
    """-> [count, H, W, 3] float32, decoded the way LoadImage decodes a file."""
    img = node_helpers.pillow(Image.open, io.BytesIO(bytes(data.cpu().numpy())))
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).reshape(count, arr.shape[0] // count, arr.shape[1], 3)


def _store_pixels(tensors, key, source, resized, compact):
    """Store `key` as the source PNG (`key`_src) or as the resized floats, whichever is smaller. Both replay to the
    same pixels; a PNG only wins when the source isn't much larger than the size it is encoded at, so a small
    reference cut from a huge photo keeps the floats. -> True if the PNG was kept."""
    png = _png_encode(source) if compact else None
    if png is None or png.numel() >= resized.numel() * resized.element_size():
        tensors[key] = resized.float().cpu().contiguous()
        return False
    tensors[f"{key}_src"] = png
    return True


def _read_pixels(tensors, key, latent_w, latent_h, count=1):
    """The pixels the text encoder saw: the stored floats, or the source PNG resized as the reference node resizes."""
    if key in tensors:
        return tensors[key]
    if f"{key}_src" not in tensors:
        raise ValueError(f"no pixels stored for {key}")
    return _resize(_png_decode(tensors[f"{key}_src"], count), latent_w, latent_h)


class H3RefCapCapture:
    @classmethod
    def INPUT_TYPES(cls):
        images = {f"image_{i}": ("IMAGE",) for i in range(1, MAX_IMAGES + 1)}
        images["image_1"] = ("IMAGE", {"tooltip": "the images connected to the reference node, in the same order. "
                                                  "Stores the pixels its text encoder saw, for H3 RefCap Replay"})
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "MiniMaxH3ReferenceToVideo output, encoded with an "
                                                             "empty prompt and image references only"}),
                "name": ("STRING", {"default": "character"}),
            },
            # added after the images so saved workflows keep their widget order
            "optional": {"description": ("STRING", {"default": "", "multiline": True}), **images,
                         "store_rows": ("BOOLEAN", {"default": False,
                                                    "tooltip": "keep the encoder's rows for H3 RefCap Apply (cached). "
                                                               "Replay rebuilds them from the pixels, so a Replay-only "
                                                               "RefCap doesn't need them: they are about 20 MB per "
                                                               "1,000 rows"}),
                         "compact": ("BOOLEAN", {"default": True,
                                                 "tooltip": "store each source image as a PNG and redo the resize at "
                                                            "replay (bit-identical, ~17x smaller) instead of keeping "
                                                            "the resized floats"})},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "capture"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Captures MiniMax H3 image references into one RefCap file in models/refcaps: each image's reference latent, "
        "the pixels its text encoder saw (as a PNG with `compact`), and optionally the encoder's rows for H3 RefCap "
        "Apply. Use it with H3 RefCap Replay."
    )

    def capture(self, conditioning, name, description="", store_rows=False, compact=True, **images):
        cond, extra = conditioning[0]
        refs = extra.get("minimax_refs") or []
        tags = extra.get("minimax_token_tags")
        if tags is None:
            raise ValueError("conditioning has no minimax_token_tags; encode it with the MiniMax H3 CLIP")
        if not refs or any(r.get("kind") != "image" for r in refs):
            raise ValueError("a RefCap needs image references only (videos and audio are not supported)")
        tags = tags.reshape(-1)
        if tags.shape[0] != cond.shape[1]:
            raise ValueError(f"{tags.shape[0]} token tags for {cond.shape[1]} conditioning rows")
        # The presentation ends on the last image's vision_end token (tag 0); any text after it is a prompt.
        if int(tags[-1]) != 0:
            raise ValueError("encode the references with an empty prompt: the conditioning ends in text rows")
        given = [images[f"image_{i}"] for i in range(1, MAX_IMAGES + 1) if images.get(f"image_{i}") is not None]
        if given and len(given) != len(refs):
            raise ValueError(f"{len(given)} images connected for {len(refs)} references: connect all of them, "
                             "in reference order, or none")
        if not given and not store_rows:
            raise ValueError("connect the images, or turn store_rows on: a RefCap with neither has nothing to replay")

        tensors = {}
        if store_rows:
            tensors["rows"] = cond[0].detach().cpu().contiguous()
            tensors["tags"] = tags.to(torch.int8).cpu().contiguous()
        for i, r in enumerate(refs):
            tensors[f"ref_{i}"] = r["latent"].detach().cpu().contiguous()
        compacted = 0
        for i, (r, image) in enumerate(zip(refs, given)):
            resized = _resize(image[:1], r["latent_w"], r["latent_h"])
            compacted += _store_pixels(tensors, f"pixels_{i}", image[:1], resized, compact)
        meta = {"format": REFCAP_FORMAT, "version": 1, "description": description, "pixels": bool(given),
                "rows": bool(store_rows),
                "refs": [{"kind": "image", "latent_h": r["latent_h"], "latent_w": r["latent_w"]} for r in refs]}

        folder = folder_paths.get_folder_paths(REFCAP_FOLDER)[0]
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{_safe_name(name)}.safetensors")
        save_file(tensors, path, metadata={REFCAP_FORMAT: json.dumps(meta)})

        vision = int((tags == 0).sum())
        summary = (f"{os.path.basename(path)}: {len(refs)} images"
                   + (f" ({compacted} as PNG)" if given else " (no pixels)")
                   + (f", {cond.shape[1]} rows ({vision} vision)" if store_rows else ", no cached rows (Replay only)")
                   + f", {sum(_ref_tokens(r) for r in refs)} reference tokens, {os.path.getsize(path) / 2**20:.1f} MB")
        return {"ui": {"text": [summary]}, "result": (summary,)}


class H3RefCapReplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "refcap": (folder_paths.get_filename_list(REFCAP_FOLDER),),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "replay"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Replays a RefCap exactly: encodes a MiniMax H3 prompt with the captured images, as MiniMaxH3ReferenceToVideo "
        "does with the originals, and attaches the captured reference latents. Needs a RefCap captured with its images "
        "connected. Refer to the images as <Picture 1>, <Picture 2>, ... Take the latent from EmptyMiniMaxH3LatentAV."
    )

    def replay(self, clip, prompt, refcap):
        meta, tensors = _read_refcap(refcap)
        if not meta.get("pixels"):
            raise ValueError(f"{refcap} has no stored pixels: capture it again with the images connected to "
                             "H3 RefCap Capture, or use H3 RefCap Apply (cached)")
        items = [{"type": "image",
                  "data": _read_pixels(tensors, f"pixels_{i}", r["latent_w"], r["latent_h"])}
                 for i, r in enumerate(meta["refs"])]
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
        return (node_helpers.conditioning_set_values(cond, {"minimax_refs": _ref_blocks(meta, tensors)}),)


class H3RefCapApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "prompt encoded without references, e.g. "
                                                             "MiniMaxH3ReferenceToVideo with no ref inputs"}),
                "refcap": (folder_paths.get_filename_list(REFCAP_FOLDER),),
                "rows": ("BOOLEAN", {"default": True, "tooltip": "prepend the text encoder's <Picture i> rows"}),
                "latents": ("BOOLEAN", {"default": True, "tooltip": "attach the reference latents to the DiT"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Applies a RefCap's cached references to an already encoded MiniMax H3 prompt: its <Picture i> rows go in "
        "front of the prompt and its latents become reference blocks. No image passes through the text encoder, but "
        "the prompt was encoded without seeing them; H3 RefCap Replay matches native references exactly."
    )

    def apply(self, conditioning, refcap, rows=True, latents=True):
        meta, tensors = _read_refcap(refcap)
        if rows and "rows" not in tensors:
            raise ValueError(f"{refcap} has no cached rows: capture it again with store_rows on, or use "
                             "H3 RefCap Replay (view sets never store rows)")
        blocks = _ref_blocks(meta, tensors)

        out = []
        for cond, extra in conditioning:
            tags = extra.get("minimax_token_tags")
            if tags is None:
                raise ValueError("conditioning has no minimax_token_tags; encode it with the MiniMax H3 CLIP")
            # RefCap rows are numbered <Picture 1..n> and its blocks are ordered to match, so references already in
            # the conditioning would collide with the labels and shift the blocks.
            if extra.get("minimax_refs") or bool((tags == 0).any()):
                raise ValueError("connect a prompt encoded without references; this one already has image rows or refs")
            extra = extra.copy()
            if rows:
                cached = tensors["rows"].to(device=cond.device, dtype=cond.dtype)
                cond = torch.cat([cached.unsqueeze(0).expand(cond.shape[0], -1, -1), cond], dim=1)
                extra["minimax_token_tags"] = torch.cat(
                    [tensors["tags"].to(device=tags.device, dtype=tags.dtype), tags.reshape(-1)])
            if latents:
                extra["minimax_refs"] = blocks
            out.append([cond, extra])
        return (out,)


class H3SaveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "filename": ("STRING", {"default": "refcap_debug/conditioning"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Debug: saves H3 conditioning rows, token tags and reference latents (ref_i, and ref_i_audio for "
                   "soundtracks and audio references) under output/ for comparison.")

    def save(self, conditioning, filename):
        cond, extra = conditioning[0]
        tensors = {"rows": cond[0].detach().cpu().contiguous()}
        if extra.get("minimax_token_tags") is not None:
            tensors["tags"] = extra["minimax_token_tags"].reshape(-1).cpu().contiguous()
        for i, r in enumerate(extra.get("minimax_refs") or []):
            if r.get("latent") is not None:
                tensors[f"ref_{i}"] = r["latent"].detach().cpu().contiguous()
            if r.get("audio_latent") is not None:
                tensors[f"ref_{i}_audio"] = r["audio_latent"].detach().cpu().contiguous()
        output = os.path.abspath(folder_paths.get_output_directory())
        path = os.path.abspath(os.path.join(output, f"{filename}.safetensors"))
        if os.path.commonpath((output, path)) != output:
            raise ValueError(f"'{filename}' is outside the output folder")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(tensors, path)
        return {}


def _load_folder(folder):
    """Every image in a folder, natural-sorted and decoded as LoadImage does -> ([IMAGE [1, H, W, 3]], names)."""
    if not os.path.isabs(folder):
        folder = os.path.join(folder_paths.get_input_directory(), folder)
    names = sorted((n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS)), key=natural_key)
    if not names:
        raise ValueError(f"no images in {folder}")
    images = []
    for name in names:
        img = node_helpers.pillow(ImageOps.exif_transpose, node_helpers.pillow(Image.open, os.path.join(folder, name)))
        images.append(torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0)[None])
    return images, names


def _encode_view(h3, vae, img, ref_image_size, width, height):
    """MiniMaxH3ReferenceToVideo's image path -> (source, resized pixels, latent, block metadata)."""
    h, w = img.shape[1], img.shape[2]
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, h3.REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(h3.CANVAS_MULTIPLE, round(w * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
    th = max(h3.CANVAS_MULTIPLE, round(h * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
    resized = h3._resize(img[:1], tw, th, "disabled")
    return (img[:1], resized, vae.encode(resized).cpu().contiguous(),
            {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16})


def _encode_clip(h3, vae, frames, length):
    """MiniMaxH3ReferenceToVideo's video path, cut to the frame count of a render at `length`
    -> (the source frames the text encoder's 2 fps sampling lands on, those frames resized, video latent, metadata)."""
    vh, vw = frames.shape[1], frames.shape[2]
    cw, ch = h3.adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw = max(h3.CANVAS_MULTIPLE, round(vw / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        ch = max(h3.CANVAS_MULTIPLE, round(vh / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
    source = frames[:h3.align_frame_count(max(5, length))]
    resized = h3._resize(frames, cw, ch, "disabled")[:h3.align_frame_count(max(5, length))]
    k = resized.shape[0]
    if k < 5:
        raise ValueError("a clip needs at least 5 frames")
    while k % 17 != 5:
        k -= 1
    resized, source = resized[:k], source[:k]
    z = vae.encode(resized)
    qwen = list(range(0, k, h3.FPS // 2))  # the text encoder sees it at 2 fps
    meta = {"kind": "video", "frames": k, "latent_t": int(z.shape[2]), "latent_h": ch // 16, "latent_w": cw // 16,
            "timestamps": [j / 2.0 for j in range(len(qwen))]}
    return source[qwen], resized[qwen].float().cpu().contiguous(), z.cpu().contiguous(), meta


def _encode_audio(h3, audio_vae, audio):
    """MiniMaxH3ReferenceToVideo's audio path -> (audio latent [1, 32, 2, T], T). The native node doesn't cut a
    soundtrack to the render's length, so neither does this; trim the AUDIO first if that matters."""
    z, t = h3._encode_ref_audio(audio_vae, audio)
    return z.cpu().contiguous(), int(t)


class H3ViewSetBuild:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "name": ("STRING", {"default": "character_views"}),
                "ref_image_size": (["max", "match"], {"tooltip": "as MiniMaxH3ReferenceToVideo: 'match' scales views down "
                                                                 "to the width x height area, 'max' to a 2048 px short edge"}),
                "width": ("INT", {"default": 608, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 352, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
            },
            "optional": {
                "views": ("IMAGE", {"tooltip": "a batch of views of one subject"}),
                "view_folder": ("STRING", {"default": "", "tooltip": "a TURNtable image sequence folder (absolute or under "
                                                                     "ComfyUI/input), natural-sorted as the Blender add-on "
                                                                     "sorts it; appended after `views`"}),
                "yaws": ("STRING", {"default": "", "tooltip": "one yaw in degrees per view, comma-separated. Empty: "
                                                              "TURNtable spacing, view i at i*360/N + frame_offset_degrees"}),
                "frame_offset_degrees": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5}),
                "reverse_direction": ("BOOLEAN", {"default": False}),
                "orbit": ("IMAGE", {"tooltip": "clip 1: orbit (or any) video frames at 24 fps, stored as a video reference"}),
                "orbit_folder": ("STRING", {"default": "", "tooltip": "or a folder of clip 1's frames"}),
                "orbit_length": ("INT", {"default": 56, "min": 5, "max": 3600, "step": 17,
                                         "tooltip": "the length you will render at: H3 cuts a reference video to the "
                                                    "generation's frame count (applies to every clip)"}),
                "description": ("STRING", {"default": "", "multiline": True}),
                # version 3: added after `description` so saved workflows keep their widget order
                "audio_vae": ("VAE", {"tooltip": "the MiniMax H3 audio VAE, needed when any audio is connected"}),
                "orbit_audio": ("AUDIO", {"tooltip": "clip 1's soundtrack, stored as its audio latent"}),
                "clip_2": ("IMAGE", {"tooltip": "clip 2 frames at 24 fps"}),
                "clip_2_folder": ("STRING", {"default": ""}),
                "clip_2_audio": ("AUDIO",),
                "clip_3": ("IMAGE", {"tooltip": "clip 3 frames at 24 fps"}),
                "clip_3_folder": ("STRING", {"default": ""}),
                "clip_3_audio": ("AUDIO",),
                "audio_1": ("AUDIO", {"tooltip": "standalone audio reference (a voice, music, ambience)"}),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "compact": ("BOOLEAN", {"default": True,
                                        "tooltip": "store sources as PNGs and redo the resize at replay "
                                                   "(bit-identical, ~17x smaller than the resized floats)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Builds a RefCap view set in models/refcaps: many views of one subject with a yaw each, up to 3 clips with "
        "optional soundtracks, and up to 3 standalone audio references, encoded exactly as MiniMaxH3ReferenceToVideo "
        "encodes references. Takes a TURNtable Viewer image sequence folder directly. Use it with H3 RefCap Replay View Set."
    )

    def build(self, vae, name, ref_image_size, width, height, views=None, view_folder="", yaws="",
              frame_offset_degrees=0.0, reverse_direction=False, orbit=None, orbit_folder="", orbit_length=56,
              description="", audio_vae=None, orbit_audio=None, clip_2=None, clip_2_folder="", clip_2_audio=None,
              clip_3=None, clip_3_folder="", clip_3_audio=None, audio_1=None, audio_2=None, audio_3=None,
              compact=True):
        from comfy_extras import nodes_minimax_h3 as h3  # imported here so a core rename breaks only these nodes

        images, labels = [], []
        if views is not None:
            images = [views[i:i + 1] for i in range(views.shape[0])]
            labels = [f"batch {i}" for i in range(len(images))]
        if view_folder.strip():
            more, names = _load_folder(view_folder.strip())
            images, labels = images + more, labels + names

        # clips in slot order; a clip's number is its position among the connected ones
        clips = []
        for k, (frames, folder, audio) in enumerate(((orbit, orbit_folder, orbit_audio), (clip_2, clip_2_folder, clip_2_audio),
                                                     (clip_3, clip_3_folder, clip_3_audio)), 1):
            label = "orbit" if k == 1 else f"clip {k}"
            if folder.strip():
                frames = torch.cat(_load_folder(folder.strip())[0])
                label = os.path.basename(os.path.normpath(folder.strip()))
            elif frames is not None and k > 1:
                label = f"clip {k} batch"
            if frames is None:
                if audio is not None:
                    raise ValueError(f"clip {k} has a soundtrack but no frames")
                continue
            clips.append((frames, audio, label))
        audios = [a for a in (audio_1, audio_2, audio_3) if a is not None]
        if not images and not clips and not audios:
            raise ValueError("connect views, a view_folder, a clip or an audio")
        if audio_vae is None and (audios or any(a is not None for _, a, _ in clips)):
            raise ValueError("connect the audio_vae to store soundtracks or audio references")

        n = len(images)
        yaw_list = parse_list(yaws, float)
        if yaw_list and len(yaw_list) != n:
            raise ValueError(f"{len(yaw_list)} yaws for {n} views")
        if not yaw_list:
            yaw_list = turntable_yaws(n, frame_offset_degrees, reverse_direction)

        # encode in the native node's order: images, then each video and its soundtrack, then standalone audio
        tensors, refs, tokens, compacted = {}, [], 0, 0
        for i, (img, label, yaw) in enumerate(zip(images, labels, yaw_list)):
            source, resized, tensors[f"ref_{i}"], block = _encode_view(h3, vae, img, ref_image_size, width, height)
            compacted += _store_pixels(tensors, f"pixels_{i}", source, resized, compact)
            refs.append({**block, "yaw": wrap(yaw), "label": label})
            tokens += _ref_tokens(block)

        clip_meta = []
        for k, (frames, audio, label) in enumerate(clips, 1):
            source, resized, tensors[f"clip{k}_latent"], block = _encode_clip(h3, vae, frames, orbit_length)
            compacted += _store_pixels(tensors, f"clip{k}_frames", source, resized, compact)
            block.update(audio_t=0, label=label)
            if audio is not None:
                tensors[f"clip{k}_audio_latent"], block["audio_t"] = _encode_audio(h3, audio_vae, audio)
            clip_meta.append(block)
            tokens += _ref_tokens(block)

        audio_meta = []
        for j, audio in enumerate(audios, 1):
            tensors[f"audio{j}_latent"], t = _encode_audio(h3, audio_vae, audio)
            audio_meta.append({"kind": "audio", "ref_audio_t": t, "label": f"audio {j}"})
            tokens += _ref_tokens(audio_meta[-1])

        meta = {"format": REFCAP_FORMAT, "version": 3, "description": description, "pixels": True, "refs": refs,
                "clips": clip_meta, "audios": audio_meta}
        folder = folder_paths.get_folder_paths(REFCAP_FOLDER)[0]
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{_safe_name(name)}.safetensors")
        save_file(tensors, path, metadata={REFCAP_FORMAT: json.dumps(meta)})

        parts = [f"{n} views"]
        parts += [f"clip {k}: {c['frames']} frames" + (f" + {_seconds(c['audio_t'])} soundtrack" if c["audio_t"] else "")
                  for k, c in enumerate(clip_meta, 1)]
        parts += [f"audio {j}: {_seconds(a['ref_audio_t'])}" for j, a in enumerate(audio_meta, 1)]
        summary = (f"{os.path.basename(path)}: " + ", ".join(parts)
                   + f"; {compacted} of {n + len(clip_meta)} stored as PNG"
                   + f"; {tokens} reference tokens if all are used, {os.path.getsize(path) / 2**20:.1f} MB")
        return {"ui": {"text": [summary]}, "result": (summary,)}


class H3ViewSetReplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "viewset": (folder_paths.get_filename_list(REFCAP_FOLDER),),
                "select": (["nearest", "all", "indices"], {"tooltip": "nearest: the views closest to the shot's angles"}),
                "target_yaws": ("STRING", {"default": "0", "tooltip": "select=nearest: the angles the shot shows, in "
                                                                      "degrees, comma-separated"}),
                "count": ("INT", {"default": 0, "min": 0, "max": MAX_IMAGES,
                                  "tooltip": "select=nearest: how many views; 0 = one per target yaw"}),
                "indices": ("STRING", {"default": "", "tooltip": "select=indices: 0-based view numbers, comma-separated"}),
                "orbit": ("BOOLEAN", {"default": False, "tooltip": "add clip 1 (the orbit) as a video reference"}),
            },
            "optional": {
                "more_clips": ("STRING", {"default": "", "tooltip": "other clip numbers to add, comma-separated (2,3)"}),
                "audios": ("STRING", {"default": "", "tooltip": "standalone audio numbers to add (1,2,3), or 'all'"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "label_guide")
    FUNCTION = "replay"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Replays chosen views, clips and audio of a RefCap view set as native MiniMax H3 references: the prompt is "
        "encoded with the stored pixels and clip frames, as MiniMaxH3ReferenceToVideo does with the originals, and the "
        "stored latents are attached. The label guide maps <Picture i>, <Video k> and <Audio j> to views, clips and "
        "audio for the prompt."
    )

    def replay(self, clip, prompt, viewset, select, target_yaws, count, indices, orbit, more_clips="", audios=""):
        meta, tensors = _read_refcap(viewset)
        if not meta.get("pixels"):
            raise ValueError(f"{viewset} has no stored pixels")
        views = meta["refs"]
        yaws = [r.get("yaw") for r in views]
        if select == "nearest" and None in yaws:
            raise ValueError(f"{viewset} has no yaws (a version-1 RefCap): use select=all or indices")
        chosen = select_views(yaws, select, target_yaws, count, indices) if views else []
        if len(chosen) > MAX_IMAGES:
            raise ValueError(f"{len(chosen)} views selected; MiniMaxH3ReferenceToVideo takes at most {MAX_IMAGES}")

        clips, standalone = _clips(meta), _audios(meta)
        clip_numbers = sorted(set(([1] if orbit else []) + parse_list(more_clips, int)))
        bad = [k for k in clip_numbers if not 1 <= k <= len(clips)]
        if bad:
            raise ValueError(f"{viewset} has {len(clips)} clips; no clip {bad}")
        if len(clip_numbers) > MAX_CLIPS:
            raise ValueError(f"{len(clip_numbers)} clips selected; MiniMaxH3ReferenceToVideo takes at most {MAX_CLIPS}")
        audio_numbers = (list(range(1, len(standalone) + 1)) if audios.strip().lower() == "all"
                         else sorted(set(parse_list(audios, int))))
        bad = [j for j in audio_numbers if not 1 <= j <= len(standalone)]
        if bad:
            raise ValueError(f"{viewset} has {len(standalone)} audio references; no audio {bad}")
        if len(audio_numbers) > MAX_AUDIOS:
            raise ValueError(f"{len(audio_numbers)} audios selected; MiniMaxH3ReferenceToVideo takes at most {MAX_AUDIOS}")

        # Native order: images, then each video (its soundtrack's <Audio j> right before its <Video k>), then
        # standalone audio. The tokenizer numbers each kind in emission order, so <Audio j> counts soundtracks too.
        items, blocks, guide, audio_n = [], [], [], 0
        for p, i in enumerate(chosen, 1):
            r = views[i]
            items.append({"type": "image",
                          "data": _read_pixels(tensors, f"pixels_{i}", r["latent_w"], r["latent_h"])})
            blocks.append({"kind": "image", "latent_h": r["latent_h"], "latent_w": r["latent_w"],
                           "latent": tensors[f"ref_{i}"]})
            guide.append(f"<Picture {p}> = view {i}" + (f", yaw {r['yaw']:g}" if r.get("yaw") is not None else "")
                         + (f" ({r['label']})" if r.get("label") else ""))
        for p, k in enumerate(clip_numbers, 1):
            c = clips[k - 1]
            if c["audio_key"]:
                audio_n += 1
                items.append({"type": "audio"})
                guide.append(f"<Audio {audio_n}> = clip {k}'s soundtrack, {_seconds(c['audio_t'])}")
            items.append({"type": "video",
                          "data": _read_pixels(tensors, c["frames_key"], c["latent_w"], c["latent_h"],
                                               len(c["timestamps"])),
                          "timestamps": c["timestamps"]})
            blocks.append({"kind": "video_audio" if c["audio_key"] else "video", "latent_t": c["latent_t"],
                           "latent_h": c["latent_h"], "latent_w": c["latent_w"], "ref_audio_t": c["audio_t"],
                           "latent": tensors[c["latent_key"]],
                           "audio_latent": tensors[c["audio_key"]] if c["audio_key"] else None})
            guide.append(f"<Video {p}> = clip {k} ({c.get('label', '')}), {c['frames']} frames (render at least that long)")
        for j in audio_numbers:
            a = standalone[j - 1]
            audio_n += 1
            items.append({"type": "audio"})
            blocks.append({"kind": "audio", "ref_audio_t": a["ref_audio_t"], "audio_latent": tensors[a["key"]]})
            guide.append(f"<Audio {audio_n}> = audio {j}, {_seconds(a['ref_audio_t'])}")
        if not blocks:
            raise ValueError("no views selected and no clip or audio")

        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
        guide.append(f"{sum(_ref_tokens(b) for b in blocks)} reference tokens")
        text = "\n".join(guide)
        return {"ui": {"text": [text]},
                "result": (node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks}), text)}


class H3OrbitViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("IMAGE", {"tooltip": "frames of an orbit that turns from the front round the subject's left "
                                              "side to directly behind"}),
                "right": ("IMAGE", {"tooltip": "the same round the subject's right side"}),
                "yaws": ("STRING", {"default": "", "tooltip": "angles to take, comma-separated degrees (negative: the "
                                                              "left orbit). Empty: every 15 degrees round the head"}),
                "measure": (["faces", "motion", "steady"],
                            {"tooltip": "how each frame's angle is found (H3's orbits and pans ease in, so frame "
                                        "position is a poor guide). faces: insightface's measured head yaw, for orbits "
                                        "round a person. motion: the image's sideways motion, for camera pans round a "
                                        "place. steady: assume a steady turn from 0 to 180"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("views", "yaws")
    FUNCTION = "pick"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Picks a turnaround from two H3 orbit renders, one round each side of the subject, or from two camera pans "
        "round a place: the frame nearest each angle, with the angle it was seen from. Feed views and yaws to H3 "
        "RefCap Build View Set."
    )

    def pick(self, left, right, yaws, measure):
        clips = {-1: (left * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
                 1: (right * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()}
        if measure == "faces":
            from insightface.app import FaceAnalysis  # optional: only this node needs it
            app = FaceAnalysis(name="buffalo_l", root=os.path.join(folder_paths.models_dir, "insightface"),
                               providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_3d_68"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            angles = {}
            for side, frames in clips.items():
                measured = []
                for rgb in frames:
                    faces = app.get(np.ascontiguousarray(rgb[:, :, ::-1]))  # insightface wants BGR
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)
                    measured.append(None if face is None or face.pose is None else float(face.pose[1]))
                angles[side] = orbit_angles(measured)
        elif measure == "motion":
            angles = {side: pan_angles(frames) for side, frames in clips.items()}
        else:
            angles = {side: np.linspace(0.0, 180.0, len(frames)) for side, frames in clips.items()}

        chosen = pick_orbit_frames(angles, orbit_targets(yaws))
        # 8-bit, as a saved frame would be, so H3 RefCap Build View Set can store them compact
        views = torch.from_numpy(np.stack([clips[side][frame] for side, frame, _ in chosen])).float() / 255.0
        out = ",".join(f"{side * float(angles[side][frame]) + 0.0:.1f}" for side, frame, _ in chosen)  # + 0.0: no "-0.0"
        return {"ui": {"text": [f"{len(chosen)} views at {out}"]}, "result": (views, out)}


NODE_CLASS_MAPPINGS = {
    "H3RefCapCapture": H3RefCapCapture,
    "H3RefCapReplay": H3RefCapReplay,
    "H3RefCapApply": H3RefCapApply,
    "H3SaveConditioning": H3SaveConditioning,
    "H3ViewSetBuild": H3ViewSetBuild,
    "H3ViewSetReplay": H3ViewSetReplay,
    "H3OrbitViews": H3OrbitViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefCapCapture": "H3 RefCap Capture",
    "H3RefCapReplay": "H3 RefCap Replay",
    "H3RefCapApply": "H3 RefCap Apply (cached)",
    "H3SaveConditioning": "H3 Save Conditioning (debug)",
    "H3ViewSetBuild": "H3 RefCap Build View Set",
    "H3ViewSetReplay": "H3 RefCap Replay View Set",
    "H3OrbitViews": "H3 RefCap Orbit Views",
}
