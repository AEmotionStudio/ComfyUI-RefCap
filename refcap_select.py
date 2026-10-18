"""Pick views from several RefCaps and replay them together: two characters and a location in one shot.

A RefCap holds one subject, and MiniMaxH3ReferenceToVideo numbers its references <Picture 1..n> in the order they are
attached. H3RefCapSelect chains one choice per file into a pack, and H3RefCapPackReplay renumbers the whole pack and
encodes the prompt with it, exactly as H3ViewSetReplay does for a single file. H3RefCapBake writes the same pack out as
one view set when a cast is worth keeping.

The image cap is the reference node's: 9 images, 3 clips and 3 audio references per render, whatever they come from.
"""

import io
import json
import os

import numpy as np
from aiohttp import web
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import server

import folder_paths
import node_helpers

from .refcap_nodes import (
    AUDIO_LATENT_FPS, MAX_AUDIOS, MAX_CLIPS, MAX_IMAGES, REFCAP_FOLDER, REFCAP_FORMAT, _audios, _clips, _read_pixels,
    _read_refcap, _ref_tokens, _safe_name, _seconds,
)

CATEGORY = "advanced/refcap"
THUMB_EDGE = 192


def _parse_views(text, count):
    """"all", or 0-based numbers and ranges: "0,2,5-7" -> [0, 2, 5, 6, 7]."""
    text = text.strip()
    if not text or text.lower() == "none":
        return []
    if text.lower() == "all":
        return list(range(count))
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    for i in out:
        if not 0 <= i < count:
            raise ValueError(f"no view {i}: this RefCap has {count}")
    return out


def _parse_numbers(text, count, what):
    """1-based clip/audio numbers, "all", or empty."""
    text = text.strip()
    if not text or text.lower() == "none":
        return []
    numbers = list(range(1, count + 1)) if text.lower() == "all" else [int(p) for p in text.split(",") if p.strip()]
    for k in numbers:
        if not 1 <= k <= count:
            raise ValueError(f"no {what} {k}: this RefCap has {count}")
    return numbers


def _entry_meta(refcap):
    """Metadata only, without reading a single tensor."""
    path = folder_paths.get_full_path_or_raise(REFCAP_FOLDER, refcap)
    with safe_open(path, "pt") as f:
        meta = json.loads((f.metadata() or {}).get(REFCAP_FORMAT, "null"))
    if not meta or meta.get("version") not in (1, 2, 3):
        raise ValueError(f"{refcap} is not a {REFCAP_FORMAT} file (version 1, 2 or 3)")
    return meta


def _totals(pack):
    images = sum(len(e["views"]) for e in pack)
    clips = sum(len(e["clips"]) for e in pack)
    audios = sum(len(e["audios"]) for e in pack)
    return images, clips, audios


def _check_caps(pack):
    images, clips, audios = _totals(pack)
    for count, cap, what in ((images, MAX_IMAGES, "images"), (clips, MAX_CLIPS, "clips"), (audios, MAX_AUDIOS, "audio references")):
        if count > cap:
            raise ValueError(f"{count} {what} selected; MiniMaxH3ReferenceToVideo takes at most {cap}")


class H3RefCapSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refcap": (folder_paths.get_filename_list(REFCAP_FOLDER),),
                "views": ("STRING", {"default": "all", "tooltip": "which views: all, none, or 0-based numbers and "
                                                                  "ranges (0,2,5-7). Use the picker button"}),
            },
            "optional": {
                "pack": ("REFCAP_PACK", {"tooltip": "another H3 RefCap Select, to add this one after it"}),
                "clips": ("STRING", {"default": "", "tooltip": "clip numbers to add (1,2,3) or all"}),
                "audios": ("STRING", {"default": "", "tooltip": "audio reference numbers to add (1,2,3) or all"}),
            },
        }

    RETURN_TYPES = ("REFCAP_PACK",)
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Picks views (and clips or audio) from one RefCap and adds them to a pack, so several RefCaps - one "
                   "per character, one for the location - can reference one shot. Chain these and feed H3 RefCap "
                   "Replay Pack.")

    def select(self, refcap, views, pack=None, clips="", audios=""):
        meta = _entry_meta(refcap)
        entry = {"refcap": refcap,
                 "views": _parse_views(views, len(meta.get("refs") or [])),
                 "clips": _parse_numbers(clips, len(_clips(meta)), "clip"),
                 "audios": _parse_numbers(audios, len(_audios(meta)), "audio reference")}
        if not (entry["views"] or entry["clips"] or entry["audios"]):
            raise ValueError(f"nothing selected from {refcap}")
        return ([*(pack or []), entry],)


def _pack_references(pack):
    """The pack as the tokenizer and the DiT want it, in MiniMaxH3ReferenceToVideo's order: every image, then each clip
    (its soundtrack's <Audio j> right before its <Video k>), then the standalone audio. -> (items, blocks, guide)."""
    _check_caps(pack)
    loaded = [(e, *_read_refcap(e["refcap"])) for e in pack]
    items, blocks, guide, audio_n = [], [], [], 0

    for entry, meta, tensors in loaded:
        if entry["views"] and not meta.get("pixels"):
            raise ValueError(f"{entry['refcap']} has no stored pixels: capture it again with its images connected")
        for i in entry["views"]:
            r = meta["refs"][i]
            items.append({"type": "image",
                          "data": _read_pixels(tensors, f"pixels_{i}", r["latent_w"], r["latent_h"])})
            blocks.append({"kind": "image", "latent_h": r["latent_h"], "latent_w": r["latent_w"],
                           "latent": tensors[f"ref_{i}"]})
            guide.append(f"<Picture {len(blocks)}> = {entry['refcap']} view {i}"
                         + (f", yaw {r['yaw']:g}" if r.get("yaw") is not None else "")
                         + (f" ({r['label']})" if r.get("label") else ""))

    video_n = 0
    for entry, meta, tensors in loaded:
        for k in entry["clips"]:
            c = _clips(meta)[k - 1]
            if c["audio_key"]:
                audio_n += 1
                items.append({"type": "audio"})
                guide.append(f"<Audio {audio_n}> = {entry['refcap']} clip {k}'s soundtrack, {_seconds(c['audio_t'])}")
            video_n += 1
            items.append({"type": "video",
                          "data": _read_pixels(tensors, c["frames_key"], c["latent_w"], c["latent_h"],
                                               len(c["timestamps"])),
                          "timestamps": c["timestamps"]})
            blocks.append({"kind": "video_audio" if c["audio_key"] else "video", "latent_t": c["latent_t"],
                           "latent_h": c["latent_h"], "latent_w": c["latent_w"], "ref_audio_t": c["audio_t"],
                           "latent": tensors[c["latent_key"]],
                           "audio_latent": tensors[c["audio_key"]] if c["audio_key"] else None})
            guide.append(f"<Video {video_n}> = {entry['refcap']} clip {k}, {c['frames']} frames "
                         "(render at least that long)")

    for entry, meta, tensors in loaded:
        for j in entry["audios"]:
            a = _audios(meta)[j - 1]
            audio_n += 1
            items.append({"type": "audio"})
            blocks.append({"kind": "audio", "ref_audio_t": a["ref_audio_t"], "audio_latent": tensors[a["key"]]})
            guide.append(f"<Audio {audio_n}> = {entry['refcap']} audio {j}, {_seconds(a['ref_audio_t'])}")

    if not blocks:
        raise ValueError("the pack is empty")
    guide.append(f"{sum(_ref_tokens(b) for b in blocks)} reference tokens")
    return items, blocks, guide


class H3RefCapPackReplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                             "pack": ("REFCAP_PACK",)}}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "label_guide")
    FUNCTION = "replay"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Replays a pack of RefCaps as native MiniMax H3 references: the prompt is encoded with every "
                   "selected view, as MiniMaxH3ReferenceToVideo does with the original files. The label guide maps "
                   "<Picture i>, <Video k> and <Audio j> to the RefCap they came from. Take the latent from "
                   "EmptyMiniMaxH3LatentAV.")

    def replay(self, clip, prompt, pack):
        items, blocks, guide = _pack_references(pack)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
        text = "\n".join(guide)
        return {"ui": {"text": [text]},
                "result": (node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks}), text)}


class H3RefCapBake:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pack": ("REFCAP_PACK",), "name": ("STRING", {"default": "cast"})},
                "optional": {"description": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "bake"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Writes a pack out as one RefCap view set in models/refcaps, so a cast you keep using becomes a "
                   "single file for H3 RefCap Replay View Set. It copies the stored pixels and latents, so nothing is "
                   "re-encoded; re-bake after re-capturing a source RefCap.")

    def bake(self, pack, name, description=""):
        tensors, refs, clip_meta, audio_meta = {}, [], [], []
        for entry in pack:
            meta, source = _read_refcap(entry["refcap"])
            if entry["views"] and not meta.get("pixels"):
                raise ValueError(f"{entry['refcap']} has no stored pixels: capture it again with its images connected")
            for i in entry["views"]:
                r = meta["refs"][i]
                n = len(refs)
                for suffix in ("", "_src"):
                    if f"pixels_{i}{suffix}" in source:
                        tensors[f"pixels_{n}{suffix}"] = source[f"pixels_{i}{suffix}"]
                tensors[f"ref_{n}"] = source[f"ref_{i}"]
                refs.append({"kind": "image", "latent_h": r["latent_h"], "latent_w": r["latent_w"],
                             "yaw": r.get("yaw"), "label": f"{entry['refcap']} view {i}"})
            for k in entry["clips"]:
                c = _clips(meta)[k - 1]
                n = len(clip_meta) + 1
                for suffix in ("", "_src"):
                    if f"{c['frames_key']}{suffix}" in source:
                        tensors[f"clip{n}_frames{suffix}"] = source[f"{c['frames_key']}{suffix}"]
                tensors[f"clip{n}_latent"] = source[c["latent_key"]]
                if c["audio_key"]:
                    tensors[f"clip{n}_audio_latent"] = source[c["audio_key"]]
                clip_meta.append({"kind": "video", "frames": c["frames"], "latent_t": c["latent_t"],
                                  "latent_h": c["latent_h"], "latent_w": c["latent_w"],
                                  "timestamps": c["timestamps"], "audio_t": c["audio_t"],
                                  "label": f"{entry['refcap']} clip {k}"})
            for j in entry["audios"]:
                a = _audios(meta)[j - 1]
                n = len(audio_meta) + 1
                tensors[f"audio{n}_latent"] = source[a["key"]]
                audio_meta.append({"kind": "audio", "ref_audio_t": a["ref_audio_t"],
                                   "label": f"{entry['refcap']} audio {j}"})

        meta = {"format": REFCAP_FORMAT, "version": 3, "pixels": True,
                "description": description or "; ".join(f"{e['refcap']} views {e['views']}" for e in pack),
                "refs": refs, "clips": clip_meta, "audios": audio_meta}
        folder = folder_paths.get_folder_paths(REFCAP_FOLDER)[0]
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{_safe_name(name)}.safetensors")
        save_file(tensors, path, metadata={REFCAP_FORMAT: json.dumps(meta)})
        tokens = sum(_ref_tokens(b) for b in refs + clip_meta + audio_meta)
        summary = (f"{os.path.basename(path)}: {len(refs)} views, {len(clip_meta)} clips, {len(audio_meta)} audio; "
                   f"{tokens} reference tokens, {os.path.getsize(path) / 2**20:.1f} MB")
        return {"ui": {"text": [summary]}, "result": (summary,)}


# --------------------------------------------------------------------------------------------------------------
# The picker's two routes: a metadata-only library and one thumbnail per view

def _library():
    out = []
    for name in folder_paths.get_filename_list(REFCAP_FOLDER):
        try:
            meta = _entry_meta(name)
        except (OSError, ValueError):
            continue
        views = [{"index": i, "label": r.get("label") or "", "yaw": r.get("yaw"),
                  "size": [r["latent_w"] * 16, r["latent_h"] * 16], "tokens": _ref_tokens(r)}
                 for i, r in enumerate(meta.get("refs") or [])]
        out.append({"name": name, "version": meta.get("version"), "description": meta.get("description", ""),
                    "pixels": bool(meta.get("pixels")), "views": views,
                    "clips": [{"number": k, "label": c.get("label") or "", "frames": c.get("frames"),
                               "audio": bool(c.get("audio_t")), "tokens": _ref_tokens(c)}
                              for k, c in enumerate(_clips(meta), 1)],
                    "audios": [{"number": j, "seconds": round(a["ref_audio_t"] / AUDIO_LATENT_FPS, 1), "tokens": _ref_tokens(a)}
                               for j, a in enumerate(_audios(meta), 1)]})
    return out


def _thumbnail(name, index):
    """One stored view as a small JPEG. Reads just that tensor."""
    meta, path = _entry_meta(name), folder_paths.get_full_path_or_raise(REFCAP_FOLDER, name)
    block = (meta.get("refs") or [])[index]
    key = f"pixels_{index}"
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        stored = {k: f.get_tensor(k) for k in (key, f"{key}_src") if k in keys}
    pixels = _read_pixels(stored, key, block["latent_w"], block["latent_h"])[0]
    image = Image.fromarray((pixels.clamp(0, 1).numpy() * 255).astype(np.uint8))
    image.thumbnail((THUMB_EDGE, THUMB_EDGE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


routes = server.PromptServer.instance.routes


@routes.get("/refcap/library")
async def _library_route(request):
    return web.json_response(_library())


@routes.get("/refcap/thumb")
async def _thumb_route(request):
    name = request.query.get("name", "")
    if name not in folder_paths.get_filename_list(REFCAP_FOLDER):  # combo values are untrusted at this boundary
        return web.Response(status=404, text="no such RefCap")
    try:
        data = _thumbnail(name, int(request.query.get("index", "0")))
    except (IndexError, KeyError, ValueError) as e:
        return web.Response(status=404, text=str(e))
    return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})


NODE_CLASS_MAPPINGS = {
    "H3RefCapSelect": H3RefCapSelect,
    "H3RefCapPackReplay": H3RefCapPackReplay,
    "H3RefCapBake": H3RefCapBake,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefCapSelect": "H3 RefCap Select",
    "H3RefCapPackReplay": "H3 RefCap Replay Pack",
    "H3RefCapBake": "H3 RefCap Bake Pack",
}
