# RefCap vs RefMod

Two ways to put a character into MiniMax H3 video from a file instead of wiring up reference photos every time.

- **RefMod** — [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) by Luisa (luisacaotica), MIT. The first of the two, with an established file format.
- **RefCap** — this pack (`refcap_nodes.py`, `refcap_select.py`, [REFCAP_README.md](REFCAP_README.md)). Written here, inspired by RefMod, with no code taken from it.

They solve the same problem and differ in one decision: **RefMod modifies references, RefCap records them.**

| | RefCap | RefMod |
|---|---|---|
| Idea | Capture what H3's own reference node produced, replay it unchanged | Store a reference latent you can compress, weaken, curve, stack and mix |
| Identity measured (London subjects) | 0.660 at `match`, **0.724** at 1024 px | 0.688 Apply / 0.705 Text Encode (Full), 0.05 (Compressed) |
| Matches native references | **Yes, bit-for-bit** | No, by design |
| File size, 5 photos | 6.5–6.9 MB with PNG sources and no cached rows; 32–163 MB when rows and resized floats are kept | 0.3–0.9 MB (Full), 0.04–0.06 MB (Compressed) |
| Render cost per clip | 37 s (`match`), ~83 s (1024 px) | 46 s (Full + Apply), 56 s (Full + Text Encode), 32 s (Compressed) |
| Controls | Two channel toggles; view selection | Strength, frame curves, denoise-step curves, copies, scrambling, A/B axes, masks |
| Beyond images | View sets (yaw-tagged views), up to 3 clips with soundtracks and 3 audio references per file, packs of several RefCaps, a thumbnail picker | Audio references, bundles, library browser, Continuum bridge |

Everything below explains where those numbers come from and which tool fits which job.

- [1. How H3 takes a reference](#1-how-h3-takes-a-reference)
- [2. What each file stores](#2-what-each-file-stores)
- [3. How each one reaches the model](#3-how-each-one-reaches-the-model)
- [4. Node map](#4-node-map)
- [5. Features only one of them has](#5-features-only-one-of-them-has)
- [6. The measurements](#6-the-measurements)
- [7. Which to use](#7-which-to-use)
- [8. Using both](#8-using-both)
- [9. Provenance and licensing](#9-provenance-and-licensing)

---

## 1. How H3 takes a reference

Every reference photo given to ComfyUI's native `MiniMaxH3ReferenceToVideo` node reaches the DiT through **two channels at once**:

1. **The text encoder (Qwen3-VL-32B).** The tokenizer builds `<Picture 1>: [vision block] <Picture 2>: [vision block] ... <prompt>`, and its output rows (5120 wide) become the conditioning. `minimax_token_tags` marks vision rows 0 and text rows 1.
2. **Reference latents.** The same resized photo is VAE-encoded into `minimax_refs` as an `image` block. `PackedLayout` gives each block its own time slot and 3D RoPE grid; all 50 DiT blocks attend to it on every sampling step.

Each photo costs roughly one token per 32×32 px **in each channel**.

This split is the whole story of the comparison: **RefMod stores channel 2 only**, and reconstructs something for channel 1 when asked. **RefCap stores both**, plus the pixels needed to rebuild channel 1 exactly.

Measured with the channels separated (same file, same seeds): latents alone scored −0.08 against native, rows alone −0.27. Neither half is sufficient on its own.

## 2. What each file stores

### RefCap (`models/refcaps/*.safetensors`, header key `h3_refcap`)

| Key | What it is |
|---|---|
| `rows` `[N, 5120]` fp32 | The text encoder's output for `<Picture i>: ` labels and vision blocks, with no prompt |
| `tags` `[N]` int8 | The modality tag per row |
| `ref_i` `[1, 24, 1, h, w]` | One VAE latent per photo, exactly as the native node encoded it |
| `pixels_i` `[1, 16h, 16w, 3]` fp32 | Each photo resized to the size the native node used |

Version 2 ("view sets") adds a `yaw` and `label` per view, plus an optional `orbit_frames` / `orbit_latent` pair for a turntable video, and drops the cached rows. Version 3 replaces the orbit with up to 3 clips (`clip{k}_frames`, `clip{k}_latent`, optionally `clip{k}_audio_latent`) and adds up to 3 standalone audio latents (`audio{j}_latent`).

### RefMod (`models/refmods/*.safetensors`, header key `refmod_meta`)

| Key | What it is |
|---|---|
| `latent` `[1, 24, T, H, W]` | One latent holding every photo, stacked along time |
| metadata | kind (`image` / `video` / `audio`), mode, pool size, refinement steps, tags, description, optional baked-in Apply settings |

Version 5 files are bundles: several references, each with its own tensor and metadata, in one file.

**The size difference is not compression cleverness.** A latent is small; the text-encoder rows and the stored pixels are what make a RefCap large. Both are now optional: rows are read only by Apply, and each source photo can be kept as a PNG that replay resizes back to exactly the same floats. That takes a five-photo RefCap from 32–163 MB down to 6.5–6.9 MB without changing a pixel the model sees. A Full Reference RefMod is still smaller, at roughly 1 MB, because it stores nothing but the latent — and that is precisely why its latents-only path scored 0.585 against our 0.660.

## 3. How each one reaches the model

### RefMod, path A: *Apply H3 RefMod*

Appends its latent to `minimax_refs` and nothing else. The text encoder never sees the photos, and no `<Picture n>` label exists, so a prompt can't point at the reference. Cheap and fast.

### RefMod, path B: *H3 RefMod Text Encode*

VAE-decodes the stored latent back into a video, hands that to Qwen at 2 fps as `<Video 1>`, and attaches the latent. This restores a version of channel 1, at the cost of a decode and of whatever the latent round trip lost. It scored highest of their paths.

### RefMod's stacking

Several photos become **one video-kind block**: the model sees them with video time spacing, in a slot where it expects causally compressed video frames rather than separate stills. It also resizes on its own terms — short edge `ref_resolution` (default 1024 px), with every photo cropped to the first photo's canvas.

### RefCap, path A: *H3 RefCap Replay* (recommended)

Runs the text encoder over the **stored pixels** plus your prompt, exactly as the native node would with the originals, then attaches the stored latents. Only loading, resizing and VAE-encoding are skipped.

The result is bit-identical to native: every row, every tag, every latent. Clips came out pixel-identical in 12 of 12 comparisons, and again for view sets (9 views plus an orbit).

### RefCap, path B: *H3 RefCap Apply (cached)*

Prepends the cached rows to a prompt that was encoded without the photos, and attaches the latents. Nothing goes through the 32B encoder.

The catch is real but small: Qwen is causal, so natively your prompt's words attend to the photos, and here they can't. Prompt rows differ from native (cosine median 0.97, minimum 0.31). It cost ~0.01 on the two test subjects (0.652 against 0.660).

### RefCap, path C: *H3 RefCap Replay View Set*

Picks views for the shot (nearest to given angles, or an explicit list), then replays them natively, with optional clips as `<Video k>` and audio references as `<Audio j>`. It also prints a label guide (`<Picture 2> = view 7, yaw 90`) so the prompt can name angles.

### RefCap, path D: *H3 RefCap Select* + *H3 RefCap Replay Pack*

Chains one choice of views, clips and audio per RefCap, two characters and a location say, and replays the whole pack natively as one conditioning, renumbered `<Picture 1..N>` with a label guide naming the source of each.

### Side by side

| | Channel 1 (text encoder) | Channel 2 (DiT latents) | Native parity |
|---|---|---|---|
| RefCap Replay | Original pixels, re-encoded with the prompt | One image block per photo | Exact |
| RefCap Apply | Cached rows (prompt encoded separately) | One image block per photo | Latents and tags exact; prompt rows differ |
| RefMod Apply | Nothing | One stacked video block | No |
| RefMod Text Encode | Latent decoded to video, seen at 2 fps | One stacked video block | No |

## 4. Node map

| Job | RefCap | RefMod |
|---|---|---|
| Make a file | **H3 RefCap Capture** (from the native node's conditioning + the same photos), **H3 RefCap Build View Set** (views, clips, audio) | **Create H3 RefMod** / **Create H3 RefMod Master** (own resize and VAE encode; also audio) |
| Use it, exact | **H3 RefCap Replay** | — |
| Use it, cheap | **H3 RefCap Apply (cached)** | **Apply H3 RefMod** |
| Use it with the text encoder | (Replay always does) | **H3 RefMod Text Encode** |
| Pick from many views | **H3 RefCap Build View Set** + **H3 RefCap Replay View Set** | **Load H3 RefMod Axis** (A/B slider), `scramble` subset selection |
| Load several | **H3 RefCap Select** chains + **H3 RefCap Replay Pack**; **H3 RefCap Bake Pack** saves a pack as one file | **Load H3 RefMods** (8 slots, strength and copies each), **Load H3 RefMod Folder** |
| Inspect | file summary lines, the **Pick views...** thumbnail browser, *H3 Save Conditioning* dumps for comparison | **Inspect H3 RefMod** (+ VAE previews), library browser |
| Save settings in the file | — | **Fix H3 RefMod Config** |
| Schedule over denoising | — | **H3 RefMod Step Curve** |
| Debug conditioning | **H3 Save Conditioning** | — |

## 5. Features only one of them has

**Only RefMod:**
- Strength below 1.0, implemented as a blur mix toward a low-passed copy, plus per-frame curves, denoise-step curves, `copies` and deterministic scrambling.
- A/B axis sliders between two references.
- Masks with background retention, `merge` (one consensus latent across many refs), `motion_only` (frame differences).
- Extraction presets, config baked into the file, and a bridge for chunked generation (Continuum).
- A much higher input cap: 16 images plus 8 videos per Create node, and a folder loader for bulk extraction.
- Tiny files, practical to host and share in large numbers.

**Only RefCap:**
- Bit-exact replay of native reference conditioning, verified against the native node (and pixel-identical clips).
- One native `image` block per photo, at the sizes the official node picks.
- The text encoder sees the **original** pixels, never a VAE round trip.
- View sets: yaw-tagged views, selection by shot angle, and a printed label guide.
- Clips with soundtracks and standalone audio references, replayed bit-identically to the native node.
- Several RefCaps in one shot (*Select* + *Replay Pack*), renumbered with a label guide, and baked into one file when worth keeping.

## 6. The measurements

**Setup.** Two consenting London Set people, two scenes, three seeds, 608×352 at 56 frames, 8 steps, on a community H3 turbo finetune (int8 DiT). Scoring is ArcFace similarity to the reference photos; for calibration, two photos of the same person score 0.83–0.89 and the closest stranger 0.20. RefMods were built with their own Create node at its defaults (1024 px, 16×16 grid, 500 refinement steps, photos stacked), using the four paths their own `PLAN_0.3.0.md` proposes to benchmark.

**Identity** (mean over the four London runs, 12 clips per arm):

| Path | Score | Reference tokens | s / clip | File |
|---|---|---|---|---|
| RefCap Replay, 1024 px photos (`max`) | **0.724** | 5,120 | ~83 | 163 MB |
| RefMod Full Reference + Text Encode | 0.705 | 5,120 | 56 | 0.9 MB |
| RefMod Full Reference + Apply | 0.688 | 5,120 | 46 | 0.9 MB |
| Native = RefCap Replay at `match` | 0.660 | 980 | 37 | 32 MB |
| RefCap Apply (cached) | 0.652 | 980 | 37 | 32 MB |
| RefMod Compressed Reference | 0.045–0.065 | 320 | 32 | 0.06 MB |
| No references (control) | 0.005 | 0 | 31 | – |

**Paired, clip by clip:**

| Comparison | Mean | Wins |
|---|---|---|
| RefCap Replay vs native | +0.000 (pixel-identical) | – |
| RefMod Full + Text Encode vs native | +0.046 | 12/12 |
| RefMod Full + Apply vs native | +0.028 | 10/12 |
| RefCap Replay 1024 px vs native | +0.064 | 12/12 |
| **RefCap Replay 1024 px vs RefMod Full + Text Encode** | **+0.018** | **11/12** |

**What it shows:**

1. **At each tool's defaults, RefMod scored a little higher.** Full Reference + Text Encode beat a `match` RefCap by 0.05. It gets there by spending five times the reference tokens (5,120 against 980), because its default resolution is 1024 px while `match` scales photos to the generation's pixel area.
2. **At the same token count, RefCap wins**: 0.724 against 0.705, higher in 11 of 12 clips. Separate image blocks, with the text encoder reading the original photos, use the same budget better than one stacked video block.
3. **RefCap pays for that in time.** At 1024 px the text encoder's 5,160 vision rows also ride through the DiT: ~83 s per clip against 56 s. The disk cost has since gone: that same capture is 6.5 MB stored as PNG sources without cached rows, against RefMod's 0.9 MB. The file sizes in the tables above are the older storage the runs used.
4. **RefMod's Compressed Reference kept no facial identity** — at stranger level in every clip, indistinguishable from rendering with no reference at all. Their own README warns that pooling destroys faces; this measures it.
5. **Reference resolution mattered more than anything else** on either side: +0.064 for RefCap from `match` to 1024 px, and RefMod's lead at defaults is the same effect.

**Fairness caveats.** RefMod was tested at its defaults with stacked photos. Per-photo image mods, larger Compressed grids, `merge`, `copies`, strength curves, masks and its audio features were not tried, and could change results. One DiT, one resolution, short clips, two subjects, and ArcFace only scores faces.

**A RefCap-only result, for scale.** A view-set test (one Multiface subject, 3 shots × 3 seeds, ~5,000 tokens per arm) found that spreading the budget around the head instead of onto five sharp front and 3/4 views changed the face score by less than 0.012 overall — but it recovered a hair clip visible only from the side and back in 9 of 9 clips, against 0 of 3 for front-only references. Coverage pays at |yaw| ≥ 70° and for what a face metric can't see.

## 7. Which to use

**Use RefMod when:**
- You want small, shareable files: a personal library of dozens or hundreds of characters.
- You want to dial a reference down, fade it across the reference frames or the denoising schedule, repeat it, or blend between two subjects.
- You need masks, `merge`, `motion_only`, or chunked generation through Continuum.
- Speed matters more than the last few points of likeness, especially through its Apply path.

**Use RefCap when:**
- You want exactly what the native reference node gives, without re-wiring photos, and want to be able to prove it (*H3 Save Conditioning* dumps both conditionings for a comparison).
- You want the most identity per reference token, and can afford the render time.
- The shot turns away from the camera, and you want per-shot view selection or an orbit video.
- You want several characters and a place in one shot, each from its own file.
- You want to measure identity rather than judge it by eye.

**Use neither** if one photo wired straight into `MiniMaxH3ReferenceToVideo` covers your case. RefCap replays that exactly; it exists for convenience and repeatability, not for extra quality at the same settings.

**Rules of thumb:**
- Build your references at ~1024 px if identity is the goal, in either tool.
- Don't use RefMod's Compressed mode for faces.
- Compare any two reference methods at matched reference tokens; the first comparison here was confounded by resolution.

## 8. Using both

They are independent packs and can be installed together. *Apply H3 RefMod* appends to whatever `minimax_refs` the conditioning already has, so a RefCap Replay output could, mechanically, be extended with a RefMod block — for example a person from a RefCap and a location from a RefMod. Two cautions: the `<Picture n>` labels come only from the RefCap's photos, so a prompt can't address the RefMod block; and this combination is **untested** here.

The reverse doesn't work: *H3 RefCap Apply* refuses conditioning that already holds references, because its cached rows are numbered from `<Picture 1>` and would collide.

The file formats are not interchangeable. A RefMod latent could in principle be loaded as a RefCap reference block, but no converter exists in either direction.

## 9. Provenance and licensing

- **RefMod** (`ComfyUI-MiniMaxH3Mod`): MIT, © 2026 Luisa (luisacaotica). This comparison reads and measures commit `8e402db`. It was installed only for the comparison; none of its code is used in RefCap. It does no global patching of ComfyUI or model code — it registers its nodes, a local `/refmods/library` route, and canvas JavaScript.
- **RefCap**: written in this pack. It calls ComfyUI's own H3 node, tokenizer and resize helpers rather than reimplementing them, which is why exact parity is possible and why it follows core changes automatically.
- The idea of saving H3 references to a small file is RefMod's; RefCap's contribution is storing both channels and the pixels, so a replay is indistinguishable from the real thing.

**Reproducing the comparison:** the comparison ran with an unpublished harness of render and scoring scripts against a running ComfyUI (see [REFCAP_README.md](REFCAP_README.md), "Reproducing the tests"). Its inputs can't be redistributed, so the numbers here are the record.
