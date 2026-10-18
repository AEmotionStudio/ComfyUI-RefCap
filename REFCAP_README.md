# RefCap: reference capture for MiniMax H3

A **RefCap** is one file per person (or object, or place) that brings their look into MiniMax H3 video, instead of wiring up reference photos every time.

It captures what H3 takes from reference photos, unmodified: the text encoder's view of each photo, the VAE latents the video model attends to, and the resized photos themselves. **H3 RefCap Replay** turns a RefCap plus a prompt into conditioning identical to the native *MiniMax H3 Reference to Video* node fed the original photos; in testing, its clips matched native references pixel for pixel. Captured from ~1024 px photos, a RefCap kept more facial identity than ComfyUI-MiniMaxH3Mod's RefMods at the same reference token count in 11 of 12 clips, at a higher render cost.

Nothing is trained and no weights change. A RefCap is not a LoRA: it is a recording of reference inputs the model already knows how to use. The name contrasts it with **RefMod**, which *modifies* references (compresses, stacks, blurs, curves them); a RefCap *captures* them and replays them as they were.

- [How H3 uses a reference photo](#how-h3-uses-a-reference-photo)
- [What a RefCap stores](#what-a-refcap-stores)
- [The nodes](#the-nodes)
- [Workflows](#workflows)
- [Implementation notes](#implementation-notes)
- [Evaluation](#evaluation)
- [How RefCap differs from RefMod](#how-refcap-differs-from-refmod)
- [View sets (billboards)](#view-sets-billboards)
- [Reproducing the tests](#reproducing-the-tests)
- [Limitations](#limitations)
- [Files](#files)
- [Credits and licenses](#credits-and-licenses)

---

## How H3 uses a reference photo

In ComfyUI's native reference-to-video path (`comfy_extras/nodes_minimax_h3.py`, `MiniMaxH3ReferenceToVideo`), every reference photo reaches the DiT through **two channels at once**.

1. **Text encoder (Qwen3-VL-32B, truncated to 50 layers).** The tokenizer (`comfy/text_encoders/minimax.py`) builds the presentation `<Picture 1>: [vision block] <Picture 2>: [vision block] ... <prompt>`. The encoder's last hidden states become the conditioning rows, 5120 wide. `minimax_token_tags` marks each row: 0 for the vision block (video modality), 1 for text. The DiT picks its adaLN modulation per row from these tags.
2. **Reference latents.** The same photo, resized, is VAE-encoded and stored in `minimax_refs` as an `image` block. `PackedLayout` (`comfy/ldm/minimax/model.py`) places each block between the text and the target streams with its own 3D RoPE grid, advancing the time axis by 1.0 per photo. The blocks are pinned near t = 1 and attended to by all 50 blocks on every step.

Each photo costs about one token per 32×32 px in **each** channel. At `ref_image_size = match`, photos are scaled to the generation's pixel area: at 608×352 a square photo costs about 200 text rows and 196 latent tokens. A 1024×1024 photo at `max` costs about 1,026 rows and 1,024 latent tokens.

Both channels matter. Rendering with only one, taken from the same RefCap, lost identity (see [Evaluation](#evaluation)): latents alone −0.08, rows alone −0.27 against native.

## What a RefCap stores

A `.safetensors` file in `ComfyUI/models/refcaps/`:

| Key | Shape / dtype | What it is |
|---|---|---|
| `rows` | `[N, 5120]` float32 | The text encoder's output for the reference presentation: every `<Picture i>: ` label row and vision block, with no prompt. Kept in the encoder's output dtype (fp32). |
| `tags` | `[N]` int8 | `minimax_token_tags` for those rows (0 = vision, 1 = text). |
| `ref_0` … `ref_{k-1}` | `[1, 24, 1, h, w]` | The VAE latent of each photo, exactly as the native node encoded it. |
| `pixels_0` … | `[1, 16h, 16w, 3]` float32 | Each photo resized to the size the native node used (16 px per latent cell). Present when the photos were connected at capture time. |
| `pixels_0_src` … | uint8 | The source photo as a PNG, stored instead of the resized floats when it is smaller (`compact`). Replay decodes it and redoes the reference node's resize, which reproduces those floats bit for bit. |

The header carries JSON metadata under the key `h3_refcap`:

```json
{"format": "h3_refcap", "version": 1, "description": "...", "pixels": true, "rows": false,
 "refs": [{"kind": "image", "latent_h": 22, "latent_w": 28}, ...]}
```

**Size.** Two switches on *H3 RefCap Capture* decide it, and neither changes what the model sees:
- `store_rows` (default off): rows and tags are read only by *H3 RefCap Apply*. Replay rebuilds them from the pixels.
- `compact` (default on): each source photo is kept as a PNG when that is smaller than the resized floats. A PNG decodes to exactly the 8-bit source the resize saw, so replay stays bit-identical; sources that aren't exactly 8-bit (a VAE-decoded batch) keep the floats automatically.

Measured on the same five London Set photos:

| Capture | Size |
|---|---|
| rows + resized fp32 pixels (how the first RefCaps were stored) | 31.8 MB |
| rows + PNG sources | 26.9 MB |
| PNG sources, no rows, `match` (1350 px sources) | 6.9 MB |
| PNG sources, no rows, 1024 px sources at `max` | **6.5 MB**, against 162.7 MB stored the old way |

File size now tracks the **source** photos rather than the encode resolution, which is why the 1024 px capture is no bigger than the `match` one. View sets, which never store rows, shrink the same way.

## The nodes

These four nodes live in `refcap_nodes.py`, category `advanced/refcap`, next to the view-set nodes and *H3 RefCap Orbit Views* described in [View sets](#view-sets-billboards). Select, Replay Pack and Bake are in `refcap_select.py`.

### H3 RefCap Capture

| Input | |
|---|---|
| `conditioning` | Output of *MiniMax H3 Reference to Video* with an **empty prompt** and image references only. |
| `name` | File stem in `models/refcaps/`. |
| `description` | Free text, stored in the metadata. |
| `image_1` … `image_9` | Optional: the same photos, in the same order, as the reference node. Needed for *Replay*. |

What it checks and stores:
- The conditioning must carry `minimax_token_tags` and image-only `minimax_refs`.
- The last row must be a vision token. A conditioning that ends in text means a prompt was encoded, and is rejected.
- If photos are connected, their count must equal the reference count.
- Each photo is resized with ComfyUI's own `common_upscale(..., "lanczos", "disabled")`, the call the native node's `_resize` makes, to the size its latent records (`latent_w·16 × latent_h·16`). The stored pixels are therefore the ones the native text encoder saw.
- Output: a summary line with photo count, rows, vision rows, reference tokens and file size.

### H3 RefCap Replay (recommended)

| Input | |
|---|---|
| `clip` | The MiniMax H3 text encoder. |
| `prompt` | Your prompt, referring to the photos as `<Picture 1>` … `<Picture k>`. |
| `refcap` | A RefCap captured **with** its photos. |

It builds the reference presentation from the stored pixels (`clip.tokenize(prompt, minimax_ref_items=[{"type": "image", "data": pixels}, ...])`), encodes it, and attaches the stored latents with `conditioning_set_values(..., {"minimax_refs": blocks})`. That is the native node's own code path, minus loading, resizing and VAE-encoding the photos. Take the latent from *Empty MiniMax H3 AV Latent*.

The conditioning it produces is bit-identical to native: every row, every tag, every latent (checked with *H3 Save Conditioning* dumps of both). Renders were pixel-identical in 12 of 12 comparisons.

### H3 RefCap Apply (cached)

| Input | |
|---|---|
| `conditioning` | A prompt encoded **without** references, e.g. *MiniMax H3 Reference to Video* with no photos connected (which also makes the latent). |
| `refcap` | Any RefCap. |
| `rows` | Prepend the cached `<Picture i>` rows and their tags. |
| `latents` | Attach the cached reference latents. |

The photos never pass through the text encoder. The trade-off is that the prompt rows are not native. Qwen is causal: natively, the prompt tokens after the vision blocks attend to them, but here the prompt was encoded alone. Against a native encode, prompt rows had cosine median 0.97 and minimum 0.31. On the two test subjects that cost 0.008 ArcFace on average (0.652 against 0.660).

It refuses conditioning that already contains image rows (any tag 0) or `minimax_refs`. RefCap rows are numbered `<Picture 1..k>`, and blocks must line up with labels, so mixing references would collide.

The `rows` / `latents` toggles exist for experiments: they isolate the two channels.

### H3 Save Conditioning (debug)

Writes a conditioning's rows, tags and reference latents to `output/<filename>.safetensors`, for comparisons like the parity check.

### H3 RefCap Select, Replay Pack and Bake Pack

One RefCap holds one subject, so a scene with two characters, or a character and a location, takes one *H3 RefCap Select* per file, chained: each picks `views` (`all`, `none`, or 0-based numbers and ranges such as `0,2,5-7`) and optionally `clips` and `audios` (1-based), and adds them to a pack. *H3 RefCap Replay Pack* renumbers the whole pack into `<Picture 1..N>`, `<Video k>` and `<Audio j>` in chain order, encodes the prompt with every selected photo, and returns the conditioning plus a label guide naming which RefCap each label came from. It is bit-identical to *MiniMax H3 Reference to Video* fed the same files: verified on three RefCaps at once (rows, tags and all eight reference latents).

Press **Pick views...** on a Select node for a thumbnail browser of `models/refcaps`, drawn from the photos each RefCap stores; it counts images and reference tokens against the reference node's caps (9 images, 3 clips, 3 audio references per render).

*H3 RefCap Bake Pack* writes the current selection to one RefCap for a cast you keep reusing. It copies stored pixels and latents without re-encoding, and the result replays identically through *H3 RefCap Replay View Set*; re-bake after re-capturing one of the sources. Bake doesn't enforce the per-render caps (Replay Pack and Replay View Set do), so a baked file can hold more than 9 views (a face set plus a body set, say), and Replay View Set picks up to 9 of them per shot.

Photos loaded from a folder by *H3 RefCap Build View Set* and photos loaded by *Load Image* are not quite the same for JPEG sources: ComfyUI decodes JPEGs through PyAV, the folder loader through PIL, and they differ by about 0.5/255. PNG sources agree exactly. Two RefCaps built from the same JPEG through the two paths therefore hold slightly different pixels.

## Workflows

Shipped in [`example_workflows/`](example_workflows/), so ComfyUI lists them under this pack in its template browser. Each has a *Read me* note; the generate workflows also have a *Prompt guide*. Their file inputs (photos, clips, audio, RefCaps) name the files the tests used: load your own in their place.

| Workflow | Does |
|---|---|
| **H3 RefCap 1 - Capture** | Photos → *MiniMax H3 Reference to Video* (empty prompt) → *H3 RefCap Capture*, with the same photos wired to both. |
| **H3 RefCap 2 - Replay (exact)** | *H3 RefCap Replay* + *Empty MiniMax H3 AV Latent* → sampler → video with audio. |
| **H3 RefCap 3 - Apply (cached)** | Prompt-only reference node → *H3 RefCap Apply (cached)* → sampler. |
| **H3 RefCap 4 - AB native vs RefCap** | One shared prompt node drives three branches on one seed: A native photos, B Replay, C Apply. Check a RefCap against its photos on your own character. |
| **H3 RefCap 5 - View set** | *H3 RefCap Build View Set* (muted until you need it) and *H3 RefCap Replay View Set* with a label guide → sampler. |
| **H3 RefCap 6 - Clips and audio** | *Load Video* + *Get Video Components* and two *Load Audio* nodes feed *H3 RefCap Build View Set* (muted) as a clip with its soundtrack and two audio references; *H3 RefCap Replay View Set* with `audios` = `1` and a dialogue prompt that binds `<Audio 1>` to the subject → sampler. |
| **H3 RefCap 7 - Two characters** | Three chained *H3 RefCap Select* nodes (two people and a location) → *H3 RefCap Replay Pack* with a label guide → sampler, plus a muted *H3 RefCap Bake Pack*. |
| **H3 RefCap 8 - Photo to RefCap** | One photo → two H3 orbits at 768² (round the subject's left and right side) → two *H3 RefCap Orbit Views* → two *H3 RefCap Build View Set*: a 24-view turnaround and the front 5 (0, ±20, ±45°) ; a body section adds a full-body photo (given, or the last frame of a short H3 render from the head shot, picked by an *If/Else Switch*), two 576×1024 full-body orbits that see the head shot and that photo and a `my_character_body` RefCap at 0, ±45, ±90 and 180°. A test render then shows a full-body walk from the front 5 plus body views at 0, ±90 and 180°, with the outfit bound to the body pictures. It runs at `match`, because nine pictures at `max` run the text encoder out of memory on 12 GB after the renders. All in one queue, about 8 minutes. The canvas version of the [H3 bootstrap](#bootstrapped-with-h3); read its results first. |
| **H3 RefCap 9 - Location** | A location photo, either given or the last frame of a short H3 render from a description (*If/Else Switch*), goes through a 180° pan to the right, then a 180° pan to the left that sees the right pan's last frame, so both end on the same wall behind. *H3 RefCap Orbit Views* (`measure` = `motion`) labels the frames by camera heading, and *H3 RefCap Build View Set* saves them. A location-only test clip follows. All in one queue, about 4 minutes. |
| **H3 RefCap 10 - Audio to RefCap** | *Load Audio* (an audio file, or a video's audio track) → *Trim Audio Duration* → *H3 RefCap Build View Set* with only `audio_1` connected: a voice RefCap with no pictures. A test clip feeds the same audio to *MiniMax H3 Reference to Video* as `<Audio 1>` with the dialogue prompt from workflow 6. All in one queue, about 50 s. |
| **H3 RefCap 11 - Video to RefCap** | *Load Video* → *Trim Video* → *Get Video Components* (its frame rate shown, since H3 reads reference frames as 24 fps) → *Scale Image to Max Dimension* (608 px) → *H3 RefCap Build View Set* with the frames as clip 1 and `orbit_length` 124 (5.2 s). A 352×608 test clip feeds the same frames to *MiniMax H3 Reference to Video* as `<Video 1>`, with a woman described in words performing the clip's movement. All in one queue. See [Video references](#video-references-2026-10-07). |
| **H3 RefCap 12 - Motion** | Two chained *H3 RefCap Select* nodes, a character and a move (a clip RefCap: `views` = `none`, `clips` = `1`), → *H3 RefCap Replay Pack* → a 352×608 × 124 full-body shot with the move bound to the character in the prompt. Example moves from Pexels: `move_jumping_jacks`, `move_yoga`, `move_hip_hop`, `move_karate`. |
| **H3 RefCap 13 - Camera move** | Three chained *H3 RefCap Select* nodes, a character, a room (workflow 9's example, views 1-3) and a camera clip RefCap, → *H3 RefCap Replay Pack* → a 608×352 × 124 shot with the clip marked as the camera-movement reference. Example moves: `camera_pan_left`, `camera_push_forward`, `camera_tilt_down`, `camera_orbit`. Measured: the words carry the move and the clip adds nothing (see Test 2). |
| **H3 RefCap 14 - Edit a clip** | *Load Video* of a finished clip → *Get Video Components* → *MiniMax H3 Reference to Video* with the frames as `<Video 1>`, the soundtrack reused as `<Audio 1>` and the character's photos, rendered at the clip's own length and seed. The prompt is the clip's own, rewritten as a `video editing` task with one change (default: the sweater made red). One queue. |
| **H3 RefCap 15 - Location walkthrough** | *Load Video* of a walk through a room → *Trim Video* → *Get Video Components* → *Scale Image to Max Dimension* → four evenly spaced frames (*Get Image from Batch* + *Batch Images*) → *H3 RefCap Build View Set* as views. A test shot puts the character (photos) in the room with those views as `<Picture 6-9>`. One queue, about 80 s. Measured: four stills keep a room as well as or better than the walk as `<Video 1>`, at a fifth of the tokens. |
| **H3 RefCap 16 - Fantasy cast** | Five chained *H3 RefCap Select* nodes, the elf ranger and the dwarf warrior (views 1-3 of each `_front`), the crystal cavern (views 0-2) and both voice RefCaps (`audios` = `1`), → *H3 RefCap Replay Pack* → the README's 960×544 × 158 two-shot with both speaking. The RefCaps are on [Hugging Face](https://huggingface.co/AEmotionStudio/H3-RefCap), with the tiefling sorceress and the wizard study. |

They load the official H3 reference model (`minimax_h3_ref2va_pruned_int8_convrot` with the 4-step ref2v turbo LoRA, as ComfyUI's H3 reference template does) and start small (608×352). The measurements in this file were made on a community H3 turbo finetune (int8) at shift 6/3, 8 steps, `res_multistep` + `simple`, so results on the official model may differ. Raise size and length for final renders.

**Prompting.** H3's reference mode reads a structured prompt (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`). Define the person once, for example: `<Subject 1> is the woman in <Picture 1>, <Picture 2> and <Picture 3>, a white woman in her mid-twenties with dark brown wavy hair`. Then describe the clothes, action, camera and sound you want. The tests never named anyone. MiniMax's guide is `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` in the MiniMax-H3 repository.

## Implementation notes

**Why capture through the native node.** Sizing, resizing, VAE encoding and the presentation all come from ComfyUI's own `MiniMaxH3ReferenceToVideo`. Capture only records its outputs. That is what makes exact parity possible, and it follows core changes automatically.

**Why store pixels.** The rows can't be re-used exactly once a prompt is added, because the prompt must be encoded after the vision blocks. Re-running the text encoder over the original pixels is the only exact route. Pixels are cheaper than rows, and they avoid a lossy VAE round trip (decoding the latent back to an image), which is what the RefMod Text Encode node does.

**Why fp32 rows.** ComfyUI's CLIP wrapper returns float32. Keeping it avoids a precision loss on top of the encoder's own variation.

**The rows are not bit-stable anyway.** With the nvfp4 text encoder, the reference rows changed by about 0.03% (median relative norm; max 0.9%) whenever the prompt after them changed. Row 0 changed too, and in a causal model the first token can't see later ones. So the shift is numeric and depends on sequence length. A native encode with a different prompt shows the same drift. Replay sidesteps it by encoding the identical token sequence.

**Reference size is the biggest lever.** With `match`, photos are scaled to the pixel area of the width/height set on the reference node at capture time. A RefCap captured at 608×352 keeps 608×352-sized references even when you later render at 1344×768: it still works, with less reference detail. With `max`, photos keep up to a 2048 px short edge. A RefCap captured from 1024 px photos at `max` scored +0.06 over `match`, higher in 12 of 12 clips. It cost about 2.2× the render time at 608×352, and a 163 MB file.

**Nine photos per capture.** *Capture* records the native node, which caps `ref_images` at 9. *Build View Set* and *Bake Pack* write larger sets, and several RefCaps combine in one shot through *Select* and *Replay Pack*. Replay View Set and Replay Pack cap each render at the native node's 9 images, 3 clips and 3 audio references; Replay and Apply have no cap.

**Label and block order.** Rows carry `<Picture 1..k>` labels in the order the blocks are attached. Apply refuses conditioning that already holds references, so labels and blocks can't drift apart.

## Evaluation

### Method

- **Subjects.** Two people drawn at random (seeded) from the **Face Research Lab London Set**: 013, woman, 26; 117, man, 26. All subjects in that set signed consent for their images to be used "in lab-based and web-based studies in their original or altered forms". Each contributed 5 views, 1350×1350: neutral and smiling front, 3/4 left and right, smiling 3/4 left.
- **Scenes.** A sunny café medium close-up and a lamplit night close-up. The prompts describe age, gender and hair, set clothing different from the photos, and never name anyone.
- **Settings.** 608×352, 56 frames, 8 steps, shift 6/3, `res_multistep` / `simple`, seeds 4242, 777, 31337. A community H3 turbo finetune (int8 DiT) and the nvfp4 Qwen3-VL-32B encoder. Every clip was rendered as its own prompt, with models freed between clips.
- **Arms.** The IDs predate the RefCap name and are kept in the result files: *encode* is Replay, *pack* is Apply.

  | Arm | What it is |
  |---|---|
  | `1native` | Native reference node with the photos (`match`) |
  | `2encode` | H3 RefCap Replay (`match` capture) |
  | `2encode_1024` | H3 RefCap Replay, captured from the photos resized to 1024 px with `max` |
  | `2pack` | H3 RefCap Apply (cached), rows + latents |
  | `3latents` | Apply, latents only |
  | `4rows` | Apply, rows only |
  | `5none` | Prompt only (control) |
  | `6rm_full_apply` | RefMod *Full Reference* + *Apply H3 RefMod* |
  | `7rm_full_text` | RefMod *Full Reference* + *H3 RefMod Text Encode* |
  | `8rm_comp_apply` | RefMod *Compressed Reference* + *Apply H3 RefMod* |
  | `9rm_comp_text` | RefMod *Compressed Reference* + *H3 RefMod Text Encode* |

- **RefMod arms.**
  - Created with ComfyUI-MiniMaxH3Mod's own *Create H3 RefMod* node at its defaults: `ref_resolution` 1024, a 16×16 grid, 500 refinement steps, a 5,120-token cap, the photos stacked.
  - They are the four paths the project's own `PLAN_0.3.0.md` proposes to benchmark.
  - Their prompts are identical except the reference label is `<Video 1>`, which is what their Text Encode node calls a stacked RefMod.
- **Scoring.** insightface buffalo_l (ArcFace w600k_r50) on the CPU. For each clip, 8 evenly spaced frames; the largest face; cosine similarity to the mean embedding of the reference photos; averaged over frames.
- **Calibration.**
  - *Same person*: each reference photo against the mean of the others (leave-one-out).
  - *Strangers*: the reference mean against the neutral front photos of the other London Set people.
  - A clip counts as **recognisable** if it scores above that subject's highest stranger.
- **Visual check.** Every run got contact sheets of the middle and last frames and of the face crops, and an overview sheet shows one crop per run, seed and arm. The sheets agree with the scores.

### Calibration

| Subject | Same person (photo vs photo) | Strangers: mean / p95 / max |
|---|---|---|
| London 013 | 0.829 (0.771–0.868) | 0.022 / 0.121 / 0.175 |
| London 117 | 0.887 (0.874–0.908) | 0.019 / 0.130 / 0.195 |

### Results

Mean ArcFace similarity per run (3 seeds each). *Mean* is over the four runs.

| Arm | lf013 café | lf013 night | lf117 café | lf117 night | **Mean** | Recognisable |
|---|---|---|---|---|---|---|
| Native (`1native`) | 0.624 | 0.649 | 0.659 | 0.707 | **0.660** | 12/12 |
| RefCap Replay (`2encode`) | 0.624 | 0.649 | 0.659 | 0.707 | **0.660** | 12/12 |
| RefCap Replay, 1024 px (`2encode_1024`) | 0.677 | 0.704 | 0.741 | 0.774 | **0.724** | 12/12 |
| RefCap Apply (`2pack`) | 0.629 | 0.642 | 0.627 | 0.709 | **0.652** | 12/12 |
| Apply, latents only (`3latents`) | 0.521 | 0.571 | 0.578 | 0.668 | **0.585** | 12/12 |
| Apply, rows only (`4rows`) | 0.380 | 0.332 | 0.392 | 0.459 | **0.391** | 12/12 |
| No references (`5none`) | −0.010 | 0.004 | −0.020 | 0.045 | **0.005** | 0/12 |
| RefMod Full + Apply (`6rm_full_apply`) | 0.615 | 0.670 | 0.704 | 0.761 | **0.688** | 12/12 |
| RefMod Full + Text Encode (`7rm_full_text`) | 0.654 | 0.694 | 0.711 | 0.763 | **0.705** | 12/12 |
| RefMod Compressed + Apply (`8rm_comp_apply`) | −0.004 | −0.001 | 0.079 | 0.105 | **0.045** | 0/12 |
| RefMod Compressed + Text Encode (`9rm_comp_text`) | −0.001 | −0.005 | 0.136 | 0.129 | **0.065** | 0/12 |

Paired differences over matching run and seed (12 clips):

| Comparison | Mean | Range | Wins |
|---|---|---|---|
| Replay vs native | +0.000 (clips pixel-identical) | – | – |
| Apply vs native | −0.008 | −0.090 to +0.046 | 6/12 |
| Latents only vs native | −0.075 | −0.130 to +0.031 | 1/12 |
| Rows only vs native | −0.269 | −0.347 to −0.162 | 0/12 |
| RefMod Full + Apply vs native | +0.028 | −0.068 to +0.095 | 10/12 |
| RefMod Full + Text Encode vs native | +0.046 | +0.015 to +0.109 | 12/12 |
| Replay 1024 px vs native | +0.064 | +0.034 to +0.136 | 12/12 |
| **Replay 1024 px vs RefMod Full + Text Encode** | **+0.018** | −0.055 to +0.058 | **11/12** |
| RefMod Full + Apply vs RefMod Full + Text Encode | −0.018 | −0.100 to +0.016 | 3/12 |

### Cost

Mean seconds per clip for the London runs (608×352, 56 frames, 8 steps, including reloading models for every clip). Reference size is for five photos.

| Path | Latent tokens | Text rows | s / clip | File |
|---|---|---|---|---|
| No references | 0 | 0 | 31.0 | – |
| RefMod Compressed + Apply / + Text Encode | 320 | none / from decoded video | 31.5 / 32.9 | 0.06 MB |
| RefCap Apply (cached) | 980 | 1,020 (cached) | 36.5 | 31.8 MB |
| Native / RefCap Replay | 980 | 1,020 | 38.4 / 37.2 | 31.8 MB |
| RefMod Full + Apply | 5,120 | none | 46.2 | 0.94 MB |
| RefMod Full + Text Encode | 5,120 | from decoded video | 56.0 | 0.94 MB |
| RefCap Replay, 1024 px | 5,120 | 5,160 | ~83 | 162.7 MB |

RefCap Replay at 1024 px: 11 of 12 clips took 81–84 s; one outlier took 162 s.

### What the results show

1. **Replay is native.** Conditioning bit-identical, frames pixel-identical.
2. **Apply is close to native:** 0.652 against 0.660, higher in 6 of 12 clips.
3. **Both channels are needed.** Latents alone keep most of the identity; rows alone keep less but are still clearly the person.
4. **At their defaults, RefMod Full Reference scores a little higher than a `match` RefCap:** +0.05 with Text Encode, +0.03 with Apply. But it spends five times the reference tokens (5,120 vs 980).
5. **At the same 5,120 latent tokens, RefCap wins:** 0.724 vs 0.705 (Text Encode) and 0.688 (Apply), higher than RefMod's best path in 11 of 12 clips. One image block per photo, with the text encoder seeing the original photos, uses the resolution better than one stacked video block.
6. **That win costs time and disk.** At 1024 px the text encoder's 5,160 vision rows also ride through the DiT: about 83 s per clip against 56 s (Text Encode) and 46 s (Apply), and a 163 MB RefCap against a 1 MB RefMod.
7. **RefMod Compressed Reference kept no identity.** It scored at stranger level, like the no-reference control, in every clip.
8. **References didn't leak clothing.** The London photos' white t-shirts never appeared, in any arm; renders followed the prompt's clothes and scene.

## How RefCap differs from RefMod

[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) (MIT, © Luisa) packages H3 references as "RefMod" files. It inspired this work; no code was taken from it. The comparison reads, and was measured against, commit `8e402db`.

The summary below is the short version. [REFCAP_VS_REFMOD.md](REFCAP_VS_REFMOD.md) covers it in full: both file formats, every path into the model, a node map, the features each one has alone, the measurements with their caveats, and which to use when.

| | RefCap | RefMod |
|---|---|---|
| **What is stored** | Text-encoder rows + tags, one VAE latent per photo, resized photos | One VAE latent: the photos stacked along time into a single video-kind block |
| **Created by** | ComfyUI's native reference node (native resize, VAE encode, presentation) | The pack's own Create node: its own resize (short edge `ref_resolution`, default 1024 px; photos share the first photo's canvas, others center-cropped), its own VAE encode |
| **Photos in the DiT** | One `image` block per photo, as native | One `video` block with k latent frames: video RoPE time spacing, and frames the model expects to be causally compressed video |
| **Text-encoder channel** | *Replay*: stored photos + prompt, identical to native. *Apply*: cached rows. | *Apply*: none; only `minimax_refs` is appended. *Text Encode*: VAE-decodes the stored latent into a video and presents it to Qwen at 2 fps as `<Video 1>`. |
| **Parity with native** | Replay: bit-identical conditioning, pixel-identical clips | Not native by design |
| **Compression** | None | *Compressed Reference*: latent average-pooled to a grid (default 16×16), then optimized so its trilinear upscale matches the full latent (MSE, no model) |
| **Strength controls** | Channel toggles only | Blur-mixing strength, per-frame curves, denoise-step curves, copies, scrambling, A/B axis sliders |
| **Other features** | Clips with soundtracks and audio references (version 3), packs of several RefCaps, a thumbnail picker | Audio references, bundles, a library browser, config baked into files, a Continuum bridge |
| **Identity, measured** | `match`: 0.660; 1024 px: **0.724** (London) | Full: 0.688 (Apply), 0.705 (Text Encode); Compressed: 0.045–0.065, none |
| **Render time, measured** | 37 s (`match`), ~83 s (1024 px) | 46 s (Full + Apply), 56 s (Full + Text Encode), 32 s (Compressed) |
| **Size, 5 photos** | 32 MB (`match`), 163 MB (1024 px) | Full Reference 0.3–0.9 MB (1,760–5,120 tokens); Compressed 0.04–0.06 MB (240–320 tokens) |
| **Photo limit** | 9 per capture and per render; view sets and baked packs hold more | 16 images + 8 videos per Create node, plus a folder loader |

In short:
- **RefMod** is a flexible, compact reference *editor*: tiny files, many controls, fast Apply, and references that can be weakened, curved, stacked and mixed. Its Full Reference mode carries identity well; its Compressed mode doesn't carry faces.
- **RefCap** is a faithful reference *recording*: large files, few controls, output identical to wiring the photos in, and the most identity per reference token in these tests, paid for in text-encoder rows.

## View sets (billboards)

A **view set** is a version-2 RefCap: many views of one subject, each tagged with the yaw it was seen from, plus an optional orbit video. Version 3 (2026-09-15) adds what the native node takes beyond that: up to three clips, each with an optional soundtrack, and up to three standalone audio references; see [Clips and audio](#clips-and-audio-version-3).

The idea comes from **TURNtable Viewer**, a Blender "multi-directional billboard" add-on:
- A plane faces the camera and shows one frame of a 360° turntable image sequence.
- The frame is the one matching the camera's horizontal angle: `floor(azimuth / (360 / N))`.
- It has no depth, no pitch and no blending between frames.

The question for H3: would all-around views keep a character's identity better than front and 3/4 photos, "a bit like a 3D character"?

### What H3 can and can't do with angles

**No lookup by angle.**
- Each image reference block gets RoPE over (time, h, w) only, at its own time slot (`PackedLayout` in `comfy/ldm/minimax/model.py`).
- Every block is attended to by all 50 DiT blocks on every step.
- There is no camera or view input.
- The model copies what it sees in context; it doesn't reconstruct a head.
- So the billboard's lookup has to happen **before** encoding: pick the views that suit the shot, and name their angles in the prompt.

**Tokens limit coverage.** Detail per view was the biggest lever measured (+0.064 at 1024 px). About 5,000 reference tokens buys one of:
- 5 views at 1024 px
- 9 views at 736 px
- a 56-frame orbit at 544 px (17 latent frames)

The native node also caps references at 9 images.

**A video reference is H3's native billboard.** A turntable clip passed as `<Video 1>` is a continuous sweep of angles, and H3 was trained to read reference videos.
- The text encoder sees it at 2 fps, so 5 frames of a 56-frame orbit.
- H3 cuts a reference video to the length of the generation.

### Format

Same folder (`models/refcaps/`), same `h3_refcap` header key, `"version": 2`.

| Key | What it is |
|---|---|
| `pixels_i`, `ref_i` | Each view resized and VAE-encoded exactly as *MiniMax H3 Reference to Video* does it, as in version 1 |
| `orbit_frames` | The orbit's frames the text encoder sees (every 12th frame, 2 fps) |
| `orbit_latent` | The orbit's video latent `[1, 24, T, h, w]` |

The metadata lists, per view, `latent_h`, `latent_w`, `yaw` and `label` (the file name). It also describes the orbit block: `frames`, `latent_t`, `latent_h`, `latent_w` and `timestamps`.

**There are no cached rows.** Replay rebuilds them from the pixels, so any subset of views replays natively. *H3 RefCap Apply (cached)* needs cached rows, so it refuses view sets with `rows` on; with `rows` off it attaches only their image latents.

**Version 3** keeps every key above and replaces the single orbit with numbered clips and audio:

| Key | What it is |
|---|---|
| `clip{k}_frames`, `clip{k}_latent` | Clip k (1-3): the 2 fps frames the text encoder sees, and its video latent |
| `clip{k}_audio_latent` | Clip k's soundtrack as the audio VAE's latent `[1, 32, 2, T]`, when it has one |
| `audio{j}_latent` | Standalone audio reference j (1-3), the same latent form |

The metadata's `clips` list has `frames`, `latent_t`, `latent_h`, `latent_w`, `timestamps`, `audio_t` (0 without a soundtrack) and `label`; `audios` has `ref_audio_t` and `label`. Audio holds no pixels-equivalent because the text encoder never sees audio: the native node gives it a bare `<Audio j>: ` label and hands only the latent to the DiT, so the stored latent *is* the native reference. A version-2 file reads as version 3 with its orbit as clip 1; *Build View Set* now always writes version 3.

**Parity, measured on a 5-photo test set with a 56-frame orbit.**
- Replay View Set's rows, tags and image latents were bit-identical to the native node fed the same files.
- The orbit latent was identical too, except when the view set was built by the server's first video VAE encode after startup. That build differed from every later encode, native or not, by a mean of 4e-4 (max 0.11), spread evenly over frames and channels. Later encodes all agreed bit for bit, whatever else was loaded.
- If exact parity matters, build view sets with clips after the server has run a full H3 graph (one that loaded the text encoder). On 2026-09-15 two builds in a row on a fresh server both drifted; every build made after the native reference node had run matched it bit for bit, and matched each other.

### Clips and audio (version 3)

**Parity**, measured on the front-5 views of the Multiface subject plus clip 1 = an H3 render (22 frames kept) with its 10.1 s soundtrack, clip 2 = the silent 22-frame orbit, and audio 1 = the first 4 s of that soundtrack, at a 22-frame render length: rows (3555 x 5120), tags, the five image latents, both video latents and both audio latents were **bit-identical** to *MiniMax H3 Reference to Video* fed the same files, and the clip rendered on one seed was **pixel-identical with identical audio samples**. The version-2 view set and a version-1 RefCap still replayed identically afterwards.

**Cost.** Audio is nearly free on the text side and costs the DiT 2 tokens per 40 Hz latent frame (80 tokens per second); a stored latent is 256 floats per frame, about 100 KB per 10 s as fp32.

**Caveats.** The audio VAE is lossy and one-way: a stored latent can't be re-encoded at another quality later. Nothing synchronises a reference soundtrack with generated motion. A voice identifies a person as much as a face does.

#### Several clips vs stills: no gain.

`b5clips3` (three 22-frame segments of the orbit, 135° each, 480 px so that the three together cost 4,725 tokens against the single orbit's 4,913) scored **0.694** over the three shots against 0.724 for the five stills and 0.718 for the one orbit; lower than the stills in 8 of 9 paired clips and below the orbit in every yaw bin including 70°+ (−0.020 against the stills there, where the orbit was +0.057). Splitting a turn into three video references buys nothing that one orbit doesn't, and the smaller canvas the token match forces costs detail. The one confound is that canvas (480 against 544 px). Use the clip slots for different material (a second scene, motion, a soundtrack), not for more angles.

#### Does a voice reference transfer? Yes.

Run on 2026-09-15. Two voices were rendered by H3 itself, with no references, saying the same line: **A**, a woman in her thirties with a warm, low, calm voice, and **B**, a man in his seventies with a deep, gravelly voice, plus a solo-cello **music** clip. They went into one version-3 view set (clip 1 = A's frames with its soundtrack, audio 1 = A, audio 2 = B, audio 3 = the music). The test prompt shows a hooded figure with its back to the camera, so the picture gives no hint of the voice, saying a different line, `Come in. The door is open.` Five arms, three seeds, 56 frames.

Speaker similarity (WeSpeaker ResNet34-LM, cosine to each reference; on this material same-source halves score 0.61 and cross-source halves 0.08):

| Arm | Reference | sim. to A | sim. to B | Whisper hears the line |
|---|---|---|---|---|
| `a0none` | none | 0.07 | 0.29 | 3/3 |
| `a1voice` | A as `<Audio 1>` | **0.58** | 0.18 | 3/3 |
| `a2clip_av` | A's clip as `<Video 1>` with its soundtrack `<Audio 1>` | **0.45** | 0.16 | 3/3 |
| `a3other` | B as `<Audio 1>` (the swap control) | 0.10 | **0.67** | 3/3 |
| `a4music` | the cello as `<Audio 1>` | 0.24 | 0.09 | 2/3 |

- **The voice follows the reference in 3 of 3 seeds each way**, to the level of the same-source calibration: A as the reference puts the generated voice at 0.58 against A (the halves of A itself score 0.61); B as the reference puts it at 0.67 against B. Without a reference the hooded figure gets a voice nearer B (0.29) than A (0.07), and both references overrode that prior. The spoken line was the prompt's, not the reference's, in every voice clip.
- **A standalone audio reference beats the clip-with-soundtrack** (0.58 against 0.45) at a fifth of the tokens (186 against 3,739): the video block adds nothing for the voice, and its kitchen scene did not leak into the picture.
- **Music:** on one seed the cello came through almost verbatim (0.95 to the music reference) and the line was dropped; on the other two the line was spoken and no music registered on this metric. A speaker model is not a music metric, so treat the music arm as a hint only.
- This contradicts the RefMod author's report that voice transfer fails. The difference may be the prompt: here the reference is bound in `subject_definitions` (`<Audio 1> is the voice of <Subject 1>.`) and in `retention_analysis`, and the subject's look gives the model no reason to pick another voice. A voice reference against a contrary visual prior was not tested.
- The two references are H3's own voices, not real speakers, and the clips are 2.3 s. Real voices, longer lines and a contrary visual prior are open.

### Nodes

**H3 RefCap Build View Set**

| Input | |
|---|---|
| `vae` | The MiniMax H3 video VAE |
| `name` | File stem in `models/refcaps/` |
| `ref_image_size`, `width`, `height` | Sizing, as on the native reference node (`max`: up to a 2048 px short edge, never upscaled) |
| `views` / `view_folder` | An image batch and/or a folder (absolute, or under `ComfyUI/input`) |
| `yaws` | One yaw in degrees per view, comma-separated. Empty: TURNtable spacing, `yaw_i = sign · (i · 360/N + frame_offset_degrees)`, with sign −1 when `reverse_direction` is set |
| `orbit` / `orbit_folder` | Clip 1 (the orbit): frames at 24 fps, stored as a video reference |
| `orbit_length` | The length you will render at; applies to every clip |
| `audio_vae` | The MiniMax H3 audio VAE, required as soon as any audio is connected |
| `orbit_audio`, `clip_2` / `clip_2_folder` / `clip_2_audio`, `clip_3` / … | Clip 1's soundtrack; clips 2 and 3 with theirs |
| `audio_1..3` | Standalone audio references (a voice, music, ambience) |

A soundtrack is **not** cut to `orbit_length`, because the native node doesn't cut one either (a 10 s soundtrack on a 22-frame clip costs 405 audio latent frames, 810 tokens); trim it first with *Trim Audio Duration* if it should be shorter. *Load Video* + *Get Video Components* give a clip's frames and soundtrack; *Load Audio* also reads the audio track of a video file.

Folders sort the way TURNtable Viewer sorts them (natural sort), so one image folder drives both the Blender billboard and the view set.

**H3 RefCap Replay View Set**

| Input | |
|---|---|
| `clip`, `prompt`, `viewset` | As *Replay* |
| `select` | `nearest`: each of `target_yaws` takes its nearest unused view, round after round, until `count` views are chosen (0 = one per angle). `all`, or `indices` (0-based) |
| `orbit` | Add clip 1 (the orbit) as a video reference |
| `more_clips` | Other clip numbers to add, e.g. `2,3` |
| `audios` | Standalone audio numbers to add, or `all` |

References are emitted in the native node's order: the pictures, then each clip (a soundtrack's `<Audio j>` right before its `<Video k>`), then standalone audio. The tokenizer numbers each kind in emission order, so a soundtrack takes an `<Audio j>` before any standalone audio; the label guide says which is which.

Outputs:
- the conditioning;
- a **label guide**, e.g. `<Picture 2> = view 7, yaw 90 (07_400357.png)`.

Replay View Set raises an error above 9 pictures, 3 clips or 3 audio references, the native node's limits.

**H3 RefCap Orbit Views** takes the frames of two orbit renders (`left`: front round the subject's left side to directly behind; `right`: the same on the right) and returns the frame nearest each of `yaws` (empty: every 15°, 24 views) as `views`, plus their angles as `yaws`, for *Build View Set*. `measure` picks how each frame's angle is found:
- `faces` labels a frame by insightface's yaw up to 75°, then interpolated to 180° at the last frame. It needs `insightface` and `models/insightface/models/buffalo_l`.
- `motion` is for camera pans round a place. It sums the sideways image shift between neighbouring frames (phase correlation, `refcap_views.pan_angles`) and scales it to 180° at the last frame.
- `steady` assumes a steady turn.

H3 eases into both orbits and pans, so `steady` is a poor guide: on the first room pan it put frame 25 at 82°, where the measured motion gave 26°, which matches what the frame shows. The frames are rounded to 8 bits so the build stores them as PNGs. On the `kbh_mara` orbits it picks the same 24 frames as `harvest`, at the same angles to within 0.1°.

### Test

**Subject: Multiface identity 002645310.** Wuu et al. 2022, CC BY-NC 4.0; renders stay private.
- 147 calibrated cameras around the head.
- One frame of the `EXP_neutral_peak` expression.
- Yaw and elevation come from the camera calibration in the tracked head frame (+y up, +z out through the nose).

**Views:**
- Cameras within 20° of head height, sorted by yaw, alternate between references and a held-out gallery.
- The references become head-centred crops (1024 and 736 px).
- The orbit is 56 frames at 544 px, starting at the front, each frame the reference nearest its angle.
- The gallery is 512 px.
- Every view of an identity gets the same colour correction (the raw dome images are dim and green).
- One camera recorded only black (400267) and was dropped. Two are mounted upside down (400295, 400370); their crops were turned upright from the projected head axis.
- That leaves 27 references and 27 gallery views. The references run from −144° to +148°, so nothing is closer than 32° to directly behind the head.
- The pose check agreed with the calibration: insightface |yaw| medians of 16°, 44° and 74° for calibration bins 0–30°, 30–60° and 60–90°. Beyond 90°, 3 of 7 gallery faces were detected.
- `b1front5` gets the views at −54°, −20°, +5°, +20° and +54° (none is nearer ±45°). On the `front` shot, `b4select5` selects exactly the same five, so those clips double as a determinism check.

**Strangers:** Multiface 6795937 and 002643814 (40 cameras each).

**Arms**, matched at about 5,000 reference tokens:

| Arm | References | Latent tokens |
|---|---|---|
| `b0none` | None (control, `b1front5`'s prompt) | 0 |
| `b1front5` | 5 views nearest 0°, ±20°, ±45° at 1024 px: the usual front and 3/4 references | 5,120 |
| `b2ring9` | 9 views nearest 40° steps around the head, 736 px | 4,761 |
| `b3orbit` | The orbit as `<Video 1>`, 544 px | 4,913 |
| `b4select5` | The 5 views nearest the shot's angles at 1024 px (the billboard's lookup) | 5,120 |

Picture labels name each view's angle ("front view", "side view", "back view").

**Shots**, all with clothes different from the capture:
- `turn`: a look over the shoulder, from profile to front.
- `orbit`: the camera circles from the face to behind the head.
- `front`: talking to the camera, as the check that frontal shots don't lose.

**Settings:** 3 seeds, as in the tests above.

**Scoring** (24 frames per clip):
- ArcFace, as before, plus insightface's head yaw per frame.
- Template: the gallery views turned less than 45°. It never includes a reference view.
- Faces are binned by |yaw|: 0–20°, 20–45°, 45–70°, 70°+. ArcFace falls at large yaw for everyone, so arms are compared within a bin.
- Each bin is calibrated with the gallery (same person) and the strangers.

**Decision rules**, fixed before any render. The plan's "7 of 9 clips" became a share, because front-shot clips may have no turned faces:
- **Coverage helps** if `b2ring9` or `b3orbit` meets both conditions:
  - a higher mean over frames with |yaw| ≥ 45° than `b1front5` in at least 80% of paired clips that have such frames in both arms, with at least 5 pairs;
  - a mean on the `front` shot no more than 0.03 below `b1front5`.
- **Per-shot selection** is recommended as the default use if `b4select5` ≥ `b1front5` in every shot's mean.
- **Otherwise,** angle coverage isn't the bottleneck at this budget. View sets stay (they are a superset of version 1), and 1024 px RefCaps stay the recommendation.

**Verification:**
- Replay View Set (9 views + orbit) against the native node fed the same files: conditioning, and rendered frames.
- A version-1 RefCap re-render is compared with its earlier clip.

### Results

Parity first, on the real subject: Replay View Set with 9 views plus the orbit produced **bit-identical** rows, tags and all 10 reference latents against the native node fed the same files, and a **pixel-identical** clip. A version-1 RefCap re-rendered to its earlier clip exactly. On the `front` shot `b4select5` selects the same five views as `b1front5` and rendered pixel-identically in 3 of 3 seeds.

**Mean face similarity** (ArcFace against a template of held-out gallery views; 3 seeds per shot):

| Arm | turn | orbit | front | all | paired vs `b1front5` |
|---|---|---|---|---|---|
| `b0none` | 0.055 | 0.020 | −0.008 | 0.022 | −0.701, higher in 0/9 |
| `b1front5` | 0.701 | 0.684 | 0.786 | **0.724** | — |
| `b2ring9` | 0.713 | 0.663 | 0.780 | 0.719 | −0.004, higher in 2/9 |
| `b3orbit` | 0.725 | 0.665 | 0.764 | 0.718 | −0.006, higher in 1/9 |
| `b5clips3` (added 2026-09-15: three 22-frame orbit segments as `<Video 1..3>`, 480 px, 4,725 tokens) | 0.649 | 0.673 | 0.759 | 0.694 | −0.030, higher in 1/9 |
| `b4select5` | 0.686 | 0.665 | 0.786 | 0.712 | −0.011, higher in 2/9 |

The four reference arms land within 0.012 of each other. Spending the same tokens on coverage instead of detail changed the face score very little.

**By head angle**, paired against `b1front5` (mean difference, clips higher / clips with faces in that bin in both):

| Arm | 0–20° | 20–45° | 45–70° | 70°+ |
|---|---|---|---|---|
| `b2ring9` | −0.014 (1/7) | −0.043 (1/6) | −0.041 (1/6) | **+0.040 (4/6)** |
| `b3orbit` | −0.029 (0/7) | −0.041 (0/6) | −0.049 (1/6) | **+0.057 (3/6)** |
| `b4select5` | −0.007 (2/7) | −0.007 (1/7) | −0.012 (2/6) | −0.018 (1/6) |

Coverage wins only where the head is turned furthest, and loses slightly everywhere else. For scale, the same-person calibration falls the same way (0.916, 0.903, 0.805, 0.573 across the bins): ArcFace itself weakens with yaw, which is why arms are compared within a bin. Strangers stay flat across those bins at 0.111, 0.105, 0.107 and 0.120 (62 views of two other identities, highest 0.244), so every reference arm is far above stranger level in all 9 clips, even at 70°+, while the control never rises above a stranger in any of its 9.

Per clip, over frames turned 45° or more: `b2ring9` +0.018 (higher in 2 of 6), `b3orbit` +0.039 (3 of 6), `b4select5` −0.021 (1 of 6).

**Verdict by the rules above: coverage does not help.** Neither `b2ring9` (2 of 6) nor `b3orbit` (3 of 6) reached 80% of at least 5 off-angle pairs, though both stayed inside the front-shot allowance (−0.005 and −0.022). Per-shot selection is not recommended either: `b4select5` came in at −0.015, −0.019 and +0.000 across the three shots.

**What coverage did change is the back of the head,** which ArcFace cannot score because no face is visible. The subject wears a distinctive yellow hair clip, seen only in her side and back views. In the orbit shot's closing frames, by eye:

| Arm | Yellow clip |
|---|---|
| `b0none`, `b1front5` (front and 3/4 only) | absent, a plain dark tie invented instead, 0 of 3 seeds each |
| `b2ring9`, `b3orbit`, `b4select5` | present, with the thin second clip, 3 of 3 seeds each |

So all-around references reproduced an accessory that front-only references missed entirely, 9 of 9 against 0 of 3. (A yellow-pixel metric over whole frames was tried and discarded: the sunlit park scored higher than the clip, and the no-reference control scored 55 per 1000 px with no clip at all.)

**Cost per clip:** `b3orbit` ~51 s against ~84 s for `b1front5`/`b4select5` and ~79 s for `b2ring9`, at matched latent tokens. A video reference is the cheapest way to carry many angles, because the text encoder only sees it at 2 fps: 5 frames instead of 27 views' worth of rows.

**Reading it.** At a fixed token budget and for facial identity, five sharp front and 3/4 views are as good as spreading the same budget around the head. Coverage pays for itself at |yaw| ≥ 70° and for what the face metric cannot see — hair shape and accessories from behind. A billboard is worth building when shots turn away from the camera, not as a general replacement for good front references.

**Caveats:** one subject, 45 clips, 608×352 at 56 frames; dome captures rather than ordinary photos; no reference within 32° of directly behind the head; and ArcFace is unreliable beyond 70°, where detection also thins out. The stranger galleries were rebuilt after the test with padded crops and everything re-scored: the first build framed the 40-camera rigs' heads so tightly that insightface found 1 face in 31 views. Padding lifted the stranger baseline from 4 usable views to 62, and every arm's score came out identical, since scoring is deterministic and the template comes from the subject's own gallery, which detected 23 of 27 and was left alone.

**Running it on 12 GB.** After a long server session (builds, parity checks and dozens of renders) the server held about 10 GB of VRAM *outside* PyTorch — dynamic VRAM staging, which `/free` does not return — and `VAEDecode` then ran out of memory on every retry of a prompt, three in a row, which stopped the chain. Earlier RefCap chains hit one retry in 162 clips; this one hit several.

The test runs freed or restarted the server before each shot and retried a failed prompt; neither changes what a render produces (renders read stored latents and never VAE-encode). A freshly started server's first video encode differs slightly from later ones, so the built view sets were kept between runs.

**With other apps on the GPU.** During the Klein run another app's GPU process (~195 MB) was open, and even a *fresh* server failed `VAEDecode` on four attempts in a row: it held ~10 GB outside PyTorch with ~100 MB free.
- **The fix:** relaunch ComfyUI with `--vram-headroom 0.75`. It makes dynamic VRAM keep that much memory free, counting other apps' usage.
- **The result:** the remaining 12 clips rendered with no retries.
- **Why it's safe:** the flag changes memory staging, not the render path.

### Generated turnarounds (Krea 2)

The views above were a real 147-camera capture. The follow-up question: could an image generator stand in for real 360° photos — for a real person, or for an original character that has no photos at all?

**Setup.** A community Krea 2 turbo finetune (called Krea 2 below), a FLUX.2-family model with a Qwen3-VL text encoder (`CLIPLoader` type `krea2`) and the Qwen-Image VAE, at 8 steps, cfg 1, `er_sde`/`beta57`, 1024². Reference images go through the text encoder via `Krea2EditRebalance` (ComfyUI-ConditioningKrea2Rebalance), up to four at a time. No Flux IPAdapter or PuLID is installed, and Redux (Flux.1) needs a sigclip encoder that isn't.

Krea 2 generated an original character three ways. She was given details visible from behind — a braid tied with a red ribbon under a copper comb — as the Multiface subject's hair clip was.
- **front:** every angle conditioned on the front portrait (12 views).
- **chain:** every angle conditioned on the portrait and its previous neighbour (24 views, 15° steps).
- **prompt:** no reference image at all, the "just prompt it well" baseline (24 views).

Each set was scored before any H3 time: ArcFace similarity to the set's own frontal views per requested |yaw| bin, and insightface's measured yaw against the requested one.

| Set | 0–20° | 20–45° | 45–70° | 70°+ | 70°+ bin: requested → measured |yaw| |
|---|---|---|---|---|---|
| front | 0.724 | **0.782** | **0.680** | 0.504 | 110° → 37° |
| chain | 0.755 | 0.748 | 0.600 | 0.470 | 124° → 49° |
| prompt | 0.745 | 0.584 | 0.568 | 0.480 | 124° → 50° |
| *real capture* | *0.916* | *0.903* | *0.805* | *0.573* | — |

- **Krea 2 does not turn a head away from the camera.** All 72 generated views in this section had a detectable face, including every view asked to show the back of the head. Asked for "seen directly from behind", it drew an over-the-shoulder look with the face swung back to camera. Measured rotation topped out near 50° in every mode, with or without a reference image, so this is the model's portrait bias rather than the conditioning.
- **Identity is weaker than a real capture at every angle** (−0.07 to −0.32), and distinctive details wander between views: the eyebrow scar resurfaced across a cheek.
- **Chaining made it worse,** not better (0.600 against 0.680 at 45–70°). Errors accumulated around the circle without buying rotation.

**The ground-truth check.** The same front-mode generation, conditioned on the Multiface subject's real front photo and scored against her real held-out camera views:

| \|yaw\| bin | Generated from her photo | Her real views | Strangers |
|---|---|---|---|
| 0–20° | 0.047 | 0.916 | 0.111 |
| 20–45° | 0.078 | 0.903 | 0.105 |
| 45–70° | 0.067 | 0.805 | 0.107 |
| 70°+ | 0.022 | 0.573 | 0.120 |

That is below stranger level. The images kept her *attributes* — centre-parted dark hair, the yellow clips, a dark top — but show a different person. They used the `normal` image-token budget because `high` ran out of VRAM at 1024². A retry at 768² with `high` also failed on every attempt, across two server restarts, so a larger budget isn't available on this 12 GB card.

**Verdict.** For a real person, a generated turnaround is no substitute for real 360° photos: identity is lost outright. For an original character, Krea 2 gives a recognisable but drifting character with no coverage past about 50° — and past 70° is exactly where the real-capture test found coverage pays. The planned 45-clip H3 run on the Krea 2 set was therefore not run: every arm would have received near-frontal views, so there was no coverage to test.

#### Bootstrapped with H3

H3 draws what Krea 2 won't: it rendered genuine back-of-head views in the real-capture test. So the Krea 2 portrait went into H3 as its only reference, and H3 rendered two 180° studio orbits — round her left side and round her right — at 768², two seeds each (~75 s per clip). Frames taken from them, labelled by insightface's measured yaw, became the turnaround.

H3's orbits ease in: by frame position, a frame "at 30°" measured 5°. So each frame was labelled by insightface's measured yaw up to 75°, then interpolated to 180° at the final frame, which shows a clean back view. Seed 4242 gave the 24 references; seed 777 gave a held-out gallery at in-between angles.

The gate, against Krea 2's best set:

| Set | 0–20° | 20–45° | 45–70° | Views with no face (true back views) |
|---|---|---|---|---|
| H3 orbit, seed 4242 | 0.973 | 0.863 | 0.758 | 7 of 24 |
| H3 orbit, seed 777 (gallery) | 0.981 | 0.850 | 0.691 | 11 of 24 |
| Krea 2, front-anchored | 0.724 | 0.782 | 0.680 | 0 of 12 |
| *real capture* | *0.916* | *0.903* | *0.805* | — |

A real turnaround, then, with identity within ~0.05 of a real capture up to 70°. It got the full 45-clip mirror. Parity held again: rows, tags and all 10 latents bit-identical and frames pixel-identical. On the front shot `b4select5` rendered pixel-identically to `b1front5` in 3 of 3 seeds.

| Arm | turn | orbit | front | all | paired vs `b1front5` |
|---|---|---|---|---|---|
| `b0none` | 0.246 | 0.228 | 0.261 | 0.245 | −0.394, higher in 0/9 |
| `b1front5` | 0.606 | 0.591 | 0.720 | **0.639** | — |
| `b2ring9` | 0.588 | 0.601 | 0.711 | 0.633 | −0.006, higher in 3/9 |
| `b3orbit` | 0.573 | 0.536 | 0.705 | 0.605 | −0.034, higher in 0/9 |
| `b4select5` | 0.624 | 0.568 | 0.720 | 0.637 | −0.002, higher in 4/9 |

By head angle against `b1front5`:

| Arm | 0–20° | 20–45° | 45–70° | 70°+ |
|---|---|---|---|---|
| `b2ring9` | +0.000 (2/6) | +0.032 (2/6) | +0.030 (3/6) | **−0.031 (2/6)** |
| `b3orbit` | −0.016 (1/6) | −0.028 (2/6) | −0.016 (2/6) | **−0.033 (3/6)** |
| `b4select5` | −0.022 (2/6) | +0.024 (3/7) | +0.045 (3/6) | **−0.036 (2/6)** |

Per clip, over frames turned 45° or more: `b2ring9` −0.005 (2 of 6), `b3orbit` −0.039 (3 of 6), `b4select5` −0.005 (3 of 6).

**Verdict: coverage does not help, and per-shot selection is not recommended** — as with the real capture. But the one edge real coverage had is gone: at 70°+ the bootstrapped arms lose (−0.03) where real views won (+0.04 and +0.06).

**Back of the head,** from zoomed last frames of the orbit shot:

| Arm | Braid from behind | Copper comb at the crown |
|---|---|---|
| `b0none` | straight down her back, as the prompt says; red ribbon in 3 of 3 | 0 of 3 |
| `b1front5` | swept to one side, as in the references | 3 of 3 (two faint) |
| `b2ring9`, `b3orbit` | swept | 0 of 3 each |
| `b4select5` | swept | 2 of 3 |

The swept braid came from the *front* views: the portrait already shows it over her shoulder. The comb is large from the front and small from behind, so the arms that spent their budget on front views reproduced it and the arms that spread it around the head lost it. No reference arm drew the red ribbon, which the references hide; the references overrode the prompt on that.

**Reading it.** The bootstrap is a working and cheap way to get a real turnaround of an original character. A view set built from it still doesn't beat five front views, though, and it loses the extreme-angle edge that real coverage had. The likely reason is that H3's back views are *inferred from the same front portrait*, so they carry no information the front didn't. Real back views help because they show things a front view can't, like the Multiface subject's hair clip. Generated ones can only restate the front, and here they even diluted a front-visible detail.

The stranger baseline (real Multiface people) sits far from this generated face (max 0.073), so "above strangers" says little here. The no-reference control, whose prompt describes her closely, already reaches 0.245; the references add ~0.39 on top of that.

#### FLUX.2 Klein 9B, an edit model

Krea 2 renders from a description, while FLUX.2 Klein 9B is built for editing. Its files are `flux-2-klein-9b_int8_convrot`, the text encoder `qwen_3_8b_fp8mixed` (`CLIPLoader` type `flux2`) and `flux2-vae`. The tests wired it as follows:
- the anchor is scaled to 1 MP, VAE-encoded, and attached with `ReferenceLatent` to both the prompt and an empty negative;
- `CFGGuider` cfg 5, Euler, `Flux2Scheduler` 20 steps, with the output at the anchor's size.

It was prompted with edit instructions ("Rotate the camera around this woman so she is seen from directly behind…"), with no identity paragraph, from the same Krea 2 portrait.

**It turns heads.** Asked for 90°, it drew a true profile. Asked for 180°, it drew a back view with the braid, comb and ribbon and no face. Krea 2 never produced either.

**It keeps a real person.** The drift check again, now with Klein:

| \|yaw\| bin | Klein, from her photo | Krea 2, from her photo | Her real views |
|---|---|---|---|
| 0–20° | **0.839** | 0.047 | 0.916 |
| 20–45° | **0.757** | 0.078 | 0.903 |
| 45–70° | **0.688** | 0.067 | 0.805 |
| 70°+ | **0.456** | 0.022 | 0.573 |

That is clearly her (strangers score ~0.11), about 0.08–0.15 below real photos of the same angle. Two flaws: only the 180° request came back as a true back view, and her yellow clip turned into rows of small beads.

**Angles are imprecise, and chaining fails.**
- Front-anchored views over-rotate small requests: 20° came back as 44°.
- The chained set froze near 39° and turned the same way on both sides, and its faces degraded step by step. Its high identity scores came from near-duplicates, which the first version of the gate rewarded.
- The gate now also requires the rendered angles to follow the request: side agreement ≥ 0.9 and rank correlation ≥ 0.6.

| Set | 0–20° | 20–45° | 45–70° | 70°+ | No face | Tracking (side / rank) |
|---|---|---|---|---|---|---|
| Klein, front-anchored | 0.892 | 0.871 | 0.774 | 0.487 | 2 of 12 | 1.00 / 0.93, follows |
| Klein, chained | 0.929 | 0.922 | 0.838 | 0.649 | 11 of 24 | 0.50 / 0.45, **fails** |

So the front-anchored set went into H3, with each view relabelled by its measured angle (insightface's yaw).
- **Views:** 0°, about ±44–53°, −61/−66°, +71°, ±90°, and true back views at +135° and +180°.
- **Gallery:** the held-out set (seed 777) over-rotated too, 7° coming back as 42°. So four straight-on views from seeds 778–781 were added, giving the scoring template real frontal views.

The run was the 45-clip mirror, scored against held-out Klein galleries.
- **Parity held:** bit-identical conditioning, pixel-identical frames, and `b4select5` rendered identically to `b1front5` on the front shot in 3 of 3 seeds.
- **VRAM:** with another app on the GPU, the front shot needed `--vram-headroom` (see *Running it on 12 GB*).

| Arm | turn | orbit | front | all | paired vs `b1front5` |
|---|---|---|---|---|---|
| `b0none` | 0.226 | 0.168 | 0.237 | 0.210 | −0.409, higher in 0/9 |
| `b1front5` | 0.589 | 0.566 | 0.702 | **0.619** | — |
| `b2ring9` | 0.557 | 0.510 | 0.698 | 0.588 | −0.031, higher in 2/9 |
| `b3orbit` | 0.553 | 0.550 | 0.687 | 0.597 | −0.023, higher in 3/9 |
| `b4select5` | 0.599 | 0.521 | 0.702 | 0.607 | −0.012, higher in 3/9 |

By head angle against `b1front5`:

| Arm | 0–20° | 20–45° | 45–70° | 70°+ |
|---|---|---|---|---|
| `b2ring9` | −0.051 (1/6) | −0.036 (2/6) | −0.012 (4/6) | **+0.020 (4/6)** |
| `b3orbit` | −0.046 (2/6) | −0.016 (1/6) | +0.047 (3/6) | **+0.025 (4/6)** |
| `b4select5` | −0.019 (1/6) | −0.038 (1/6) | −0.033 (4/6) | **+0.042 (5/6)** |

Per clip, over frames turned 45° or more: `b2ring9` −0.004 (4 of 6), `b3orbit` +0.010 (4 of 6), `b4select5` +0.001 (2 of 6).

**Verdict: coverage does not help** (4 of 6 off-angle pairs for the ring and the orbit, short of 80%), **and per-shot selection is not recommended** (−0.045 on the orbit shot).
- **At 70°+ Klein's views bring back a small edge,** +0.02 to +0.04, in the same direction as real photos. The H3 bootstrap lost there.
- **Near the front they cost 0.02–0.05,** so five front views still win overall.

**Back of the head,** from zoomed orbit frames:
- The arms that included Klein's back views sometimes copied its 180° view's centred braid (`b4select5` 2 of 3, `b2ring9` 1 of 3).
- The front-only and orbit arms drew the braid swept over the shoulder, as in the portrait.
- The copper comb appeared in `b1front5` 2/3, `b4select5` 2/3, `b2ring9` 1/3, `b3orbit` 0/3 and the control 0/3.
- This character's back isn't well defined, though: the portrait has the braid over her shoulder, and Klein's back view has it down her back.

**Across the reference sources.** Compare within a row; absolute levels differ with each run's template.

| References | front5 | ring9 | orbit | select5 | 70°+ vs front5 (ring / orbit / select) |
|---|---|---|---|---|---|
| Real 147-camera capture | **0.724** | 0.719 | 0.718 | 0.712 | +0.040 / +0.057 / −0.018 |
| Krea 2 | not run: no coverage | | | | |
| Krea 2 portrait + H3 orbit | **0.639** | 0.633 | 0.605 | 0.637 | −0.031 / −0.033 / −0.036 |
| Krea 2 portrait + Klein 9B edits | **0.619** | 0.588 | 0.597 | 0.607 | +0.020 / +0.025 / +0.042 |

**Reading it.** Whatever the source, five front views are the best use of the reference budget for faces. Extra angles help only past 70°, and only as far as the views carry real information: real photos most, Klein's edits somewhat, H3's inferred orbit frames not at all. For a real person Klein is the first generator here that keeps the identity, so it can fill in a missing angle when no photo exists — but a real photo of that angle is still better.

### Face + body (workflow 8's body section, 2026-10-04)

One generated character (`SNK_Reg_T_1_00003.png`, a bare-shouldered head shot). Workflow 8 generated a full-body photo in a red polka-dot sundress with white sneakers, then a 6-view body RefCap. The test rendered two shots whose prompts don't mention clothing (a full-body walk at 352×608, a cafe close-up at 608×352), 2 seeds each, through *Replay Pack*. ArcFace is measured against the head shot:

| references | walk | close-up | outfit in the clip |
|---|---|---|---|
| face 5 | 0.417 | 0.431 | no dress: the head shot's bare shoulders, 4/4 |
| body 6 | 0.239 | 0.297 | **the dress and sneakers, 4/4** |
| face 5 + body 4 (0, ±90, 180) | 0.440 | 0.470 | **no dress, 0/4: the head shot's bare shoulders** |
| the same 9, outfit bound in the prompt | 0.420 | **0.492** | **the dress and sneakers, 4/4** |

- **On their own, body views carry the outfit,** and they keep it in a close-up that shows only the top of the dress. Identity is weak, because the face is small in every body view.
- **Adding face views kept identity and lost the outfit.** Identity rose slightly over face views alone (higher in 3 of 4 pairs). But every clip followed the bare-shouldered head shots rather than the dress. With nothing in the prompt binding clothes to particular pictures, the 5 face views outvoted the 4 body views. The community finetune the tests ran on may have pushed the same way.
- **Binding the outfit in the prompt fixes it:** `<Subject 1> is the woman in <Picture 1>, ... and <Picture 9>, wearing exactly what she wears in <Picture 6>, <Picture 7>, <Picture 8> and <Picture 9>.` That gave the dress in 4 of 4 clips, with identity as good as or better than face views alone. This arm ran from one baked file, `my_character_full_blady` (face 5 + body 6, 8.3 MB), via *Replay View Set* `indices` = `0,1,2,3,4,5,7,9,10`.
- **So:** combine face and body views for identity plus outfit, and **name the body pictures as the clothing source**. To get a different outfit, describe it in the prompt instead.

### Locations (workflow 9, 2026-10-05)

Workflow 9 built `location_livingroom` from a generated empty living room: two 180° pans and 6 views labelled by measured pan motion. The test shots put blady (her 5 face views) in it, 2 seeds per arm:
- **forward:** the camera faces the sofa wall, which the photo shows;
- **reverse:** the camera stands at the sofa and looks at the opposite side of the room.

| Location references | Forward: the RefCap's room | Reverse: the RefCap's back wall |
|---|---|---|
| none (the room described in text) | 0/2; a different generic room per seed | 0/2; the sofa put behind her again |
| the source photo | **2/2** | 0/2; the far side invented differently per seed |
| 3 views nearest the shot's heading | **2/2** | 0/2; doors and furniture invented |
| the same 3 views, background named in the prompt | | **2/2** |

- **Location references keep a room** for any shot facing what they show. Text alone doesn't.
- **A reverse shot needs the background named:** `behind her is the wall in <Picture 8>`. This is the same lesson as the outfit: references only stick where the prompt says which picture to copy.
- The generated back wall was nearly blank, a plain wall with the bookshelf's edge. A wall with something distinctive on it (describe it in *Generate the location*) gives a reverse shot more to hold on to.
- ArcFace isn't comparable across these arms, because framing changed (the reverse photo clips show her small). It is left out.

### Video references (2026-10-07)

What a reference *clip* adds beyond stills and standalone audio, in four tests: a movement, a camera move, an edit of a finished clip, and a walk through a place. The clip results above were measured with references at or below the model card's 2 s minimum. These use 124-frame renders (5.2 s) and references of the same length, since H3 cuts a reference clip to the render's length.

The model card limits reference clips to 2-15 s each and 15 s in total. MiniMax's prompt guide uses `<Video N>` for whole-video relationships: editing it, continuing it, or following its camera movement, cuts or timing. A person, action or place taken from it is still defined as a `<Subject N>`.

**Material.** There are twelve Pexels clips, picked by eye. Each one's page, photographer and start second were recorded.
- Four movements, each one person, full body, with a still camera: jumping jacks, a slow standing yoga sequence, hip hop, and a karate kata.
- Four camera moves: a pan right over a lake, a drone flying forward along a road, a tilt down a skyscraper, and an orbit around a steel frame. None shows a room, so content leaking into a room shot is easy to see.
- Three room walkthroughs with no people: a living room, an apartment and a library.

Each clip was cut to 124 frames at 24 fps, at most 608 px on the long side, as `input/{move,camera,walk}_<slug>.mp4`. People in Pexels clips are used for their movement only, never as a face reference.

**Cost.**
- A clip's reference tokens follow its canvas, which is never upscaled. At 608 px, a 124-frame clip costs 7,733 tokens (37 latent frames × 209) on every step: as much as a 608×352 render of the same length. At 720p it would be 32,560.
- The text encoder reads the clip at 2 fps: 11 frames, about 1,250 rows.
- A stored move is 29.8 MB. Most of that is its 2 fps frames. *Load Video* decodes to float RGB, so they aren't exact 8-bit values and `compact` can't keep them as a PNG.

**Parity.** We compared *Replay Pack* of lf_013 (five views) plus the jumping-jacks move against *MiniMax H3 Reference to Video* fed the five photos and the same frames, at 352×608 × 124.
- Rows (2,681 × 5,120), tags, the five image latents and the video latent were bit-identical.
- The rendered clip was pixel-identical.
- Workflow 11 ran in one queue in 124 s. Its frame-rate check read 24.0.

#### Test 1, motion: a clip carries a movement far better than words.

- **Setup.** lf_013's five front views (*Replay Pack*) at 352×608 × 124 on a beach at sunset, in a red t-shirt and black leggings. Three seeds and four moves.
- **Arms:**
  - **A**: the move in words only.
  - **B**: A plus the move's clip as `<Video 1>`, bound in `subject_definitions` (`...whose movement comes from <Video 1>`) and in `retention_analysis` (`attribute_transfer - ...that person's appearance and surroundings do not appear`).
  - **C**: B without the words.
  - **D**: the clip with no pictures; the woman is described in words.
- **Scored with:**
  - **Motion:** DWPose joint angles (elbows, shoulders, hips, knees) and a banded DTW against the move's own reference. Lower is closer. The four references score 1.6-3.6 against each other.
  - **Timing:** the lag of the best motion-energy match.
  - **Identity:** ArcFace to lf_013 and to the reference person.
  - **Leak:** CLIP-H with the people blanked, and the person crops.

| Move | A words | B clip + words | C clip | D clip, no pictures |
|---|---|---|---|---|
| jumping jacks | 0.67 / 0.189 | 0.24 / 0.173 | 0.23 / 0.173 | 0.26 |
| slow yoga | 0.58 / 0.278 | 0.16 / 0.070 | 0.46 / 0.167 | 0.16 |
| hip hop | 1.17 / 0.169 | 0.57 / 0.200 | 0.46 / 0.206 | 0.52 |
| karate kata | 1.88 / 0.208 | 0.64 / 0.153 | 0.60 / 0.134 | 0.64 |
| **mean** | **1.08 / 0.211** | **0.40 / 0.149** | **0.44 / 0.170** | **0.40** |
| reference tokens, s per clip | 980, 57 s | 8,713, 101 s | 8,713, 102 s | 7,733, 88 s |

Each cell is DTW to the move / ArcFace to lf_013, the mean over 3 seeds.

- **The clip wins on movement in 12 of 12 seed pairs for every clip arm,** about 2.7 times closer than words. Each clip-arm output was nearest its own move's reference among the four; with words only, karate went to another move in 2 of 3.
- **Karate shows it best.** Words alone gave a different routine each time. With the clip she ran the reference's kata pose for pose: the crossed-arm guard, the knee lift, the step into a stance, the punch.
- **Timing transfers:**
  - On the two moves that don't repeat, the clip arms' motion matched the reference with a lag of 7 frames or less; words alone landed anywhere within ±24.
  - Jumping jacks repeat every second, so their lag can't be read.
- **Words that contradict the clip win.** The pilot's yoga description didn't match the window cut from the clip: arms raised, then lowered to the chest and raised again, where the clip only lowers them slowly to the forehead. Arms B and D then followed the words, and only C followed the clip. With matching words, B was the closest arm on yoga (0.16).
- **Nothing leaked:**
  - The reference person stayed at stranger level (ArcFace at most 0.08).
  - Her outfit and setting never appeared.
  - CLIP-H similarity to the reference's surroundings was flat across the arms (0.50-0.52).
  - The clip does pull the framing and camera angle toward the reference's: karate came out three-quarter and wider.
- **Identity costs 0.04-0.06** (B lower than A in 8 of 12 pairs, C in 8 of 12). The loss sits in moves that hide or turn the face: slow yoga, with the hands over the head, and karate's three-quarter stance. Hip hop rose a little (+0.03). Faces were about 50 px against 55 px for words alone.
- **Use B:** clip + matching words. A move costs 7,733 reference tokens and about 44 s more per 5 s clip on a 12 GB GPU.

#### Test 3, edit: fix one detail and keep the take.

- **Material.** Three of our own finished H3 renders of lf_013: a candlelit desk (124 frames, rendered with a film LoRA), a cistern hatch with a lantern (158), and an archive (158).
- **Two edits:** the dark green sweater made bright red, and a red apple added in one spot.
- **Every arm reuses the clip's own graph** (its LoRA, sampler and steps), read from the file's metadata.
- **Arms:**
  - **B0:** the graph re-queued unchanged.
  - **B:** the clip's prompt with only the detail changed, on the clip's own seed. This is the best a re-render can do.
  - **A0:** an edit that changes nothing.
  - **A:** an edit.
- **How an edit is set up.**
  - The pack is B's (lf_013's views) plus the clip as `<Video 1>`: an `edit_<clip>` RefCap holding it with its soundtrack.
  - The prompt is B's, rewritten the way MiniMax's guide writes an edit:
    - `<Video 1> is the source video for the target video edit.`
    - a summary starting `[video editing + audio reuse] The target video is an edited version of <Video 1> in which ...`
    - `<Video 1> (source video): partially_preserved - ... only <the change> differs.`
  - The clip's own soundtrack is reused as `<Audio 1>`.
  - Edits ran on the clip's seed, 4242 and 777.
- **Scores:**
  - SSIM to the source, whole frame and outside the person.
  - WeSpeaker against the source's voice.
  - ArcFace.
  - An eye check of the sweater and the apple.

| Arm | SSIM to the source | outside the person | voice vs source | s per clip | reference tokens |
|---|---|---|---|---|---|
| B0, re-render unchanged | 0.974 | 0.979 | 1.00 | 71 | 1,166 |
| B, re-render: sweater / apple | 0.384 / 0.403 | 0.39 / 0.37 | 0.93 / 0.93 | 70 | 1,166 |
| A0, edit changing nothing | 0.874 | 0.877 | 0.83 | 126 | 10,595 |
| A, edit on the clip's own seed: sweater / apple | **0.858 / 0.863** | 0.87 / 0.86 | 0.83 / 0.83 | 127 | 10,595 |
| A, edit on other seeds | 0.573 / 0.574 | 0.56 / 0.55 | | | |

- **Edit mode keeps the take; a re-render does not.**
  - On the clip's own seed, the edits kept 0.86 SSIM on average (lowest 0.82), about as much as an edit asked to change nothing. A re-render of the changed prompt on that seed kept 0.38-0.40: the same idea, but a different shot.
  - On other seeds, edits kept 0.57, so use the clip's seed.
- **Every change landed:**
  - The sweater turned red in 9 of 9 edits.
  - The apple appeared in 9 of 9. In the archive it was often two or three apples on the shelves, not one.
- **The reused soundtrack is regenerated, not copied:** voice similarity to the original was 0.83, below a re-render's 0.93. A visual-only edit keeps the timing, so laying the original audio back on might do better (untested).
- **Identity** barely moved: ArcFace 0.61 for the edits and 0.62-0.66 for the re-renders.
- **Cost:** about 1.8 times a re-render (127 s against 70 s), at 9 times the reference tokens.
- B0 differed from the clip it re-rendered by a mean 1.8/255: the stack has moved slightly since the clips were made two days earlier.

#### Test 2, camera: a clip does not carry a camera move.

- **Shots.** Workflow 9's example room (`location_livingroom`, views 1-3) at 608×352 × 124:
  - empty, on 3 seeds;
  - with lf_013 reading on the sofa (`_char`), on 1 seed.
- **Four Pexels moves.** For each, the reference's motion as the metric measures it: a pan left over a lake (+0.32 frame widths), a drone flying forward along a road (×2.7 zoom), a tilt down a skyscraper (−0.43 frame heights) and an orbit around a steel frame (almost no whole-image motion: the parallax cancels).
- **Arms:**
  - **A**: the move in words.
  - **B**: the words plus the clip as `<Video 1>`, with `[reference generation]`, `<Video 1> is the camera-movement reference for [Shot 1].`, `<Video 1> (camera movement): weak_reference - only the camera's path and speed are followed; none of its content appears.` and `The camera moves exactly as the camera in <Video 1> does`.
  - **C**: B without the words.
- **Scored with:**
  - The image's motion: shifts by phase correlation, and rotation and zoom from tracked features, with people masked.
  - The cosine of the whole move against the reference's, and the amplitude ratio along its main component.
  - CLIP-H similarity to the reference clip.

| Move | A words | B words + clip | C clip only |
|---|---|---|---|
| pan left | 0.89, ×1.61 | 0.86, ×1.50 | −0.29: a slow pan the other way |
| push forward | 0.99, ×0.90 | 0.99, ×0.87 | −0.21: a pan, no push |
| tilt down | 0.96, ×1.61 | 0.94, ×1.57 | −0.16: a pan, no tilt |
| orbit (judged by eye) | arcs around the table | arcs around the table | a pan |
| with lf_013 on the sofa (mean of pan, push and tilt) | 0.92 | 0.94 | −0.45 |
| reference tokens, s per clip (empty room) | 1,008, 54 s | 8,741, 100 s | 8,741, 100 s |

Each cell gives the direction match (cosine) and the amplitude against the reference: room shots over 3 seeds, character shots on 1.

- **The words carry the move; the clip adds nothing.** With words, H3 shot each named move in the right direction (cosine 0.86-0.99). The clip changed neither the direction nor the speed: pans and tilts overshot the reference about 1.5 times with or without it.
- **With the clip alone, every move became the same slow pan to the right.** That's whatever H3 does with "the camera moves as in `<Video 1>`" when nothing else is said; the reference's path didn't come through.
- **Nothing leaked:** CLIP-H similarity to the reference clips stayed at 0.20-0.38 in every arm (to the room's views: 0.66-0.86). Lakes, roads and towers never appeared.
- **So, for now: say the camera move in words.** A clip costs 7,733 tokens and about 46 s per shot for nothing. Workflow 13 is there for trying other clips and wordings. Untested:
  - references whose content matches the target, such as a move through another room;
  - the clip as an explicit storyboard;
  - lower `shift`.

#### Test 4, location: a walkthrough keeps the place, and four stills from it keep it better.

- **Material.** Three Pexels walkthroughs with no people, each 5.2 s: a living room with a dining table, an apartment that pans from a wardrobe to a dining corner, and a school library.
- **A walk RefCap** holds four evenly spaced frames as views (yaw 0: they carry no heading), the clip, and the clip at 320 px.
- **Shots.** lf_013 (*Replay Pack*) walks into the room and looks around, 608×352 × 124, 2 seeds. There are two shots:
  - **start:** the camera faces what the walk's first frames show;
  - **end:** it faces what its last frames show.

  These walks move forward and turn up to about 90°, so neither shot is a true reverse angle.
- **Arms.** Every arm describes the room in words as well, and names the part the camera faces with its picture or with `<Video 1>`.
  - **A**: the 4 views.
  - **B**: the clip as `<Video 1>` (`<Subject 2> is the room in <Video 1>, ...`).
  - **Bs**: the 320 px clip.
  - **C**: the first and last views plus the clip.
  - **D**: the room in words only.
- **Scoring.** CLIP-H similarity, with people blanked, to the walk's frames on the side the shot faces. The same side of the other two rooms is the baseline.

| Arm | similarity to the room (start / end) | other rooms | reference tokens | s per clip |
|---|---|---|---|---|
| A, 4 views | **0.807 / 0.816** | 0.47 | 1,816 | 65 |
| B, the clip | 0.790 / 0.785 | 0.47 | 8,713 | 101 |
| Bs, the clip at 320 px | 0.797 / 0.780 | 0.47 | 3,200 | 68 |
| C, 2 views + the clip | 0.797 / 0.809 | 0.47 | 9,131 | 107 |
| D, words only | 0.662 / 0.655 | 0.48 | 980 | 55 |

- **Any reference keeps the room:** the clip beat words in 12 of 12 pairs (+0.13). By eye, the apartment's wardrobe, air conditioner, globe lamp, white dining set and framed pictures all came through, where words gave a different room every time.
- **Four stills from the walk did better than the walk itself:**
  - +0.02 over the full clip, higher in 10 of 12 pairs, at a fifth of the tokens and 36 s less per clip.
  - The 320 px clip did as well as the full one (+0.00) at 37% of its tokens.
  - Adding the clip to two views didn't beat the four views (6 of 12).
  - The clip arms also faced the requested part of the room less reliably.
- **So:** make a location from a walkthrough by taking stills from it. Workflow 15 does that. Storing the clip buys nothing a few frames don't. A true reverse angle, the wall behind a walk, is still open: these walks never showed one.

## Reproducing the tests

The tests ran with an unpublished harness of render and scoring scripts against a running ComfyUI. Its inputs (the London Set, Multiface, Pexels clips and our H3 renders) can't be redistributed, so the numbers here are the record.

- The London Set comes from Figshare (doi:10.6084/m9.figshare.5047666); see [Credits and licenses](#credits-and-licenses) for the others.
- Arm IDs keep the names they had before the RefCap rename: `2encode` is Replay and `2pack` is Apply.
- Face scores used insightface's `buffalo_l` models; the RefMod arms used ComfyUI-MiniMaxH3Mod.
- Renders went one prompt at a time with `/free` between them: on a 12 GB card, different H3 graphs alternating on the queue have faulted.

## Limitations

- **Small sample.** The main comparison has two people, two scenes, three seeds and short clips (56 frames at 608×352, 12 clips per arm). The view-set tests add one Multiface subject and generated characters, and the video tests render 124 frames. Higher resolutions are untested.
- **RefMod was tested at its defaults.** Photos were stacked into one mod; per-photo image mods, larger Compressed grids, `merge`, copies and strength curves were not tried.
- **High-resolution RefCaps are slow and big.** The text encoder sees the full-resolution photos. A hybrid, with latents from large photos and rows from small ones, might keep the identity at RefMod's cost. It would no longer be exactly native, and it is untested.
- **Per-render caps.** Replay View Set and Replay Pack stop at the native node's 9 images, 3 clips and 3 audio references. *Apply (cached)* takes one RefCap and refuses conditioning that already holds references; combine RefCaps with *Select* and *Replay Pack* instead.
- **File size.** Solved for the common case: `compact` PNG sources with `store_rows` off give 6.5–6.9 MB for five photos, about 25× smaller than the first format, and replay stays bit-identical. Going further (bf16 rows, latents or pixels) would break that parity; untested.
- **Metric.** ArcFace similarity measures facial identity only, on small faces. It says nothing about hair, body or voice. The face sheets were checked by eye.
- **Consent.** Use photos of people who agreed. The London Set's consent covers research use; renders of those subjects are not distributed. Pexels clips were used for movement, camera moves and places only, never as a face reference.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Registers the nodes and the picker |
| `refcap_nodes.py` | Capture, Replay, Apply, Save Conditioning, the view-set nodes and Orbit Views |
| `refcap_select.py` | Select, Replay Pack and Bake Pack, and the picker's `/refcap/library` and `/refcap/thumb` routes |
| `refcap_views.py` | View selection, TURNtable ordering and orbit frame picking (no ComfyUI imports) |
| `web/refcap_picker.js` | The **Pick views...** thumbnail browser on *H3 RefCap Select* |
| `example_workflows/` | The 16 canvas workflows "H3 RefCap 1-16" |
| `REFCAP_README.md`, `REFCAP_VS_REFMOD.md`, `REFCAP_AV_EXPLAINER.md` | These docs |
| `README.md`, `LICENSE` | The overview and the GPL-3.0 licence |

## Credits and licenses

- **Face Research Lab London Set:** DeBruine, L. & Jones, B. (2017). doi:10.6084/m9.figshare.5047666. CC BY 4.0; the subjects' consent covers research use.
- **insightface buffalo_l models:** non-commercial research use only (deepinsight/insightface). Used by the face scores and by *H3 RefCap Orbit Views*' `faces` measure.
- **Pexels clips:** the [Pexels License](https://www.pexels.com/license/): free to use and modify, attribution optional; no redistribution of unaltered files, no implied endorsement, and identifiable people may not be shown in a bad light. Each clip's page and photographer were recorded. The clips are not in this repository.
- **WeSpeaker** `voxceleb_resnet34_LM`: from [huggingface.co/Wespeaker](https://huggingface.co/Wespeaker), toolkit [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker); trained on VoxCeleb. Used only for the voice test's scores.
- **ComfyUI-MiniMaxH3Mod:** MIT, © 2026 Luisa (luisacaotica). Installed only for the comparison; none of its code is used here.
- **MiniMax H3 reference prompt format:** MiniMax, `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`.
- **Multiface:** Wuu, C. et al. (2022). *Multiface: A Dataset for Neural Face Rendering*. arXiv:2207.11243, github.com/facebookresearch/multiface. CC BY-NC 4.0: non-commercial use only.
- **TURNtable Viewer** (Blender add-on, "Multi-Directional-Billboard.py", shared on r/comfyui): inspired view sets; none of its code is used, only its sort order and frame-angle convention are matched.
