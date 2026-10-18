# Clips and audio in RefCaps: what was built, how it works, what it costs

An explainer for the version-3 RefCap work of 2026-09-15. The reference for exact details is `REFCAP_README.md`
("View sets" and "Clips and audio"); this file explains the ideas and the numbers.

## 1. What was done, in order

1. **The file format grew to version 3.** A RefCap could already hold photos (version 1) and many views of one person
   plus one orbit video (version 2). Version 3 holds up to **three clips**, each with an optional **soundtrack**, and up
   to **three standalone audio references**, alongside the views. Old files still load: a version-2 orbit reads as
   clip 1.
2. **The two view-set nodes grew with it.** *H3 RefCap Build View Set* got an `audio_vae` input, a soundtrack input per
   clip (`orbit_audio`, `clip_2_audio`, `clip_3_audio`), two more clip slots, and `audio_1..3`. *H3 RefCap Replay View
   Set* got `more_clips` (which clips to attach beyond the first) and `audios` (which standalone audios).
3. **Parity was proven before anything was measured.** A new parity check builds a set with views, two clips, one
   soundtrack and one standalone audio, replays it, and compares it against the native *MiniMax H3 Reference to Video*
   node fed the same files. Rows, tags and every latent (image, video and audio) were bit-identical, and a render from
   each was pixel-identical with identical audio samples. Version 1 and 2 files were re-checked and still match.
4. **A metric was added for voices**, because judging voices by ear is the trap the face work avoided. It
   uses WeSpeaker ResNet34-LM (a 26 MB speaker-recognition model, run on the CPU) to give every clip a 256-number
   "voiceprint" and compares voiceprints by cosine similarity, exactly the way ArcFace scores faces in the earlier tests.
5. **Two experiments were run.** Does a voice or music reference transfer? Do several clips beat stills for identity?
6. **Workflow 6 and the docs** were written so you can use it from the canvas.

## 2. Why it was done this way

**The rule behind every RefCap step is: copy the native node exactly, then prove it.** The native reference node
resizes, VAE-encodes and orders references in a specific way, and the H3 tokenizer numbers `<Picture i>`, `<Video k>`
and `<Audio j>` in the order they are emitted. A RefCap stores the *result* of those steps and feeds it back in the same
order, so the DiT cannot tell the difference. That is why the parity check comes before any measurement: if the
conditioning is bit-identical, every result about "references" is also a result about the native node, and nothing
about the storage can be blamed or credited.

**Audio is stored differently from images, and for a good reason.** An image reference reaches H3 by two roads: the
text encoder looks at the pixels (so the prompt can attend to them), and the DiT gets the VAE latent. A RefCap must
therefore store both pixels and latent. Audio has one road: the text encoder never hears it. It only receives the
label text `<Audio 1>: `, and the DiT receives the audio VAE's latent. So a stored audio latent *is* the native
reference, with nothing left out.

**Why the "warm-up" rule changed.** The first video-VAE encodes after a fresh server start differ from every later one
by a mean of 4e-4. Previously one warm-up build was thought to be enough; on 2026-09-15 two builds in a row both drifted, while
every build made after a full H3 graph had run (with the text encoder loaded) matched native bit for bit. The parity
action now builds, runs the native encode, builds again, and keeps that second build. Practical rule: build sets with
clips after a render, not on a cold server.

**Why the experiments were designed as they were.**
- The voices are H3's own renders (a woman with a warm low voice, an elderly man with a deep gravelly voice), because no
  consented or public-domain speech was on disk. Synthetic references also remove any consent question.
- The test subject is a hooded figure with its back to the camera, so the picture gives H3 no hint about the voice. The
  RefMod author's failed voice test used a masculine voice for a female character; a visual prior fighting the reference
  is a different, harder question, left open.
- The two references say one line and the test prompt asks for a different line, so a high similarity means the
  *timbre* transferred, not that the reference audio was copied.
- A swap control (voice B as the reference) and a no-reference control bracket the result.
- The clip test matched reference tokens to the earlier orbit arm, because token count is what the DiT actually pays
  for, and the earlier view-set test was run at matched tokens too.

## 3. How a reference reaches H3

H3 generates video and audio jointly. Its DiT works on a single packed sequence of tokens: the target video's tokens,
the target audio's tokens, the text encoder's rows, and every reference block. Reference tokens ride through every
sampling step; they are re-injected each step and never denoised.

| Reference | What the text encoder sees | What the DiT gets | DiT tokens |
|---|---|---|---|
| Picture | `<Picture i>: ` + the pixels as a vision block | video latent `[1, 24, 1, H/16, W/16]` | (H/32) × (W/32) |
| Clip | `<Video k>: ` + the frames at 2 fps as vision blocks, each with a timestamp | video latent `[1, 24, T, H/16, W/16]` | T × (H/32) × (W/32) |
| Soundtrack of a clip | `<Audio j>: ` only, emitted just before its `<Video k>` | audio latent `[1, 32, 2, A]` | 2 × A |
| Standalone audio | `<Audio j>: ` only | audio latent `[1, 32, 2, A]` | 2 × A |

- The video VAE compresses 16× in space and turns every 17 frames into 5 latent frames (plus 2): 22 frames → 7 latent
  frames, 56 → 17. The DiT then patches latents 2×2, which is where the (H/32)(W/32) comes from.
- The audio VAE runs at **40 latent frames per second**; each latent frame has 32 channels for 2 (stereo) positions and
  costs the DiT **2 tokens**. So audio costs 80 tokens per second, whatever its content.
- The native node cuts a clip to the render's length (on the 17k+5 frame grid) but does **not** cut a soundtrack; the
  RefCap mirrors that. Trim long audio first with *Trim Audio Duration*.

Emission order, which fixes the numbering: all pictures; then each clip, with its soundtrack's `<Audio j>` right before
its `<Video k>`; then standalone audio. A clip's soundtrack therefore takes an `<Audio j>` before any standalone audio.
The label guide from Replay View Set prints the resulting names.

## 4. How many latents, and how big

The audio test set `av_audio.safetensors` (13.6 MB) holds, for a 56-frame clip at 608×352 with a 2.3 s soundtrack and
three 2.3 s standalone audios:

| Stored tensor | Shape | Role | DiT tokens when attached |
|---|---|---|---|
| `clip1_frames` | `[5, 352, 608, 3]` | the 5 frames the text encoder sees (every 12th of 56) | 0 (text side) |
| `clip1_latent` | `[1, 24, 17, 22, 38]` | the clip's video latent, 17 latent frames | 17 × 11 × 19 = 3,553 |
| `clip1_audio_latent` | `[1, 32, 2, 93]` | its soundtrack, 93 latent frames = 2.3 s | 186 |
| `audio1_latent` … `audio3_latent` | `[1, 32, 2, 93]` each | voice A, voice B, the music | 186 each |

So **one video latent per clip, one audio latent per soundtrack, one per standalone audio**; a full set can attach up
to 3 + 3 + 3 = nine latents on top of up to 9 picture latents. Storage: an audio latent is 32 × 2 × 4 bytes per latent
frame, about **10 KB per second** as fp32 (a 10 s voice is ~100 KB). The video latent of that 56-frame clip is 1.4 MB;
the 2 fps frames the text encoder needs are the bulk of the file (12.8 MB), because they are stored as fp32 pixels.

The three-clip set for the identity test (`bb_002645310_clips.safetensors`, 17.6 MB) holds three 22-frame clips at
480×480: each `clip_latent` is `[1, 24, 7, 30, 30]`, 7 × 15 × 15 = 1,575 tokens, 4,725 for the three; each clip's text
side is only 2 frames.

## 5. Does it save memory or time?

**What a RefCap skips at generation time:** loading the image/video decoders and running the VAE encodes. The video
VAE stages 2.7 GB and the audio VAE 0.6 GB; with a RefCap neither is loaded for encoding, and no resize or encode runs.
The text encoder still runs over the stored pixels and clip frames, because that is what makes the conditioning exactly
native; with only audio references, the text side is just a few label tokens.

**What it does not change:** the DiT's cost. The reference tokens are the same as native, and they ride through every
step, so the per-step memory and time of sampling are identical to the native node with the same references. A RefCap
is a way to *store and replay* references exactly and cheaply, not a way to compress them.

**Where audio wins on cost:** an audio reference is nearly free on the text side and light on the DiT. Measured on the
audio test (56 frames, 608×352, 8 steps):

| Arm | Reference tokens in the DiT | Mean render time (3 seeds) |
|---|---|---|
| no reference | 0 | 45 s* |
| voice A as `<Audio 1>` | 186 | 36 s |
| voice A's clip with its soundtrack (`<Video 1>` + `<Audio 1>`) | 3,739 | 65 s |
| voice B as `<Audio 1>` | 186 | 35 s |
| music as `<Audio 1>` | 186 | 34 s |

\*The no-reference arm's mean includes one seed that hit a VAE-decode out-of-memory and was retried after a server
restart; its clean renders took about 35 s, the same as the audio arms.

Put differently: a 2.3 s voice costs 186 tokens; five 1024 px stills cost about 5,000. Reading a voice from a RefCap
adds nothing measurable to a render, while attaching the video the voice came from nearly doubles it.

## 6. What the measurements showed

**Voice transfer works.** Speaker similarity (cosine of voiceprints; on this material two halves of the same clip score
0.61, halves of different clips 0.08):

| Reference | to voice A | to voice B |
|---|---|---|
| none | 0.07 | 0.29 |
| voice A as `<Audio 1>` | **0.58** | 0.18 |
| voice A's clip + soundtrack | 0.45 | 0.16 |
| voice B as `<Audio 1>` | 0.10 | **0.67** |

The generated voice followed the reference in 3 of 3 seeds each way, to the level of the same-source calibration, and the
spoken words were the prompt's line every time (Whisper heard "Come in, the door is open" in all 12 voice clips). The
standalone audio did better than the clip with its soundtrack, at a twentieth of the tokens, and the clip's kitchen
scene did not leak into the picture. This contradicts the RefMod author's report that voice transfer fails; the likely
differences are the prompt binding (`<Audio 1> is the voice of <Subject 1>.` in `subject_definitions`, plus a
`retention_analysis` line) and a subject whose look gives no reason to pick another voice.

**Music is a hint, not a result.** On one seed the cello came through almost verbatim (0.95 similarity to the music
reference) and the dialogue was dropped; on the other two the line was spoken and no music registered. A speaker model
is not a music metric.

**Three clips do not beat stills or one orbit.** At matched tokens, three orbit segments scored 0.694 face similarity
against 0.724 for five stills and 0.718 for one orbit, lower in 8 of 9 paired clips and in every head-yaw bin. The
smaller canvas the token match forces (480 against 544 px) costs detail, and more video references do not add angles
that one orbit lacks. Use the clip slots for different material: a second scene, a motion, a soundtrack.

## 7. Benefits, in short

- **One file per character can now carry looks and a voice**: views with angles, a clip, and audio, replayed exactly.
- **Exact and auditable.** Bit-identical conditioning means results about RefCaps are results about H3 itself.
- **Voices are cheap to keep and to use**: ~10 KB per second stored, 80 DiT tokens per second attached, no encoder
  load at render time.
- **A measured recipe for voice transfer** that worked 3 of 3 each way, with the prompt pattern that goes with it.
- **A metric** (WeSpeaker speaker similarity) so the next voice question can be answered with numbers instead of ears.

## 8. Caveats and open questions

- The audio VAE is lossy and one-way: a stored latent cannot be re-encoded at another quality later. Keep the source.
- Nothing synchronises a reference soundtrack with generated motion; a soundtrack is a timbre and style cue.
- The voices tested were H3's own, 2.3 s long, on a subject with no visual voice prior. Real speakers, longer lines and
  a reference that contradicts the picture are untested.
- A voice identifies a person as much as a face does. Use voices with consent.

## 9. Where things are

| | |
|---|---|
| Nodes | `refcap_nodes.py` (`H3ViewSetBuild`, `H3ViewSetReplay`, `H3SaveConditioning`) |
| Canvas | "H3 RefCap 6 - Clips and audio" (and 5 for view sets) in [`example_workflows/`](example_workflows/) |
| Tests | The parity check, the audio test and the clip test (arm `b5clips3`) ran with an unpublished harness of render and scoring scripts; its inputs can't be redistributed, so the numbers above are the record |
| Speaker model | `voxceleb_resnet34_LM.onnx` ([huggingface.co/Wespeaker](https://huggingface.co/Wespeaker)), used by the voice metric |
