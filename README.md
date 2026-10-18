<div align="center">

# ComfyUI-RefCap

**Reference capture for MiniMax H3: one file per character, object or place that replays H3's reference conditioning exactly.**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-green?style=for-the-badge)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange?style=for-the-badge)](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases)
[![License](https://img.shields.io/badge/License-GPLv3-red?style=for-the-badge)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen?style=for-the-badge&color=blue)](#-installation)
[![Last Commit](https://img.shields.io/github/last-commit/AEmotionStudio/ComfyUI-RefCap?style=for-the-badge&label=Last%20Update&color=orange)](https://github.com/AEmotionStudio/ComfyUI-RefCap/commits)
[![Activity](https://img.shields.io/github/commit-activity/m/AEmotionStudio/ComfyUI-RefCap?style=for-the-badge&label=Activity&color=yellow)](https://github.com/AEmotionStudio/ComfyUI-RefCap/commits)

![RefCap: one photo, a RefCap turnaround, any shot](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_hero.png)

*Capture a character's reference photos once, then replay them in any shot: no photo loaders, no resizing, no VAE encode, and conditioning bit-identical to the native reference node.*

[Features](#-features) • [Examples](#-examples) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Visual Guide](#-visual-guide) • [Nodes](#-nodes) • [Example Workflows](#-example-workflows) • [Models](#-models) • [Documentation](#-documentation) • [Troubleshooting](#-troubleshooting) • [Contributing](#-contributing)

</div>

---

## 🚀 What's New in v1.0.0

*   🎉 **First public release:** RefCap capture and exact replay, view sets with clips and audio, multi-RefCap packs with a thumbnail picker, and 16 example workflows in ComfyUI's template browser.

---

[<img src="https://img.youtube.com/vi/ORBPb_Ap-fs/maxresdefault.jpg" width="100%">](https://youtu.be/ORBPb_Ap-fs)
<p align="center"><i>One Photo, Any Shot: Consistent MiniMax H3 Characters. (Click to watch on YouTube)</i></p>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Exact Replay
**H3 RefCap Replay** hands the model exactly what *MiniMax H3 Reference to Video* makes from the original photos: in the tests its clips were **pixel-identical** to native references. Only the photo loading, resizing and VAE encode are skipped.

</td>
<td width="50%">

### 💾 Small Files
Photos are stored as PNGs and resized again at replay, bit for bit: a 1024 px photo costs about 0.7 MB instead of 12 MB. The text encoder's cached rows are optional, and only *Apply (cached)* needs them.

</td>
</tr>
<tr>
<td width="50%">

### 🔄 View Sets
Many views of one subject, each with the angle it was seen from. **Replay View Set** picks the views nearest the shot's angles, and a view set can also carry up to 3 clips with soundtracks and 3 audio references.

</td>
<td width="50%">

### 👥 Packs
Two characters and a location in one shot: chain **H3 RefCap Select** nodes, each with a thumbnail picker, into **Replay Pack**. **Bake Pack** saves a cast you keep using as one file.

</td>
</tr>
<tr>
<td width="50%">

### 🌀 Photo to Turnaround
**Orbit Views** picks a turnaround from two H3 orbit renders by the head yaw insightface measures, or from two camera pans round a place by their sideways motion. Workflow 8 turns one photo into RefCaps in a single queue.

</td>
<td width="50%">

### 🎬 16 Example Workflows
**H3 RefCap 1-16** go from capture to motion, camera and location references, and workflow 16 runs on ready-made fantasy RefCaps. Each carries a note with its steps and what the tests measured.

</td>
</tr>
</table>

---

## 📸 Examples

<table>
<tr>
<td width="50%">

![The elf ranger replayed from her RefCap](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_replay_elf.webp)

**Replay:** the elf ranger from her 5-view RefCap, walking a forest path she was never photographed in.

</td>
<td width="50%">

![The elf ranger and the dwarf warrior in the crystal cavern](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_pack_elf_dwarf.webp)

**Replay Pack:** three views each of the elf and the dwarf, plus three of the crystal cavern, in one shot.

</td>
</tr>
</table>

> Rendered at 608×352 with a community MiniMax H3 turbo finetune (int8, 8 steps) on a 12 GB GPU. The example workflows default to the official model.

---

## 📦 Installation

### Requirements

- **ComfyUI** with the core MiniMax H3 nodes (tested on 0.41.0)
- The **MiniMax H3 models** ([Models](#-models))
- **insightface** (optional, for Orbit Views' `faces` measure)

### ComfyUI Manager

Search for **`ComfyUI-RefCap`** in ComfyUI Manager and click **Install**, or run `comfy node install comfyui-refcap`.

### Manual Install

<details>
<summary><b>Linux / macOS</b></summary>

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/AEmotionStudio/ComfyUI-RefCap.git
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/AEmotionStudio/ComfyUI-RefCap.git
```

</details>

> **Note:** The nodes need nothing beyond ComfyUI, so there is nothing to `pip install`.

Restart ComfyUI after installation.

---

## 🚀 Quick Start

1. Open **H3 RefCap 1 - Capture** from ComfyUI's template browser (listed under ComfyUI-RefCap).
2. Load up to 9 photos of one person, and connect them **in the same order** to both the reference node and *H3 RefCap Capture*. Leave the reference node's prompt **empty**.
3. Set width and height to the size you'll generate at, give the RefCap a `name`, and queue. It's saved as `ComfyUI/models/refcaps/<name>.safetensors`.
4. Open **H3 RefCap 2 - Replay (exact)**, refresh the browser, pick your RefCap in `refcap`, and write the prompt with the photos as `<Picture 1>` ... `<Picture N>`.
5. Queue. The latent comes from *Empty MiniMax H3 AV Latent*, as with native references.

> **💡 Tip:** For the most identity, capture from photos of about 1024 px with `ref_image_size` = `max`: +0.06 face similarity in the tests, at about 2.2x the render time.

---

## 📖 Visual Guide

### 1. Capture a character

**H3 RefCap 1 - Capture**: five photos go to the reference node and to *H3 RefCap Capture* in the same order, the reference node's prompt stays empty, and one queue writes the RefCap. *Capture summary* shows what it stored.

![Capture: five photos of the elf ranger into one RefCap](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_capture.png)

### 2. Generate with it

**H3 RefCap 2 - Replay (exact)**: pick the RefCap, write the prompt with the photos as `<Picture 1>` ... `<Picture 5>`, and queue. The model gets exactly what the reference node would make from the original photos.

![Replay: the elf ranger's RefCap and a forest prompt](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_replay.png)

### 3. No photo set? Start from one photo

**H3 RefCap 8 - Photo to RefCap**: H3 renders two orbits round the character, one past each side; *Orbit Views* picks the frame nearest each angle by measured head yaw; and *Build View Set* saves a 24-view turnaround and a 5-view front RefCap. A full-body section does the same from a full-body photo, and a test shot uses both.

![Photo to RefCap: one photo, two orbits, picked views and two RefCaps](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_photo_to_refcap.png)

<details>
<summary><b>The 24 views it picked, and the test shot</b></summary>

![The elf ranger's 24-view turnaround, sorted by yaw](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_turnaround.png)

![Workflow 8's test shot: face and body views together](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_photo_to_refcap_test.png)

</details>

### 4. Several characters and a place in one shot

**H3 RefCap 7 - Two characters**: one *H3 RefCap Select* per RefCap. **Pick views...** opens a thumbnail picker of everything in `models/refcaps`, with each view's angle, a search box, and a running count against the 9-image cap.

![The RefCap picker browsing the elf ranger's 24-view turnaround](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_picker.png)

The Select nodes chain into *Replay Pack*, and its label guide says which `<Picture i>` is whom, so the prompt can name them.

![Replay Pack: the elf, the dwarf and the crystal cavern, with the label guide](https://github.com/AEmotionStudio/ComfyUI-RefCap/releases/download/assets-v1/refcap_pack.png)

---

## 🧩 Nodes

All nodes are in the `advanced/refcap` category.

| Node | What It Does |
| :--- | :--- |
| **H3 RefCap Capture** | Captures H3 image references (latents, the pixels the encoder saw, optionally its rows) into one file |
| **H3 RefCap Replay** | RefCap + prompt -> conditioning identical to the native node fed the original photos |
| **H3 RefCap Apply (cached)** | Applies a RefCap's cached rows and latents to an already encoded prompt |
| **H3 RefCap Build View Set** | A RefCap of many views (with yaws), up to 3 clips with soundtracks and 3 audios |
| **H3 RefCap Replay View Set** | Replays chosen views, clips and audio of a view set as native references |
| **H3 RefCap Orbit Views** | Picks a turnaround from two H3 orbit renders or camera pans: the frame nearest each angle |
| **H3 RefCap Select** | Picks views, clips and audios from one RefCap into a pack chain ("Pick views..." button) |
| **H3 RefCap Replay Pack** | Replays a pack chain from several RefCaps as one conditioning |
| **H3 RefCap Bake Pack** | Saves a pack chain as one view set |
| **H3 Save Conditioning (debug)** | Dumps a conditioning to `output/` |

RefCaps live in `ComfyUI/models/refcaps/` (`.safetensors`, metadata key `h3_refcap`).

---

## 🎬 Example Workflows

All 16 are in [`example_workflows/`](example_workflows), and ComfyUI's template browser lists them under ComfyUI-RefCap. Their file inputs name the test files; load your own photos, clips and RefCaps in their place. To try RefCap without capturing your own first, download the fantasy cast from [Hugging Face](https://huggingface.co/AEmotionStudio/H3-RefCap) into `ComfyUI/models/refcaps/` and run workflow 16.

<details>
<summary><b>H3 RefCap 1-16: RefCaps</b></summary>

| Workflow | What It Does |
| :--- | :--- |
| **1 - Capture** | Captures reference photos into a RefCap |
| **2 - Replay (exact)** | Generates with a RefCap, identical to native references |
| **3 - Apply (cached)** | Generates with a RefCap's cached rows, without the photos going through the text encoder |
| **4 - AB native vs RefCap** | Checks a RefCap against the photos it was captured from, on your character |
| **5 - View set** | Builds a view set and replays the views nearest the shot's angles |
| **6 - Clips and audio** | Up to 3 clips with soundtracks and 3 audio references in one view set |
| **7 - Two characters** | Two characters and a place in one shot, from a pack of RefCaps |
| **8 - Photo to RefCap** | One photo to face, 24-view and full-body RefCaps through H3 orbit renders, in one queue |
| **9 - Location** | A location view set from one photo of a place, or from a description |
| **10 - Audio to RefCap** | A voice RefCap from one sample |
| **11 - Video to RefCap** | Stores one clip in a RefCap |
| **12 - Motion** | A character performs a move stored as a clip RefCap |
| **13 - Camera move** | A clip RefCap as the camera-movement reference |
| **14 - Edit a clip** | Fixes one thing in a finished clip and keeps the rest of the take |
| **15 - Location walkthrough** | A location RefCap from a video walking through a place |
| **16 - Fantasy cast** | The elf, the dwarf, the crystal cavern and both voices from downloadable RefCaps |

</details>

---

## 🧠 Models

The workflows use the official MiniMax H3 files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), with the 4-step Lightning LoRA as in ComfyUI's reference-to-video template. For the base model, bypass the *Lightning LoRA* node and use 20 steps.

| Model | Size | Stored In | Download |
| :--- | :--- | :--- | :--- |
| **MiniMax H3 ref2va** (int8) | 19.53 GB | `ComfyUI/models/diffusion_models/` | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |
| **Qwen3-VL 32B** text encoder | 14.61 GB | `ComfyUI/models/text_encoders/` | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| **Video VAE** | 2.62 GB | `ComfyUI/models/vae/` | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) |
| **Audio VAE** | 577.2 MB | `ComfyUI/models/vae/` | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| **Lightning LoRA** (4 steps) | 1.82 GB | `ComfyUI/models/loras/` | [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) |
| **insightface buffalo_l** (optional) | ~300 MB | `ComfyUI/models/insightface/` | Downloaded by insightface on first use. ⚠️ Non-commercial research licence |

---

## 📚 Documentation

| Doc | What's In It |
| :--- | :--- |
| [REFCAP_README.md](REFCAP_README.md) | The full write-up: how a reference reaches the model, the file format, every node, and the evaluation |
| [REFCAP_VS_REFMOD.md](REFCAP_VS_REFMOD.md) | RefCap against ComfyUI-MiniMaxH3Mod's RefMods |
| [REFCAP_AV_EXPLAINER.md](REFCAP_AV_EXPLAINER.md) | Clips and audio in view sets |

---

## 🐛 Troubleshooting

<details>
<summary><b>"encode the references with an empty prompt"</b></summary>

*H3 RefCap Capture* stores only the references. Clear the prompt on the *MiniMax H3 Reference to Video* node that feeds it.

</details>

<details>
<summary><b>"N images connected for M references"</b></summary>

Connect the same photos, in the same order, to the reference node and to Capture, or connect none to Capture and turn on `store_rows` (an Apply-only RefCap).

</details>

<details>
<summary><b>"has no stored pixels"</b></summary>

The RefCap was captured without its photos, so only *H3 RefCap Apply (cached)* can use it. Capture it again with the photos connected to *H3 RefCap Capture*.

</details>

<details>
<summary><b>"has no cached rows"</b></summary>

*Apply (cached)* needs the text encoder's rows. Capture again with `store_rows` on, or use *H3 RefCap Replay*, which needs no rows. View sets never store them.

</details>

<details>
<summary><b>"N views selected; MiniMaxH3ReferenceToVideo takes at most 9"</b></summary>

A render takes up to 9 images, 3 clips and 3 audio references, as the native reference node does. Pick fewer views in *Replay View Set* or with the *Select* picker, whose counter warns when a selection goes over the cap.

</details>

<details>
<summary><b>The RefCap doesn't show up in the dropdown</b></summary>

RefCaps are read from `ComfyUI/models/refcaps/`. Refresh the browser after capturing one.

</details>

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, new workflows, or improvements — your help is appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the [GPL-3.0 License](LICENSE) — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Developed by [Æmotion Studio](https://aemotionstudio.org/)**

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@aemotionstudio/videos)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/UzC9353mfp)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/aemotionstudio)

</div>
