# ComfyUI Ultimate Face Fix

Model-aware face repair for one or many faces. It detects faces, repairs each square crop with your connected generation model, builds a precise semantic mask, and blends only the face back into the original image. Anime, illustration, and realistic checkpoints are supported.

<img width="960" height="540" alt="Cover_1920x1080" src="https://github.com/user-attachments/assets/ffb9b7fa-6e95-454d-819c-e511cc988e57" />

<img width="1672" height="941" alt="before_after_composite" src="https://github.com/user-attachments/assets/0375234a-08e2-4463-ab2b-ed9f6968c9f3" />

**Supported models:** SD 1.5, SD 2.x, SDXL, SDXL Turbo, Illustrious XL, Pony Diffusion, NoobAI, Stable Cascade, SD3/SD3.5, FLUX.1, FLUX.2, Z-Image, Qwen-Image, HunyuanDiT, Hunyuan Image 2.1, PixArt Alpha/Sigma, AuraFlow, Lumina Image 2.0, HiDream-I1, Kolors, Sana, Krea 2, OmniGen/OmniGen2, and more.

## Install

In **ComfyUI Manager**, search for **Ultimate Face Fix** and select **Install**. Manager installs the Python dependencies and automatically downloads, verifies, and prepares the three required face-analysis models. Restart ComfyUI when setup finishes.

For a manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-Ultimate-Face-Fix.git
cd ComfyUI-Ultimate-Face-Fix
pip install -r requirements.txt
python scripts/prepare_models.py --comfy-root ../..
```

All five extension nodes are under **Add Node → ultimate face fix**. For the integrated workflow, add the detector loader, parser loader, and **Ultimate Face Fix**, then connect them. Use a generation model matching the source image. Native MediaPipe and SAM 3/3.1 connections are optional.

Setup uses Hugging Face's accelerated Xet downloader when available. The large SegFace `.pt` source is placed in `face_fix/parsers`, converted to the runtime `.safetensors`, and removed after a successful conversion; pass `--keep-segface-source` to retain it.

```text
ComfyUI/models/
├── detection/mediapipe_face_fp32.safetensors
└── face_fix/
    ├── detectors/face_yolov9c.pt
    └── parsers/segface_convnext_celeba_512.safetensors
```

## Settings

| Control | What it does |
|---|---|
| `image` | Source image or image batch. |
| `model`, `vae`, `positive`, `negative` | Generation checkpoint components and prompts used for crop img2img. |
| `face_detector` | Required YOLO loader connection unless `bboxes` is connected. |
| `face_parser` | Recommended SegFace loader connection; without it, masks fall back to SAM/landmarks/geometry. |
| `face_landmarker` | Optional native MediaPipe connection for more stable crops and masks. |
| `bboxes` | Optional external boxes that completely replace YOLO detection. |
| `face_selection` | `all`, `largest`, or the largest `max_faces`. |
| `detection_quality` | `maximum` adds tiled detection; `balanced` uses the full image only. |
| `max_faces` | Maximum faces selected and processed sequentially. |
| `min_face_size` | Ignores faces smaller than this many source pixels. |
| `detection_confidence` | Minimum YOLO confidence. Lower values find more faces and false positives. |
| `context_scale` | Size of the square around the face; `1.8` is the recommended default. |
| `target_resolution` | `512`–`1536` resizes every crop; `none` processes each native crop without resizing. |
| `repair_mode` | Denoise presets: Detail `0.28`, Repair `0.42`, Reconstruct `0.58`, or Custom. |
| `custom_denoise` | Used only by Custom repair mode. Higher values change identity/geometry more. |
| `seed` | Base seed; each face receives a deterministic derived seed. |
| `steps`, `cfg` | Sampling steps and guidance strength. |
| `sampler_name`, `scheduler` | ComfyUI sampler and scheduler used for every face crop. |
| `mask_preset` | `core_face`; `portrait_safe` adds ears/glasses; `head` also adds hair. |
| `sam_model`, `sam_refine` | Optional SAM boundary model and `auto`, `always`, or `off` usage. |
| `mask_grow_percent` | Expands the semantic mask relative to face size. |
| `mask_feather_percent` | Softens the mask edge relative to face size. |
| `color_match_strength` | Matches processed colors to the original boundary; `0` disables it. |
| `blend_mode` | `multiband` hides seams best; `alpha` is faster. |

Outputs are the fixed image, original face crops, processed face crops, full-image face mask, and debug preview.

## Custom crop pipeline

Use **Ultimate Face Fix (Extract)** and **Ultimate Face Fix (Process)** when you want custom face upscaling or enhancement before the normal generation-model repair. Send `extract_image` through your image-processing nodes, then connect the result and the matching `face_fix_context` to Process. The custom pipeline may uniformly resize or enhance crops, but it must preserve their square shape, alignment, batch count, and face order. Process returns the same five outputs as the integrated node.

Example workflows:

- [`gen_to_img_face_fix_SDXL.json`](examples/gen_to_img_face_fix_SDXL.json) — generate with SDXL, then fix every face.
- [`img_to_img_face_fix_SDXL.json`](examples/img_to_img_face_fix_SDXL.json) — fix every face in an existing image with SDXL.
- [`gen_to_img_face_fix_KREA_2_Turbo.json`](examples/gen_to_img_face_fix_KREA_2_Turbo.json) — generate with local Krea 2 Turbo, then fix every face.
- [`img_to_img_face_fix_KREA_2_Turbo.json`](examples/img_to_img_face_fix_KREA_2_Turbo.json) — fix every face in an existing image with local Krea 2 Turbo.

Generation model and input-image selectors are intentionally blank. Choose your own compatible files before running. Krea 2 Turbo workflows use 8 steps and CFG 1.0 for generation and face repair; SDXL workflows use 25 steps and CFG 5.5.

## License

AGPL-3.0-or-later. Model weights are downloaded separately and keep their upstream terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
