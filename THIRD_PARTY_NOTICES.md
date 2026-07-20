# Third-party notices

## SegFace

The inference model definition in `vendor/` is adapted from [Kartik-3004/SegFace](https://github.com/Kartik-3004/SegFace), copyright Kartik Narayan, licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to inclusion of the copyright and permission notice.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

The optional checkpoint is downloaded from the upstream Hugging Face repository at the pinned revision recorded in `MODEL_MANIFEST.json`. The checkpoint is not bundled.

## Ultralytics

Detection uses `ultralytics==8.4.102` under AGPL-3.0. The `face_yolov9c.pt` checkpoint is downloaded from the Apache-2.0 [Bingsu/adetailer model repository](https://huggingface.co/Bingsu/adetailer). See [Ultralytics licensing](https://www.ultralytics.com/license) and the installed package license.

## SAM 3 / SAM 3.1

Optional boundary refinement calls ComfyUI's native SAM 3/3.1 implementation. The supplied checkpoint is not redistributed. See [facebookresearch/sam3](https://github.com/facebookresearch/sam3) for its model and software terms.

## MediaPipe Face Landmarker

Optional landmark-aware cropping calls ComfyUI's native pure-PyTorch MediaPipe port and the Comfy-Org-converted face landmarker weights. The weights are downloaded during explicit setup, never at runtime.

## Other runtime libraries

ComfyUI, PyTorch, torchvision, Kornia, OpenCV, Pillow, NumPy, and safetensors retain their upstream licenses.
