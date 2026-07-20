SEGFACE_URL = (
    "https://huggingface.co/kartiknarayan/SegFace/resolve/"
    "5e093b03c0523f7f32a9845bbbc75ecb027c8bee/convnext_celeba_512/model_299.pt"
)
YOLO_URL = (
    "https://huggingface.co/Bingsu/adetailer/resolve/"
    "53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov9c.pt"
)
MEDIAPIPE_URL = (
    "https://huggingface.co/Comfy-Org/mediapipe/resolve/"
    "b98d050e8bf406f14f063bdba697e5b5391bbbf5/detection/"
    "mediapipe_face_fp32.safetensors"
)

SEGFACE_SOURCE_SHA256 = "3cd535bfcfab4e5c67b7df7015370c2b18cb9a4044ecb00c7c1fa8555558c507"
YOLO_SHA256 = "d02fe493c31e1bbc6450f4dc6f1db86a02a59322ff1f6d318da0661d72ddd084"
MEDIAPIPE_SHA256 = "a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a"

SEGFACE_REQUIRED_TENSORS = {
    "backbone.0.0.0.weight": (128, 3, 4, 4),
    "linear_fuse.weight": (256, 1024, 1, 1),
    "pe_layer.positional_encoding_gaussian_matrix": (2, 128),
}
