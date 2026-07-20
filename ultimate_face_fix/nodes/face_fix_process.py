from __future__ import annotations

from comfy_api.latest import io

from ..core.pipeline import prepare_external_crops, repair_and_composite
from .common import FaceFixContextType, composite_inputs, sampling_inputs


class UltimateFaceFixProcess(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFixProcess",
            display_name="Ultimate Face Fix (Process)",
            category="ultimate face fix",
            description="Repairs extracted faces after an optional custom pipeline and blends them into the source image.",
            inputs=[
                io.Image.Input("extract_image"),
                FaceFixContextType.Input(
                    "face_fix_context",
                    tooltip="Connect face_fix_context from the matching Ultimate Face Fix (Extract) node.",
                ),
                io.Model.Input("model", tooltip="Generation model used for crop img2img."),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                *sampling_inputs(),
                *composite_inputs(),
            ],
            outputs=[
                io.Image.Output("fixed_image"),
                io.Image.Output("original_face_crops"),
                io.Image.Output("processed_face_crops"),
                io.Mask.Output("face_mask"),
                io.Image.Output("debug_preview"),
            ],
        )

    @classmethod
    def execute(
        cls,
        extract_image,
        face_fix_context,
        model,
        vae,
        positive,
        negative,
        repair_mode,
        custom_denoise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        mask_preset,
        sam_refine,
        mask_grow_percent,
        mask_feather_percent,
        color_match_strength,
        blend_mode,
        face_parser=None,
        sam_model=None,
    ) -> io.NodeOutput:
        input_crops, regions = prepare_external_crops(extract_image, face_fix_context)
        outputs = repair_and_composite(
            image=face_fix_context.source_image,
            original_crops=face_fix_context.original_crops,
            input_crops=input_crops,
            regions=regions,
            model=model,
            vae=vae,
            positive=positive,
            negative=negative,
            repair_mode=repair_mode,
            custom_denoise=custom_denoise,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            mask_preset=mask_preset,
            sam_refine=sam_refine,
            mask_grow_percent=mask_grow_percent,
            mask_feather_percent=mask_feather_percent,
            color_match_strength=color_match_strength,
            blend_mode=blend_mode,
            face_parser=face_parser,
            sam_model=sam_model,
        )
        return io.NodeOutput(*outputs)
