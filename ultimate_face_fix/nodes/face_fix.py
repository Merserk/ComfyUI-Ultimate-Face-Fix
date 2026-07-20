from __future__ import annotations

from comfy_api.latest import io

from ..core.pipeline import analyze, repair_and_composite
from .common import analysis_inputs, composite_inputs, sampling_inputs


class UltimateFaceFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFix",
            display_name="Ultimate Face Fix",
            category="ultimate face fix",
            description="Detects, reconstructs, semantically masks, and seamlessly blends one or many faces.",
            inputs=[
                io.Image.Input("image"),
                io.Model.Input("model", tooltip="Generation model used for crop img2img."),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                *analysis_inputs()[1:],
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
        image,
        model,
        vae,
        positive,
        negative,
        face_selection,
        detection_quality,
        max_faces,
        min_face_size,
        detection_confidence,
        context_scale,
        target_resolution,
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
        face_detector=None,
        face_parser=None,
        face_landmarker=None,
        bboxes=None,
        sam_model=None,
    ) -> io.NodeOutput:
        regions, crops = analyze(
            image=image,
            face_detector=face_detector,
            face_selection=face_selection,
            detection_quality=detection_quality,
            max_faces=max_faces,
            min_face_size=min_face_size,
            detection_confidence=detection_confidence,
            context_scale=context_scale,
            target_resolution=target_resolution,
            face_landmarker=face_landmarker,
            bboxes=bboxes,
        )
        outputs = repair_and_composite(
            image=image,
            original_crops=crops,
            input_crops=crops,
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
