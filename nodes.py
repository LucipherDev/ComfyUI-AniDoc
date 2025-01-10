import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import snapshot_download
import torchvision.transforms as T

from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import rbd

from models_diffusers.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)
from models_diffusers.controlnet_svd import ControlNetSVDModel

try:
    from cotracker.predictor import CoTrackerPredictor, sample_trajectories_with_ref
except:
    pass

from lineart_extractor.canny import CannyDetector
from lineart_extractor.hed import HEDdetector
from lineart_extractor.lineart import LineartDetector
from lineart_extractor.lineart_anime import LineartAnimeDetector

from pipelines.AniDoc import AniDocPipeline

from comfy.utils import ProgressBar
from comfy.model_management import XFORMERS_IS_AVAILABLE
import folder_paths

log = logging.getLogger("AniDoc")

DIFFUSERS_DIR = os.path.join(folder_paths.models_dir, "diffusers")
ANIDOC_DIR = os.path.join(DIFFUSERS_DIR, "anidoc")
ANIDOC_CONTROLNET_DIR = os.path.join(ANIDOC_DIR, "controlnet")
SVD_I2V_DIR = os.path.join(
    DIFFUSERS_DIR,
    "stable-video-diffusion-img2vid-xt",
)

COTRACKER_DIR = os.path.join(folder_paths.models_dir, "cotracker")
if "cotracker" not in folder_paths.folder_names_and_paths:
    current_paths = [COTRACKER_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["cotracker"]
folder_paths.folder_names_and_paths["cotracker"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


class AniDocLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("ANIDOC_PIPELINE",)
    RETURN_NAMES = ("anidoc_pipeline",)
    OUTPUT_TOOLTIPS = ("AniDoc Pipeline",)

    CATEGORY = "AniDoc"
    FUNCTION = "load_anidoc"
    DESCRIPTION = "Load AniDoc pipeline"

    def __init__(self):
        self.pipeline = None

    def load_anidoc(
        self,
        anidoc_path=ANIDOC_DIR,
        svd_img2vid_path=SVD_I2V_DIR,
        controlnet_path=ANIDOC_CONTROLNET_DIR,
        device="cuda",
        dtype=torch.float16,
    ):

        if self.pipeline is None:
            pbar = ProgressBar(5)

            pbar.update(1)

            log.info(f"Loading model from: {anidoc_path}")
            log.info("Missing models will be downloaded")

            try:
                snapshot_download(
                    repo_id="Yhmeng1106/anidoc",
                    ignore_patterns=["*.md"],
                    local_dir=DIFFUSERS_DIR,
                    local_dir_use_symlinks=False,
                )
            except:
                log.info("Couldn't download models")

            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                anidoc_path,
                subfolder="unet",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                custom_resume=True,
            )
            unet.to(device, dtype)

            pbar.update(1)

            log.info(f"Loading controlnet from: {controlnet_path}")
            controlnet = ControlNetSVDModel.from_pretrained(controlnet_path)

            controlnet.to(device, dtype)

            pbar.update(1)

            if XFORMERS_IS_AVAILABLE:
                log.info("Enabling XFormers")
                unet.enable_xformers_memory_efficient_attention()

            log.info(f"Loading model from: {svd_img2vid_path}")
            log.info("Missing models will be downloaded")

            try:
                snapshot_download(
                    repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1",
                    allow_patterns=["*.json", "*fp16*"],
                    ignore_patterns=["*unet*"],
                    local_dir=svd_img2vid_path,
                    local_dir_use_symlinks=False,
                )
            except:
                log.info("Couldn't download models")

            pbar.update(1)

            pipeline = AniDocPipeline.from_pretrained(
                svd_img2vid_path,
                unet=unet,
                controlnet=controlnet,
                low_cpu_mem_usage=False,
                torch_dtype=dtype,
                variant="fp16",
            )
            pipeline.to(device)

            self.pipeline = pipeline

            pbar.update(1)

        return (self.pipeline,)


class LoadAniDocCoTracker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracking": ("BOOLEAN", {"default": True}),
                "cotracker_model": (folder_paths.get_filename_list("cotracker"),),
                "tracker_shift_grid": ([0, 1], {"default": 0}),
                "tracker_grid_size": (
                    "INT",
                    {"default": 8, "min": 2, "max": 32, "step": 2},
                ),
                "tracker_grid_query_frame": ("INT", {"default": 0}),
                "tracker_backward_tracking": ("BOOLEAN", {"default": False}),
                "max_points": (
                    "INT",
                    {"default": 50, "min": 10, "max": 100, "step": 5},
                ),
            },
        }

    RETURN_TYPES = ("ANIDOC_COTRACKER",)
    RETURN_NAMES = ("cotracker",)
    OUTPUT_TOOLTIPS = ("CoTracker",)

    CATEGORY = "AniDoc"
    FUNCTION = "load_tracker"
    DESCRIPTION = "Load CoTracker for AniDoc"

    def __init__(self):
        self.tracker = None
        self.tracker_shift_grid = None

    def load_tracker(
        self,
        cotracker_model,
        tracking,
        tracker_shift_grid,
        tracker_grid_size,
        tracker_grid_query_frame,
        tracker_backward_tracking,
        max_points,
        device="cuda",
        dtype=torch.float32,
    ):
        try:
            import cotracker
        except:
            raise ImportError("Couldn't import cotracker module. Please install it to use this node")
        
        if tracking:
            if self.tracker is None or self.tracker_shift_grid != tracker_shift_grid:
                cotracker_model_path = folder_paths.get_full_path(
                    "cotracker", cotracker_model
                )

                log.info(f"Loading tracker model from {cotracker_model_path}")

                tracker = CoTrackerPredictor(
                    checkpoint=cotracker_model_path,
                    shift_grid=tracker_shift_grid,
                )
                tracker.requires_grad_(False)
                tracker.to(device, dtype=dtype)

                self.tracker = tracker
                self.tracker_shift_grid = tracker_shift_grid

            return (
                {
                    "tracking": tracking,
                    "tracker": self.tracker,
                    "grid_size": self.tracker_shift_grid,
                    "grid_query_frame": tracker_grid_query_frame,
                    "backward_tracking": tracker_backward_tracking,
                    "max_points": max_points,
                },
            )

        else:
            return (
                {
                    "tracking": False,
                    "tracker": None,
                    "grid_size": tracker_grid_size,
                    "grid_query_frame": tracker_grid_query_frame,
                    "backward_tracking": tracker_backward_tracking,
                    "max_points": max_points,
                },
            )


class GetAniDocControlnetImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_images": ("IMAGE",),
                "lineart_detector": (
                    ["none", "canny", "hed", "lineart", "lineart_anime"],
                    {"default": "lineart"},
                ),
                "sketch_quantization": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "height": ("INT", {"default": 320, "min": 64, "max": 1024, "step": 8}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("controlnet_images",)
    OUTPUT_TOOLTIPS = ("Processed controlnet images",)

    CATEGORY = "AniDoc"
    FUNCTION = "get_controlnet_images"
    DESCRIPTION = "Get lineart controlnet images for AniDoc"

    def load_lineart_detector(self, lineart_detector, device):
        if lineart_detector == "none":
            return None

        if lineart_detector == "canny":
            return CannyDetector()

        elif lineart_detector == "hed":
            return HEDdetector()

        elif lineart_detector == "lineart":
            return LineartDetector(device)

        elif lineart_detector == "lineart_anime":
            return LineartAnimeDetector(device)

    def invert_images(self, image):
        if np.unique(image).size == 2:
            return cv2.bitwise_not(image)
        else:
            return 255 - image

    def get_controlnet_images(
        self,
        input_images,
        lineart_detector,
        sketch_quantization,
        width,
        height,
        device="cuda",
    ):
        pbar = ProgressBar(len(input_images) + 1)

        log.info(f"Loading lineart detector: {lineart_detector}")
        detector = self.load_lineart_detector(lineart_detector, device)

        pbar.update(1)

        log.info("Processing images with lineart detector")

        controlnet_images = []

        for img_tensor in input_images:
            sketch = (img_tensor * 255.0).clamp(0, 255).byte().cpu().numpy()

            if detector is not None:
                if lineart_detector == "canny":
                    sketch = detector(sketch, 100, 200)
                else:
                    sketch = detector(sketch)

                if lineart_detector in ["canny", "hed"]:
                    sketch = self.invert_images(sketch)

            if len(sketch.shape) == 2:
                sketch = np.repeat(sketch[:, :, np.newaxis], 3, axis=2)

            if not sketch_quantization:
                sketch = (sketch > 200).astype(np.uint8) * 255

            sketch = torch.nn.functional.interpolate(
                torch.from_numpy(sketch).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            controlnet_images.append(sketch)

            pbar.update(1)

        controlnet_images = torch.stack(controlnet_images)

        controlnet_images = controlnet_images.permute(0, 2, 3, 1)

        return (controlnet_images,)


class AniDocSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anidoc_pipeline": ("ANIDOC_PIPELINE",),
                "controlnet_images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "repeat_matching": ("BOOLEAN", {"default": False}),
                "fps": ("INT", {"default": 7, "min": 7, "max": 100, "step": 1}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1}),
                "noise_aug": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "motion_bucket_id": (
                    "INT",
                    {"default": 127, "min": 1, "max": 300, "step": 1},
                ),
                "decode_chunk_size": (
                    "INT",
                    {"default": 8, "min": 1, "max": 256, "step": 1},
                ),
            },
            "optional": {
                "cotracker": ("ANIDOC_COTRACKER",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video_frames",)
    OUTPUT_TOOLTIPS = ("Video Frames",)

    CATEGORY = "AniDoc"
    FUNCTION = "sample"
    DESCRIPTION = "Sampler for AniDoc"

    def safe_round(self, coords, size):
        height, width = size[1], size[2]
        rounded_coords = np.round(coords).astype(int)
        rounded_coords[:, 0] = np.clip(rounded_coords[:, 0], 0, width - 1)
        rounded_coords[:, 1] = np.clip(rounded_coords[:, 1], 0, height - 1)

        return rounded_coords

    def generate_point_map(self, size, coords0, coords1):
        h, w = size[1], size[2]
        mask0 = np.zeros((h, w), dtype=np.uint8)
        mask1 = np.zeros((h, w), dtype=np.uint8)

        for i, (coord0, coord1) in enumerate(zip(coords0, coords1)):
            x0, y0 = int(round(coord0[0])), int(round(coord0[1]))
            x1, y1 = int(round(coord1[0])), int(round(coord1[1]))

            if 0 <= x0 < w and 0 <= y0 < h:
                mask0[y0, x0] = i + 1

            if 0 <= x1 < w and 0 <= y1 < h:
                mask1[y1, x1] = i + 1

        return mask0, mask1

    def select_multiple_points(self, points0, points1, num_points):
        N = len(points0)
        num_points = min(num_points, N)
        indices = np.random.choice(N, size=num_points, replace=False)

        return points0[indices], points1[indices]

    def generate_point_map_frames(self, size, coords0, coords1, visibility):
        h, w = size[1], size[2]
        mask0 = np.zeros((h, w), dtype=np.uint8)
        num_frames = coords1.shape[0]
        mask1 = np.zeros((num_frames, h, w), dtype=np.uint8)

        for i, coord0 in enumerate(coords0):
            x0, y0 = int(round(coord0[0])), int(round(coord0[1]))

            if 0 <= x0 < w and 0 <= y0 < h:
                mask0[y0, x0] = i + 1

        for frame_idx in range(num_frames):
            coords_frame = coords1[frame_idx]

            for i, coord1 in enumerate(coords_frame):
                x1, y1 = int(round(coord1[0])), int(round(coord1[1]))

                if 0 <= x1 < w and 0 <= y1 < h and visibility[frame_idx, i]:
                    mask1[frame_idx, y1, x1] = i + 1

        return mask0, mask1

    def get_conditioning(
        self,
        tracker,
        extractor,
        matcher,
        controlnet_images,
        reference_image,
        frames,
        width,
        height,
        repeat_matching,
        tracking,
        tracker_grid_size,
        tracker_grid_query_frame,
        tracker_backward_tracking,
        max_points,
        device,
        dtype=torch.float16,
    ):
        controlnet_sketch_condition = [
            T.ToTensor()(img).unsqueeze(0) for img in controlnet_images
        ]
        controlnet_sketch_condition = (
            torch.cat(controlnet_sketch_condition, dim=0)
            .unsqueeze(0)
            .to(device, dtype=dtype)
        )
        controlnet_sketch_condition = (controlnet_sketch_condition - 0.5) / 0.5

        with torch.no_grad():
            ref_img_value = (
                T.ToTensor()(reference_image).to(device, dtype=dtype).to(torch.float32)
            )
            current_img = (
                T.ToTensor()(controlnet_images[0])
                .to(device, dtype=dtype)
                .to(torch.float32)
            )
            feats0 = extractor.extract(ref_img_value)
            feats1 = extractor.extract(current_img)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01["matches"]
            points0 = feats0["keypoints"][matches[..., 0]]
            points1 = feats1["keypoints"][matches[..., 1]]
            points0 = points0.cpu().numpy()
            points1 = points1.cpu().numpy()

            points0 = self.safe_round(points0, current_img.shape)
            points1 = self.safe_round(points1, current_img.shape)

            num_points = min(50, points0.shape[0])
            points0, points1 = self.select_multiple_points(points0, points1, num_points)
            mask1, mask2 = self.generate_point_map(
                size=current_img.shape, coords0=points0, coords1=points1
            )
            point_map1 = torch.from_numpy(mask1)
            point_map2 = torch.from_numpy(mask2)
            point_map1 = (
                point_map1.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device, dtype=dtype)
            )
            point_map2 = (
                point_map2.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device, dtype=dtype)
            )
            point_map = torch.cat([point_map1, point_map2], dim=2)
            conditional_pixel_values = ref_img_value.unsqueeze(0).unsqueeze(0)
            conditional_pixel_values = (conditional_pixel_values - 0.5) / 0.5

            point_map_with_ref = torch.cat([point_map, conditional_pixel_values], dim=2)
            original_shape = list(point_map_with_ref.shape)
            new_shape = original_shape.copy()
            new_shape[1] = frames - 1

            if repeat_matching:
                matching_controlnet_image = point_map_with_ref.repeat(
                    1, frames, 1, 1, 1
                )
                controlnet_condition = torch.cat(
                    [controlnet_sketch_condition, matching_controlnet_image], dim=2
                )
            elif tracking:
                with torch.no_grad():
                    video_for_tracker = (
                        controlnet_sketch_condition * 0.5 + 0.5
                    ) * 255.0
                    queries = np.insert(points1, 0, 0, axis=1)
                    queries = (
                        torch.from_numpy(queries).to(device, torch.float).unsqueeze(0)
                    )

                    if queries.shape[1] == 0:
                        mask1 = np.zeros((height, width), dtype=np.uint8)
                        mask2 = np.zeros((frames, height, width), dtype=np.uint8)
                    else:
                        pred_tracks, pred_visibility = tracker(
                            video_for_tracker.to(dtype=torch.float32),
                            queries=queries,
                            grid_size=tracker_grid_size,
                            grid_query_frame=tracker_grid_query_frame,
                            backward_tracking=tracker_backward_tracking,
                        )
                        (
                            pred_tracks_sampled,
                            pred_visibility_sampled,
                            points0_sampled,
                        ) = sample_trajectories_with_ref(
                            pred_tracks.cpu(),
                            pred_visibility.cpu(),
                            torch.from_numpy(points0).unsqueeze(0).cpu(),
                            max_points=max_points,
                            motion_threshold=1,
                            vis_threshold=3,
                        )
                        if pred_tracks_sampled is None:
                            mask1 = np.zeros((height, width), dtype=np.uint8)
                            mask2 = np.zeros((frames, height, width), dtype=np.uint8)
                        else:
                            pred_tracks_sampled = (
                                pred_tracks_sampled.squeeze(0).cpu().numpy()
                            )
                            pred_visibility_sampled = (
                                pred_visibility_sampled.squeeze(0).cpu().numpy()
                            )
                            points0_sampled = points0_sampled.squeeze(0).cpu().numpy()
                            for frame_id in range(frames):
                                pred_tracks_sampled[frame_id] = self.safe_round(
                                    pred_tracks_sampled[frame_id], current_img.shape
                                )
                            points0_sampled = self.safe_round(
                                points0_sampled, current_img.shape
                            )

                            mask1, mask2 = self.generate_point_map_frames(
                                size=current_img.shape,
                                coords0=points0_sampled,
                                coords1=pred_tracks_sampled,
                                visibility=pred_visibility_sampled,
                            )

                    point_map1 = torch.from_numpy(mask1)
                    point_map2 = torch.from_numpy(mask2)
                    point_map1 = (
                        point_map1.unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(1, frames, 1, 1, 1)
                        .to(device, dtype=dtype)
                    )
                    point_map2 = (
                        point_map2.unsqueeze(0).unsqueeze(2).to(device, dtype=dtype)
                    )
                    point_map = torch.cat([point_map1, point_map2], dim=2)

                    conditional_pixel_values_repeat = conditional_pixel_values.repeat(
                        1, frames, 1, 1, 1
                    )

                    point_map_with_ref = torch.cat(
                        [point_map, conditional_pixel_values_repeat], dim=2
                    )
                    controlnet_condition = torch.cat(
                        [controlnet_sketch_condition, point_map_with_ref], dim=2
                    )
            else:
                zero_tensor = torch.zeros(new_shape).to(device, dtype=dtype)
                matching_controlnet_image = torch.cat(
                    (point_map_with_ref, zero_tensor), dim=1
                )
                controlnet_condition = torch.cat(
                    [controlnet_sketch_condition, matching_controlnet_image], dim=2
                )

        return controlnet_condition

    def sample(
        self,
        anidoc_pipeline,
        controlnet_images,
        reference_image,
        repeat_matching=False,
        cotracker={
            "tracking": False,
            "tracker": None,
            "grid_size": 8,
            "grid_query_frame": 0,
            "backward_tracking": False,
            "max_points": 50,
        },
        fps=7,
        steps=25,
        noise_aug=0.02,
        seed=0,
        motion_bucket_id=127,
        decode_chunk_size=8,
        device="cuda",
        dtype=torch.float16,
    ):
        extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        width, height = controlnet_images.shape[2], controlnet_images.shape[1]

        reference_image = reference_image.permute(0, 3, 1, 2)
        reference_image = torch.nn.functional.interpolate(
            reference_image, size=(height, width), mode="bilinear", align_corners=False
        )
        reference_image = reference_image.squeeze(0)

        reference_image = reference_image.permute(1, 2, 0).cpu().numpy()
        reference_image = (reference_image * 255).clip(0, 255).astype(np.uint8)
        reference_image = Image.fromarray(reference_image)

        images_np = (controlnet_images.cpu().numpy() * 255).astype(np.uint8)
        controlnet_images = [Image.fromarray(image) for image in images_np]

        pbar = ProgressBar(steps + 1)

        log.info("Getting controlnet conditioning for AniDoc")

        controlnet_condition = self.get_conditioning(
            cotracker["tracker"],
            extractor,
            matcher,
            controlnet_images,
            reference_image,
            len(controlnet_images),
            width,
            height,
            repeat_matching,
            cotracker["tracking"],
            cotracker["grid_size"],
            cotracker["grid_query_frame"],
            cotracker["backward_tracking"],
            cotracker["max_points"],
            device,
            dtype,
        )

        pbar.update(1)

        generator = torch.manual_seed(seed)

        log.info("Generating video frames with AniDoc")

        with torch.inference_mode():
            video_frames = anidoc_pipeline(
                reference_image,
                controlnet_condition,
                width=width,
                height=height,
                num_frames=len(controlnet_images),
                num_inference_steps=steps,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug,
                decode_chunk_size=decode_chunk_size,
                generator=generator,
                callback_on_step_end=lambda *args, **kwargs: (pbar.update(1), kwargs)[
                    -1
                ],
            ).frames[0]

        tensor_frames = [T.ToTensor()(img) for img in video_frames]
        tensor_frames = torch.stack(tensor_frames)
        tensor_frames = tensor_frames.permute(0, 2, 3, 1)

        return (tensor_frames,)


NODE_CLASS_MAPPINGS = {
    "AniDocLoader": AniDocLoader,
    "LoadCoTracker": LoadAniDocCoTracker,
    "GetAniDocControlnetImages": GetAniDocControlnetImages,
    "AniDocSampler": AniDocSampler,
}
