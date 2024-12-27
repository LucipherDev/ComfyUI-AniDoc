# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from cotracker.models.build_cotracker import build_cotracker


def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # Guass Map
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    # isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
    return isotropicGrayscaleImage


def draw_heatmap(img, center_coordinate, heatmap_template, side, width, height):
    x1 = max(center_coordinate[0] - side, 1)
    x2 = min(center_coordinate[0] + side, width - 1)
    y1 = max(center_coordinate[1] - side, 1)
    y2 = min(center_coordinate[1] + side, height - 1)
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        print(center_coordinate, "x1, x2, y1, y2", x1, x2, y1, y2)
        return img

    need_map = cv2.resize(heatmap_template, (x2-x1, y2-y1))

    img[y1:y2,x1:x2] = need_map

    return img


def generate_gassian_heatmap(pred_tracks, pred_visibility=None, image_size=None, side=20):
    width, height = image_size
    num_frames, num_points = pred_tracks.shape[:2]

    point_index_list = [point_idx for point_idx in range(num_points)]
    heatmap_template = gen_gaussian_heatmap()


    image_list = []
    for frame_idx in range(num_frames):
        
        img = np.zeros((height, width), np.float32)
        for point_idx in point_index_list:
            px, py = pred_tracks[frame_idx, point_idx]

            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            if pred_visibility is not None:
                if (not pred_visibility[frame_idx, point_idx]):
                    continue

            img = draw_heatmap(img, (px, py), heatmap_template, side, width, height)

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        image_list.append(img)
    
    video_gaussion_map = torch.stack(image_list, dim=0)

    return video_gaussion_map


# TODO: need further check and investigation
def sample_trajectories(
        pred_tracks, pred_visibility, 
        max_points=10,
        motion_threshold=1,
        vis_threshold=5,

    ):
    # pred_tracks: (b, f, num_points, 2)
    # pred_visibility: (b, f, num_points)
    batch_size, num_frames, num_points = pred_visibility.shape

    # 1. Remove points with low visibility
    mask = pred_visibility.sum(dim=1) > vis_threshold
    mask = mask.unsqueeze(1).repeat(1, num_frames, 1)
    pred_tracks = pred_tracks[mask].view(batch_size, num_frames, -1, 2)
    pred_visibility = pred_visibility[mask].view(batch_size, num_frames, -1)

    # 2. Thresholding: remove points with too small motions
    # compute the motion of each point
    diff = pred_tracks[:, 1:] - pred_tracks[:, :-1]
    # (b, f-1, num_points), sqrt(x^2 + y^2)
    motion = torch.norm(diff, dim=-1)
    # (b, num_points), mean motion for each point
    motion = torch.mean(motion, dim=1)
    # apply threshold
    mask = motion > motion_threshold  # (b, num_points)
    assert mask.shape[0] == 1
    num_keeped = mask.sum()
    if num_keeped < max_points:
        indices = torch.argsort(motion, dim=-1, descending=True)[:, :max_points]  # (bs, max_points)
        mask = torch.zeros_like(mask)  # (bs, num_points)
        # set mask to 1 for the top max_points
        mask[0, indices] = 1
        num_keeped = mask.sum()  # note sometimes mask.sum() < max_points

    motion = motion[mask].view(batch_size, num_keeped)
    # keep shape
    mask = mask.unsqueeze(1).repeat(1, num_frames, 1)
    pred_tracks = pred_tracks[mask].view(batch_size, num_frames, num_keeped, 2)
    pred_visibility = pred_visibility[mask].view(batch_size, num_frames, num_keeped)


    # 3. Sampling with larger prob for large motions
    num_points = min(max_points, num_keeped)
    if num_points == 0:
        warnings.warn("No points left after filtering")
        return None, None

    prob = motion / motion.max()
    prob = prob / prob.sum()
    sampled_indices = torch.multinomial(prob, num_points, replacement=False)

    sampled_indices = sampled_indices.squeeze(0)  # (num_points, )
    pred_tracks_sampled = pred_tracks[:, :, sampled_indices]
    pred_visibility_sampled = pred_visibility[:, :, sampled_indices]

    return pred_tracks_sampled, pred_visibility_sampled
def sample_trajectories_with_ref(
        pred_tracks, pred_visibility, coords0,
        max_points=10,
        motion_threshold=1,
        vis_threshold=5,
    ):

 

    batch_size, num_frames, num_points = pred_visibility.shape


    visibility_sum = pred_visibility.sum(dim=1)
    vis_mask = visibility_sum > vis_threshold  # (batch_size, num_points)



    pred_tracks = pred_tracks * vis_mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, num_frames, num_points, 2)
    pred_visibility = pred_visibility * vis_mask.unsqueeze(1)

   
    indices = vis_mask.nonzero(as_tuple=False)  # (num_visible_points, 2)
    if indices.size(0) == 0:
        warnings.warn("No points left after visibility filtering")
        return None, None, None

    batch_indices, point_indices = indices[:, 0], indices[:, 1]

    coords0_filtered = coords0[batch_indices, point_indices]  # (num_visible_points, 2)


    diff = pred_tracks[:, 1:] - pred_tracks[:, :-1]  # (batch_size, num_frames-1, num_points, 2)
    motion = torch.norm(diff, dim=-1).mean(dim=1)  # (batch_size, num_points)

    motion_mask = motion > motion_threshold
    combined_mask = vis_mask & motion_mask  # (batch_size, num_points)


    indices = combined_mask.nonzero(as_tuple=False)
    if indices.size(0) == 0:
        warnings.warn("No points left after motion filtering")
        return None, None, None

    batch_indices, point_indices = indices[:, 0], indices[:, 1]

    pred_tracks_filtered = pred_tracks[batch_indices, :, point_indices, :]  # (num_filtered_points, num_frames, 2)
    pred_visibility_filtered = pred_visibility[batch_indices, :, point_indices]  # (num_filtered_points, num_frames)
    coords0_filtered = coords0[batch_indices, point_indices, :]  # (num_filtered_points, 2)
    motion_filtered = motion[batch_indices, point_indices]  # (num_filtered_points)


    num_keeped = motion_filtered.size(0)
    num_points_sampled = min(max_points, num_keeped)
    if num_points_sampled == 0:
        warnings.warn("No points left after filtering")
        return None, None, None

    prob = motion_filtered / motion_filtered.max()
    prob = prob / prob.sum()
    sampled_indices = torch.multinomial(prob, num_points_sampled, replacement=False)

    pred_tracks_sampled = pred_tracks_filtered[sampled_indices]  # (num_points_sampled, num_frames, 2)
    pred_visibility_sampled = pred_visibility_filtered[sampled_indices]  # (num_points_sampled, num_frames)
    coords0_sampled = coords0_filtered[sampled_indices]  # (num_points_sampled, 2)


    pred_tracks_sampled = pred_tracks_sampled.view(batch_size, num_points_sampled, num_frames, 2).transpose(1, 2)
    pred_visibility_sampled = pred_visibility_sampled.view(batch_size, num_points_sampled, num_frames).transpose(1, 2)
    coords0_sampled = coords0_sampled.view(batch_size, num_points_sampled, 2)

    return pred_tracks_sampled, pred_visibility_sampled, coords0_sampled


class CoTrackerPredictor(torch.nn.Module):
    def __init__(
            self,
            checkpoint="./checkpoints/cotracker2.pth",
            shift_grid=False,
        ):

        super().__init__()
        self.support_grid_size = 6
        model = build_cotracker(checkpoint)
        self.interp_shape = model.model_resolution
        self.model = model
        self.model.eval()
        self.shift_grid = shift_grid

    @torch.no_grad()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (B, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
    ):
        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )
        else:
            tracks, visibilities = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )

        return tracks, visibilities

    def _compute_dense_tracks(self, video, grid_query_frame, grid_size=80, backward_tracking=False):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):
        B, T, C, H, W = video.shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device, shift_grid=self.shift_grid)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(
                self.support_grid_size, self.interp_shape, device=video.device, shift_grid=self.shift_grid,
            )
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        tracks, visibilities, __ = self.model.forward(video=video, queries=queries, iters=6)

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities
            )
            if add_support_grid:
                queries[:, -self.support_grid_size**2 :, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size**2]
            visibilities = visibilities[:, :, : -self.support_grid_size**2]
        thr = 0.9
        visibilities = visibilities > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, inv_visibilities, __ = self.model(video=inv_video, queries=inv_queries, iters=6)

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        arange = torch.arange(video.shape[1], device=queries.device)[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities


class CoTrackerOnlinePredictor(torch.nn.Module):
    def __init__(self, checkpoint="./checkpoints/cotracker2.pth"):
        super().__init__()
        self.support_grid_size = 6
        model = build_cotracker(checkpoint)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        queries: torch.Tensor = None,
        grid_size: int = 10,
        grid_query_frame: int = 0,
        add_support_grid=False,
    ):
        B, T, C, H, W = video_chunk.shape
        # Initialize online video processing and save queried points
        # This needs to be done before processing *each new video*
        if is_first_step:
            self.model.init_video_online_processing()
            if queries is not None:
                B, N, D = queries.shape
                assert D == 3
                queries = queries.clone()
                queries[:, :, 1:] *= queries.new_tensor(
                    [
                        (self.interp_shape[1] - 1) / (W - 1),
                        (self.interp_shape[0] - 1) / (H - 1),
                    ]
                )
            elif grid_size > 0:
                grid_pts = get_points_on_a_grid(
                    grid_size, self.interp_shape, device=video_chunk.device
                )
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                    dim=2,
                )
            if add_support_grid:
                grid_pts = get_points_on_a_grid(
                    self.support_grid_size, self.interp_shape, device=video_chunk.device
                )
                grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
                queries = torch.cat([queries, grid_pts], dim=1)
            self.queries = queries
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        tracks, visibilities, __ = self.model(
            video=video_chunk,
            queries=self.queries,
            iters=6,
            is_online=True,
        )
        thr = 0.9
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )
