# Shree KRISHNAya Namaha
# Differentiable warper implemented in PyTorch. Warping is done on batches.
# Tested on PyTorch 1.8.1
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from utils import CommonUtils, MpiUtils


class Warper:
    def __init__(self, device: str = 'gpu0'):
        self.device = CommonUtils.get_device(device)
        return

    def forward_warp_frame2mpi(self, input_batch: dict) -> dict:
        b, c, h, w = input_batch['frame1'].shape
        pos_vectors = self.create_grid(b, h, w).to(input_batch['frame1'])
        depth1a = input_batch['depth1']
        if ('local_flow12' in input_batch) and (input_batch['local_flow12'] is not None):
            pos_vectors += input_batch['local_flow12'][:, :2]
            depth1a += input_batch['local_flow12'][:, 2:3]

        trans_points12 = self.compute_transformed_points(depth1a, input_batch['transformation1'],
                                                         input_batch['transformation2'],
                                                         input_batch['intrinsic1'],
                                                         input_batch['intrinsic2'],
                                                         pos_vectors=pos_vectors)
        trans_depth2 = trans_points12[:, 2:3]
        trans_coordinates12 = trans_points12[:, :2] / trans_depth2

        if 'depth2_planes' in input_batch:
            depth2_planes = input_batch['depth2_planes']
        else:
            min_depth = trans_depth2.reshape((b, h*w)).min(dim=1)[0]
            max_depth = torch.clip(trans_depth2.reshape((b, h*w)).max(dim=1)[0], min=0, max=1000)
            depth2_planes = MpiUtils.get_depth_planes_tr(input_batch['num_mpi_planes'], min_depth, max_depth)

        warped_mpi2_rgb, warped_mpi2_alpha = self.bilinear_splatting_frame2mpi(input_batch['frame1'],
                                                                               trans_depth2,
                                                                               trans_coordinates12,
                                                                               depth2_planes)

        output_batch = {
            'warped_mpi2_rgb': warped_mpi2_rgb,
            'warped_mpi2_alpha': warped_mpi2_alpha,
        }
        return output_batch

    @staticmethod
    def compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2, pos_vectors: torch.Tensor):
        b, _, h, w = depth1.shape
        pos_vectors = pos_vectors.permute([0, 2, 3, 1])  # (b, h, w, 2)
        ones_2d = torch.ones(size=(b, h, w, 1)).to(depth1)  # (b, h, w, 1)
        pos_vectors_homo = torch.cat([pos_vectors, ones_2d], dim=3)[:, :, :, :, None]  # (b, h, w, 3, 1)
        ones_4d = torch.ones(size=(b, h, w, 1, 1)).to(depth1)  # (b, h, w, 1, 1)

        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)
        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        trans_norm_points = (trans_norm_points[:, :, :, :, 0]).permute([0, 3, 1, 2])  # (b, 3, h, w)
        return trans_norm_points

    @staticmethod
    def bilinear_splatting_frame2mpi(frame1, trans_depth2, trans_pos2, depth2_planes):
        b, c, h, w = frame1.shape
        _, d = depth2_planes.shape
        nearest_plane_index = (depth2_planes[:, None, None, None, :] - trans_depth2[:, :, :, :, None]).abs().argmin(dim=4)
        
        trans_pos_offset = trans_pos2 + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(trans_depth2, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, d, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, d, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0], nearest_plane_index),
                                frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], nearest_plane_index),
                                frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], nearest_plane_index),
                                frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], nearest_plane_index),
                                frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0], nearest_plane_index),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], nearest_plane_index),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], nearest_plane_index),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], nearest_plane_index),
                                  weight_se, accumulate=True)

        warped_frame_cf = warped_frame.permute([0, 4, 1, 2, 3])
        warped_weights_cf = warped_weights.permute([0, 4, 1, 2, 3])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        return warped_frame2, mask2

    def bilinear_interpolation_mpi(self, frame2: torch.Tensor, mask2: Optional[torch.Tensor], flow12: torch.Tensor,
                                   flow12_mask: Optional[torch.Tensor], normalize_by_alpha: bool = False,
                                   is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear interpolation
        :param frame2: (b, c, h, w, d)
        :param mask2: (b, 1, h, w, d): 1 for known, 0 for unknown. Optional
        :param flow12: (b, 2+ds, h, w, d)
        :param flow12_mask: (b, 1, h, w, d): 1 for valid flow, 0 for invalid flow. Optional
        :param normalize_by_alpha: If true, warped rgb and alpha are divided by warped alpha
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame1: (b, c, h, w, d)
                 mask1: (b, 1, h, w, d): 1 for known and 0 for unknown
        """
        b, c, h, w, d = frame2.shape
        if mask2 is None:
            mask2 = torch.ones(size=(b, 1, h, w, d)).to(frame2)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w, d)).to(flow12)
        grid = self.create_grid_mpi(b, h, w, d).to(frame2)
        trans_pos = flow12[:, :2] + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        weight_nw = torch.moveaxis(prox_weight_nw * flow12_mask, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_sw = torch.moveaxis(prox_weight_sw * flow12_mask, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_ne = torch.moveaxis(prox_weight_ne * flow12_mask, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_se = torch.moveaxis(prox_weight_se * flow12_mask, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])

        frame2_offset = F.pad(frame2, [1, 1, 1, 1, 1, 1])
        mask2_offset = F.pad(mask2, [1, 1, 1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None, None]

        warped_frame1_total, mask1_total = torch.zeros_like(frame2), torch.zeros_like(mask2)
        for ds in range(flow12.shape[1]-2):
            di = torch.ones((1, 1, 1, d), dtype=torch.long) * ds + 1

            f2_nw = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
            f2_sw = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
            f2_ne = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
            f2_se = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]

            m2_nw = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
            m2_sw = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
            m2_ne = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
            m2_se = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]

            nr = weight_nw * f2_nw * m2_nw + weight_sw * f2_sw * m2_sw + \
                 weight_ne * f2_ne * m2_ne + weight_se * f2_se * m2_se
            dr = weight_nw * m2_nw + weight_sw * m2_sw + weight_ne * m2_ne + weight_se * m2_se

            zero_tensor = torch.tensor(0, dtype=nr.dtype, device=nr.device)
            warped_frame1 = torch.where(dr > 0, nr / (dr + 1e-12), zero_tensor)
            mask1 = (dr > 0).to(frame2)

            # Convert to channel first
            warped_frame1 = torch.moveaxis(warped_frame1, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])
            mask1 = torch.moveaxis(mask1, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

            warped_frame1_total += flow12[:, ds+2:ds+3] * warped_frame1
            mask1_total += flow12[:, ds+2:ds+3] * mask1

        if normalize_by_alpha:
            warped_frame1 = warped_frame1_total / (mask1_total + 1e-3)
            mask1 = mask1_total / (mask1_total + 1e-3)
        else:
            warped_frame1 = warped_frame1_total
            mask1 = mask1_total

        if normalize_by_alpha and is_image:
            assert warped_frame1.min() >= -1.1  # Allow for rounding errors
            assert warped_frame1.max() <= 1.1
            warped_frame1 = torch.clamp(warped_frame1, min=-1, max=1)
        return warped_frame1, mask1

    def bilinear_interpolation_mpi_flow2d(self, input_batch: dict) -> dict:
        mpi_rgb = input_batch['mpi_rgb']
        mpi_alpha = input_batch['mpi_alpha']
        disoccluded_flow = input_batch['disoccluded_flow']

        b, c, h, w, d = mpi_rgb.shape
        grid = self.create_grid_mpi(b, h, w, d).to(mpi_rgb)

        mpi_rgb_offset = F.pad(mpi_rgb, [0, 0, 1, 1, 1, 1])
        mpi_alpha_offset = F.pad(mpi_alpha, [0, 0, 1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None, None]
        di = torch.arange(d)[None, None, None, :]

        trans_pos = disoccluded_flow + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        prox_weight_sum = prox_weight_nw + prox_weight_sw + prox_weight_ne + prox_weight_se
        prox_weight_nw = prox_weight_nw / prox_weight_sum
        prox_weight_sw = prox_weight_sw / prox_weight_sum
        prox_weight_ne = prox_weight_ne / prox_weight_sum
        prox_weight_se = prox_weight_se / prox_weight_sum

        weight_nw = torch.moveaxis(prox_weight_nw, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_sw = torch.moveaxis(prox_weight_sw, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_ne = torch.moveaxis(prox_weight_ne, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_se = torch.moveaxis(prox_weight_se, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])

        f2_nw = mpi_rgb_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
        f2_sw = mpi_rgb_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
        f2_ne = mpi_rgb_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
        f2_se = mpi_rgb_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]
        warped_rgb = weight_nw * f2_nw + weight_sw * f2_sw + weight_ne * f2_ne + weight_se * f2_se
        warped_rgb = torch.moveaxis(warped_rgb, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

        a2_nw = mpi_alpha_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
        a2_sw = mpi_alpha_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
        a2_ne = mpi_alpha_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
        a2_se = mpi_alpha_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]
        warped_alpha = weight_nw * a2_nw + weight_sw * a2_sw + weight_ne * a2_ne + weight_se * a2_se
        warped_alpha = torch.moveaxis(warped_alpha, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

        output_batch = {
            'warped_rgb': warped_rgb,
            'warped_alpha': warped_alpha,
        }
        return output_batch

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_tensor = torch.tensor(0, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def create_grid_mpi(b, h, w, d):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None, :, :, :, None].repeat([b, 1, 1, 1, d])
        return batch_grid
