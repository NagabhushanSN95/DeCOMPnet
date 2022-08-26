# Shree KRISHNAya Namaha
# PWCnet after feature extraction
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import torch
import torch.nn.functional as F

from flow_estimation.libraries.partial_convolution.partialconv3d import PartialConv3d
from utils.WarperPytorch import Warper


def verify_mask(mask, useless_multichannel: bool = True):
    if useless_multichannel:
        # multi-channel mask used for the sake of partial convolution only. Mask value remains the same across all
        # channels.
        assert torch.std(mask, dim=1, unbiased=False).max() == 0
        assert torch.std(mask, dim=1, unbiased=False).min() == 0
    assert mask.unique().numel() <= 2
    assert mask.min() >= 0
    assert mask.max() <= 1
    return


def verify_flow(flow):
    assert flow[:, 2:].min() >= 0 - 1e-3
    assert flow[:, 2:].max() <= 1 + 1e-3
    assert torch.sum(flow[:, 2:], dim=1).max() <= 1 + 1e-3
    return


def mask_or(m1, m2):
    m1_or_m2 = torch.clamp(m1 + m2, min=0, max=1)

    # All below checks verified
    # verify_mask(m1_or_m2)
    return m1_or_m2


class FlowEstimator(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.search_range = self.configs['flow_estimation']['flow_estimator']['search_range_spatial']
        self.output_level = 4
        self.num_levels = 7
        self.upsample_flow = configs['flow_estimation']['flow_estimator']['upsample_flow']

        if 'data_loader' in self.configs:
            self.num_mpi_planes = self.configs['data_loader']['num_mpi_planes']
        elif 'data_loader' in self.configs['flow_estimation']:
            self.num_mpi_planes = self.configs['flow_estimation']['data_loader']['num_mpi_planes']
        else:
            raise RuntimeError
        self.spatial_patch_size = self.configs['flow_estimation']['flow_estimator']['search_range_spatial'] * 2 + 1
        self.depth_patch_size = self.configs['flow_estimation']['flow_estimator']['search_range_depth'] * 2 + 1
        corr_dim = (self.spatial_patch_size ** 2) * self.depth_patch_size
        crude_flow_input_dim = 32 + corr_dim + 2 + self.depth_patch_size
        fine_flow_input_dim = 32 + 2

        self.warper = Warper(configs['device'])
        self.correlation = CorrelationLayer(self.configs)
        self.leakyRELU = torch.nn.LeakyReLU(0.1)
        self.feature_aggregator = FeatureAggregator()
        self.crude_flow_estimator = CrudeFlowEstimator(configs, crude_flow_input_dim)
        # self.fine_flow_estimator = FineFlowEstimator(configs, fine_flow_input_dim)
        return

    def forward(self, input_batch):
        flows, flow_masks = [], []
        x1_pyramid = input_batch['mpi1_features'][::-1]
        x2_pyramid = input_batch['mpi2_features'][::-1]

        b, _, h, w, d = x1_pyramid[0][0].shape
        flow = torch.zeros(size=(b, 2 + self.num_mpi_planes, h, w, d)).to(x1_pyramid[0][0])
        m_flow = torch.ones(size=(b, 2 + self.num_mpi_planes, h, w, d)).to(x1_pyramid[0][0])
        for k in range(self.num_mpi_planes):
            flow[:, 2 + k, :, :, k] = 1

        for i, ((x1, m1), (x2, m2)) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # All below checks verified
            # verify_mask(m1)
            # verify_mask(m2)

            # warping
            if i == 0:
                x2_warp = x2
                m2_warp = m2
                m_flow = torch.broadcast_to(m1[:, :1], m_flow.shape)
            else:
                # mask-aware upsampling
                # flow_bkp, m_flow_bkp = flow, m_flow
                flow = torch.cat([F.interpolate(flow[:, :2] * 2, x1.shape[2:], mode='trilinear', align_corners=True),
                                  F.interpolate(flow[:, 2:] * 1, x1.shape[2:], mode='trilinear', align_corners=True)],
                                 dim=1)
                # verify_flow(flow)  # Verified
                m_flow = F.interpolate(m_flow, m1.shape[2:], mode='trilinear', align_corners=True)
                # m_flow_bkp2 = m_flow
                flow = flow / (m_flow + 1e-6)
                # verify_flow(flow)  # verified
                m_flow = (m_flow >= 0.001).float()
                flow = flow * m_flow  # TODO: check if this is required
                # All below checks verified
                # verify_flow(flow)
                # assert ((flow > 0.1).float() * (1 - m_flow)).max() == 0
                # assert ((flow > 0.1).float() * (1 - m_flow)).min() == 0

                # Wherever m1 is False, we are not interested in such flow
                m_flow = m_flow * m1[:, :1]

                m_flow1 = torch.mean(m_flow, dim=1, keepdim=True)
                x2_warp, m2_warp = self.warper.bilinear_interpolation_mpi(x2, m2, flow, m_flow1, is_image=False)
                m2_warp = (m2_warp >= 0.05).float()
                del m_flow1
                
                # All below checks verified
                # verify_mask(m_flow)
                # verify_mask(m_flow1)
                # verify_mask(m2_warp)

            # correlation
            correlation_inputs = {
                'mpi1_features': x1,
                'mpi2_features': x2_warp,
                'mpi1_alpha': m1,
                'mpi2_alpha': m2_warp,
            }
            corr_outputs = self.correlation(correlation_inputs)
            out_corr, m_corr = corr_outputs['cost_volume'], corr_outputs['cost_volume_mask']
            out_corr_relu = self.leakyRELU(out_corr)
            # Although correlation is valid even in alpha=0 regions, we are not interested in estimating flow in such
            # regions. Hence, set them to 0.
            m_corr = m_corr * m1[:, :1]
            if not self.training:
                del x2, m2, x2_warp, m2_warp

            # All below checks verified
            # verify_mask(m_corr, useless_multichannel=False)

            # concat and estimate flow
            x1_1by1, m1_1by1 = self.feature_aggregator(x1, m1, i)
            flow1, m_flow1 = self.convert_all_plane_flow_to_nbr_plane_flow(flow, m_flow, self.depth_patch_size)
            crude_flow_inputs = {
                'mpi_features': torch.cat([out_corr_relu, x1_1by1, flow1], dim=1),
                'mpi_mask': torch.cat([m_corr, m1_1by1, m_flow1], dim=1),
            }
            if not self.training:
                del x1, m1, x1_1by1, m1_1by1, out_corr, out_corr_relu, m_corr, flow1, m_flow1
            crude_flow_outputs = self.crude_flow_estimator(crude_flow_inputs)
            # x_intm, m_intm = crude_flow_outputs['mpi_feature_intermediate'], crude_flow_outputs['mpi_feature_intermediate_mask']
            flow_res, m_flow_res = crude_flow_outputs['mpi_flow'], crude_flow_outputs['mpi_flow_mask']
            # Wherever flow is invalid, set all flow values to 0, including softmax
            flow_res = flow_res * m_flow_res
            flow, m_flow = self.add_flow(flow, m_flow, flow_res, m_flow_res)
            # Wherever flow is invalid, set all flow values to 0, including softmax
            flow = flow * m_flow

            if not self.training:
                del flow_res, m_flow_res

            # All below checks verified
            # verify_mask(m_flow1)
            # verify_mask(m_intm)
            # verify_mask(m_flow_res)
            # verify_mask(m_flow)
            # assert torch.equal(m1[:, 0], m_flow[:, 0])
            # assert ((flow > 0.1).float() * (1 - m_flow)).max() == 0
            # assert ((flow > 0.1).float() * (1 - m_flow)).min() == 0
            # verify_flow(flow_res)
            # verify_flow(flow)

            # fine_flow_inputs = {
            #     'mpi_features': torch.cat([x_intm, flow], dim=1),
            #     'mpi_mask': torch.cat([m_intm, m_flow], dim=1),
            # }
            # fine_flow_outputs = self.fine_flow_estimator(fine_flow_inputs)
            # flow_fine, m_flow_fine = fine_flow_outputs['mpi_flow'], fine_flow_outputs['mpi_flow_mask']
            # flow = flow + flow_fine
            # m_flow = mask_or(m_flow, m_flow_fine)
            #
            # TO DO: Comment after verifying
            # verify_mask(m_flow_fine)
            # verify_mask(m_flow)

            m_flow1 = torch.mean(m_flow, dim=1, keepdim=True)
            flows.append(flow)
            flow_masks.append(m_flow1)

            if not self.training:
                if len(flows) >= 2:
                    del flows[0], flow_masks[0]

            # upsampling or post-processing
            if i == self.output_level:
                break

        if self.upsample_flow:
            upsampled_flows, upsampled_flow_masks = [], []
            for flow, flow_mask in zip(flows, flow_masks):
                # Mask aware upsampling
                upsampled_flow_shape = [flow.shape[2]*4, flow.shape[3]*4, flow.shape[4]]
                upsampled_flow = torch.cat([F.interpolate(flow[:, :2] * 4, upsampled_flow_shape, mode='trilinear', align_corners=True),
                                            F.interpolate(flow[:, 2:] * 1, upsampled_flow_shape, mode='trilinear', align_corners=True)],
                                           dim=1)
                # verify_flow(flow)  # Verified
                upsampled_flow_mask = F.interpolate(flow_mask, upsampled_flow_shape, mode='trilinear', align_corners=True)
                upsampled_flow = upsampled_flow / (upsampled_flow_mask + 1e-6)
                upsampled_flow_mask = (upsampled_flow_mask >= 0.001).float()
                upsampled_flows.append(upsampled_flow)
                upsampled_flow_masks.append(upsampled_flow_mask)
                
                # All below checks verified
                # assert upsampled_flow_mask.max() <= 1
                # assert upsampled_flow_mask.min() >= 0
            flows = upsampled_flows
            flow_masks = upsampled_flow_masks
            # flow_masks = upsampled_flow_masks
        flows = list(zip(flows, flow_masks))
        flows = flows[::-1]

        result_dict = {
            'estimated_mpi_flows12': flows,
        }
        return result_dict

    def add_flow(self, flow: torch.Tensor, m_flow: torch.Tensor, flow_res: torch.Tensor, m_flow_res: torch.Tensor):
        old_flow_xy = flow[:, :2]
        old_flow_z = flow[:, 2:]
        m_old_flow = m_flow[:, :1]
        new_flow_xy = flow_res[:, :2]
        new_flow_z = flow_res[:, 2:]
        m_new_flow = m_flow_res[:, :1]
        total_flow_xy = old_flow_xy + new_flow_xy

        unity_flow_z = torch.zeros_like(old_flow_z)
        for k in range(self.num_mpi_planes):
            unity_flow_z[:, k, :, :, k] = 1
        new_flow_xy1 = torch.cat([new_flow_xy, unity_flow_z], dim=1)
        warped_old_flow_z = self.warper.bilinear_interpolation_mpi(old_flow_z, m_old_flow, new_flow_xy1, m_new_flow)[0]

        total_flow_z = torch.zeros_like(old_flow_z)
        pad_val = self.configs['flow_estimation']['flow_estimator']['search_range_depth']
        padded_old_flow_z = F.pad(warped_old_flow_z, [pad_val, pad_val])
        for k in range(self.num_mpi_planes):
            for k1 in range(self.depth_patch_size):
                total_flow_z[:, :, :, :, k] += padded_old_flow_z[:, :, :, :, k + k1] * new_flow_z[:, k1: k1+1, :, :, k]
        total_flow = torch.cat([total_flow_xy, total_flow_z], dim=1)
        m_total_flow = torch.broadcast_to(m_new_flow, m_flow.shape)
        return total_flow, m_total_flow

    @staticmethod
    def convert_all_plane_flow_to_nbr_plane_flow(flow: torch.Tensor, m_flow, depth_patch_size):
        pad_length = depth_patch_size // 2
        flow_xy = flow[:, :2]
        flow_z = flow[:, 2:]
        m_flow_xy = m_flow[:, :2]
        m_flow_z = m_flow[:, 2:]
        padded_flow = F.pad(flow_z, [0, 0, 0, 0, 0, 0, pad_length, pad_length])
        mask_pads = m_flow_z[:, :pad_length]
        padded_m_flow = torch.cat([mask_pads, m_flow_z, mask_pads], dim=1)
        nbr_plane_flows = []
        nbr_plane_m_flows = []
        for i in range(flow.shape[4]):
            nbr_plane_flows.append(padded_flow[:, i:i+depth_patch_size, :, :, i])
            nbr_plane_m_flows.append(padded_m_flow[:, i:i+depth_patch_size, :, :, i])
        nbr_plane_flows = torch.stack(nbr_plane_flows, dim=4)
        nbr_plane_m_flows = torch.stack(nbr_plane_m_flows, dim=4)
        nbr_plane_flow = torch.cat([flow_xy, nbr_plane_flows], dim=1)
        nbr_plane_m_flow = torch.cat([m_flow_xy, nbr_plane_m_flows], dim=1)
        return nbr_plane_flow, nbr_plane_m_flow


class CorrelationLayer(torch.nn.Module):
    def __init__(self, configs):
        super(CorrelationLayer, self).__init__()
        self.configs = configs
        self.max_displacement_spatial = self.configs['flow_estimation']['flow_estimator']['search_range_spatial']  # 4
        self.max_displacement_depth = self.configs['flow_estimation']['flow_estimator']['search_range_depth']  # 1
        self.spatial_output_dim = 2 * self.max_displacement_spatial + 1
        self.depth_output_dim = 2 * self.max_displacement_depth + 1
        return

    def forward(self, input_batch):
        mpi1_features = input_batch['mpi1_features']
        mpi1_alpha = input_batch['mpi1_alpha']
        mpi2_features = input_batch['mpi2_features']
        mpi2_alpha = input_batch['mpi2_alpha']
        b, c, h, w, d = mpi1_features.shape

        mask1 = torch.sum(mpi1_alpha, dim=-1, keepdim=True)
        mask2 = torch.sum(mpi2_alpha, dim=-1, keepdim=True)
        mask1 = torch.clip(mask1, min=0, max=1)  # Downsampled MPI alpha at a location can have 1 at multiple planes
        mask2 = torch.clip(mask2, min=0, max=1)
        mask1 = torch.broadcast_to(mask1, mpi1_alpha.shape)
        mask2 = torch.broadcast_to(mask2, mpi1_alpha.shape)

        # All below checks verified
        # verify_mask(mask1)
        # verify_mask(mask2)

        padding = [self.max_displacement_depth] * 2 + [self.max_displacement_spatial] * 4
        mpi2_features = F.pad(mpi2_features, padding, mode='constant', value=0)
        mask2 = F.pad(mask2, padding, mode='constant', value=1)  # padded regions have valid correlation=0

        cv = []
        cvm = []
        for k in range(self.depth_output_dim):
            for i in range(self.spatial_output_dim):
                for j in range(self.spatial_output_dim):
                    cost = mpi1_features * mpi2_features[:, :, i:(i + h), j:(j + w), k:(k + d)]
                    cost_mask = mask1 * mask2[:, :, i:(i + h), j:(j + w), k:(k + d)]
                    cost = torch.mean(cost, dim=1, keepdim=True)
                    cost_mask = torch.clip(torch.sum(cost_mask, dim=1, keepdim=True), min=0, max=1)
                    cv.append(cost)
                    cvm.append(cost_mask)
        cost_volume = torch.cat(cv, 1)
        cost_volume_mask = torch.cat(cvm, 1)

        output_batch = {
            'cost_volume': cost_volume,
            'cost_volume_mask': cost_volume_mask,
        }
        return output_batch


class FeatureAggregator(torch.nn.Module):
    """
    A 1x1 convolution layer to reduce feature dimension to 32, at every level
    """
    def __init__(self):
        super().__init__()
        self.conv1 = PartialConv3d(in_channels=192, out_channels=32, kernel_size=1, stride=1, padding=0, return_mask=True, multi_channel=True)
        self.conv2 = PartialConv3d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, return_mask=True, multi_channel=True)
        self.conv3 = PartialConv3d(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0, return_mask=True, multi_channel=True)
        self.conv4 = PartialConv3d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, return_mask=True, multi_channel=True)
        self.conv5 = PartialConv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, return_mask=True, multi_channel=True)
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    def forward(self, features, mask, level):
        agg_features, agg_features_mask = self.conv_layers[level](features, mask)
        agg_features = self.leaky_relu(agg_features)
        return agg_features, agg_features_mask


class CrudeFlowEstimator(torch.nn.Module):
    """
    FlowEstimatorReduce in ARFlow code
    """
    def __init__(self, configs: dict, num_input_channels: int):
        super().__init__()
        self.configs = configs
        self.num_output_channels = 2 + 2 * self.configs['flow_estimation']['flow_estimator']['search_range_depth'] + 1
        self.conv1 = PartialConv3d(in_channels=num_input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv2 = PartialConv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv3 = PartialConv3d(in_channels=128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv4 = PartialConv3d(in_channels=128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv5 = PartialConv3d(in_channels=96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv6 = PartialConv3d(in_channels=64 + 32, out_channels=self.num_output_channels, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    def forward(self, input_batch):
        features = input_batch['mpi_features']
        mask = input_batch['mpi_mask']
        x1, m1 = self.conv1(features, mask)
        x1 = self.leaky_relu(x1)
        if not self.training:
            del features, mask
        
        x2, m2 = self.conv2(x1, m1)
        x2 = self.leaky_relu(x2)
        
        x3 = torch.cat([x1, x2], dim=1)
        m3 = torch.cat([m1, m2], dim=1)
        if not self.training:
            del x1, m1
        x3, m3 = self.conv3(x3, m3)
        x3 = self.leaky_relu(x3)
        
        x4 = torch.cat([x2, x3], dim=1)
        m4 = torch.cat([m2, m3], dim=1)
        if not self.training:
            del x2, m2
        x4, m4 = self.conv4(x4, m4)
        x4 = self.leaky_relu(x4)

        x5 = torch.cat([x3, x4], dim=1)
        m5 = torch.cat([m3, m4], dim=1)
        if not self.training:
            del x3, m3
        x5, m5 = self.conv5(x5, m5)
        x5 = self.leaky_relu(x5)

        x6 = torch.cat([x4, x5], dim=1)
        m6 = torch.cat([m4, m5], dim=1)
        if not self.training:
            del x4, m4
        flow, flow_mask = self.conv6(x6, m6)
        flow = torch.cat([flow[:, :2], torch.softmax(flow[:, 2:], dim=1)], dim=1)
        output_batch = {
            # 'mpi_feature_intermediate': x5,
            # 'mpi_feature_intermediate_mask': m5,
            'mpi_flow': flow,
            'mpi_flow_mask': flow_mask,
        }
        return output_batch


class FineFlowEstimator(torch.nn.Module):
    """
    ContextNetwork in ARFlow code
    """
    def __init__(self, num_input_channels: int):
        super().__init__()
        self.conv1 = PartialConv3d(in_channels=num_input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, return_mask=True, multi_channel=True)
        self.conv2 = PartialConv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, return_mask=True, multi_channel=True)
        self.conv3 = PartialConv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, return_mask=True, multi_channel=True)
        self.conv4 = PartialConv3d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8, return_mask=True, multi_channel=True)
        self.conv5 = PartialConv3d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16, return_mask=True, multi_channel=True)
        self.conv6 = PartialConv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, return_mask=True, multi_channel=True)
        self.conv7 = PartialConv3d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, return_mask=True, multi_channel=True)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    def forward(self, input_batch):
        features = input_batch['mpi_features']
        mask = input_batch['mpi_mask']
        x1, m1 = self.conv1(features, mask)
        x1 = self.leaky_relu(x1)

        x2, m2 = self.conv2(x1, m1)
        x2 = self.leaky_relu(x2)

        x3, m3 = self.conv3(x2, m2)
        x3 = self.leaky_relu(x3)

        x4, m4 = self.conv4(x3, m3)
        x4 = self.leaky_relu(x4)

        x5, m5 = self.conv5(x4, m4)
        x5 = self.leaky_relu(x5)

        x6, m6 = self.conv6(x5, m5)
        x6 = self.leaky_relu(x6)

        flow, flow_mask = self.conv7(x6, m6)
        output_batch = {
            'mpi_flow': flow,
            'mpi_flow_mask': flow_mask,
        }
        return output_batch
