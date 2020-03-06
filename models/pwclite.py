import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
from .correlation_package.correlation import Correlation
# from .correlation_native import Correlation

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class PWCLite(nn.Module):
    def __init__(self, cfg):
        super(PWCLite, self).__init__()
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = cfg.upsample
        self.n_frames = cfg.n_frames
        self.reduce_dense = cfg.reduce_dense

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1)

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(
            (self.flow_estimators.feat_dim + 2) * (self.n_frames - 1))

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward_2_frames(self, x1_pyramid, x2_pyramid):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(
                torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward_3_frames(self, x0_pyramid, x1_pyramid, x2_pyramid):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 4, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()

        for l, (x0, x1, x2) in enumerate(zip(x0_pyramid, x1_pyramid, x2_pyramid)):
            # warping
            if l == 0:
                x0_warp = x0
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                x0_warp = flow_warp(x0, flow[:, :2])
                x2_warp = flow_warp(x2, flow[:, 2:])

            # correlation
            corr_10, corr_12 = self.corr(x1, x0_warp), self.corr(x1, x2_warp)
            corr_relu_10, corr_relu_12 = self.leakyRELU(corr_10), self.leakyRELU(corr_12)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            feat_10 = [x1_1by1, corr_relu_10, corr_relu_12, flow[:, :2], -flow[:, 2:]]
            feat_12 = [x1_1by1, corr_relu_12, corr_relu_10, flow[:, 2:], -flow[:, :2]]
            x_intm_10, flow_res_10 = self.flow_estimators(torch.cat(feat_10, dim=1))
            x_intm_12, flow_res_12 = self.flow_estimators(torch.cat(feat_12, dim=1))
            flow_res = torch.cat([flow_res_10, flow_res_12], dim=1)
            flow = flow + flow_res

            feat_10 = [x_intm_10, x_intm_12, flow[:, :2], -flow[:, 2:]]
            feat_12 = [x_intm_12, x_intm_10, flow[:, 2:], -flow[:, :2]]
            flow_res_10 = self.context_networks(torch.cat(feat_10, dim=1))
            flow_res_12 = self.context_networks(torch.cat(feat_12, dim=1))
            flow_res = torch.cat([flow_res_10, flow_res_12], dim=1)
            flow = flow + flow_res

            flows.append(flow)

            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]

        flows_10 = [flo[:, :2] for flo in flows[::-1]]
        flows_12 = [flo[:, 2:] for flo in flows[::-1]]
        return flows_10, flows_12

    def forward(self, x, with_bk=False):
        n_frames = x.size(1) / 3

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        res_dict = {}
        if n_frames == 2:
            res_dict['flows_fw'] = self.forward_2_frames(x[0], x[1])
            if with_bk:
                res_dict['flows_bw'] = self.forward_2_frames(x[1], x[0])
        elif n_frames == 3:
            flows_10, flows_12 = self.forward_3_frames(x[0], x[1], x[2])
            res_dict['flows_fw'], res_dict['flows_bw'] = flows_12, flows_10
        elif n_frames == 5:
            flows_10, flows_12 = self.forward_3_frames(x[0], x[1], x[2])
            flows_21, flows_23 = self.forward_3_frames(x[1], x[2], x[3])
            res_dict['flows_fw'] = [flows_12, flows_23]
            if with_bk:
                flows_32, flows_34 = self.forward_3_frames(x[2], x[3], x[4])
                res_dict['flows_bw'] = [flows_21, flows_32]
        else:
            raise NotImplementedError
        return res_dict

