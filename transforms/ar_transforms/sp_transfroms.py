# Part of the code from https://github.com/visinf/irr/blob/master/augmentations.py

import torch
import torch.nn as nn
from transforms.ar_transforms.interpolation import Interp2
from transforms.ar_transforms.interpolation import Meshgrid
import numpy as np


def denormalize_coords(xx, yy, width, height):
    """ scale indices from [-1, 1] to [0, width/height] """
    xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
    yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
    return xx, yy


def normalize_coords(xx, yy, width, height):
    """ scale indices from [0, width/height] to [-1, 1] """
    xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
    yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
    return xx, yy


def apply_transform_to_params(theta0, theta_transform):
    a1 = theta0[:, 0]
    a2 = theta0[:, 1]
    a3 = theta0[:, 2]
    a4 = theta0[:, 3]
    a5 = theta0[:, 4]
    a6 = theta0[:, 5]
    #
    b1 = theta_transform[:, 0]
    b2 = theta_transform[:, 1]
    b3 = theta_transform[:, 2]
    b4 = theta_transform[:, 3]
    b5 = theta_transform[:, 4]
    b6 = theta_transform[:, 5]
    #
    c1 = a1 * b1 + a4 * b2
    c2 = a2 * b1 + a5 * b2
    c3 = b3 + a3 * b1 + a6 * b2
    c4 = a1 * b4 + a4 * b5
    c5 = a2 * b4 + a5 * b5
    c6 = b6 + a3 * b4 + a6 * b5
    #
    new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
    return new_theta


class _IdentityParams(nn.Module):
    def __init__(self):
        super(_IdentityParams, self).__init__()
        self._batch_size = 0
        self.register_buffer("_o", torch.FloatTensor())
        self.register_buffer("_i", torch.FloatTensor())

    def _update(self, batch_size):
        torch.zeros([batch_size, 1], out=self._o)
        torch.ones([batch_size, 1], out=self._i)
        return torch.cat([self._i, self._o, self._o, self._o, self._i, self._o], dim=1)

    def forward(self, batch_size):
        if self._batch_size != batch_size:
            self._identity_params = self._update(batch_size)
            self._batch_size = batch_size
        return self._identity_params


class RandomMirror(nn.Module):
    def __init__(self, vertical=True, p=0.5):
        super(RandomMirror, self).__init__()
        self._batch_size = 0
        self._p = p
        self._vertical = vertical
        self.register_buffer("_mirror_probs", torch.FloatTensor())

    def update_probs(self, batch_size):
        torch.ones([batch_size, 1], out=self._mirror_probs)
        self._mirror_probs *= self._p

    def forward(self, theta_list):
        batch_size = theta_list[0].size(0)
        if batch_size != self._batch_size:
            self.update_probs(batch_size)
            self._batch_size = batch_size

        # apply random sign to a1 a2 a3 (these are the guys responsible for x)
        sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
        i = torch.ones_like(sign)
        horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
        theta_list = [theta * horizontal_mirror for theta in theta_list]

        # apply random sign to a4 a5 a6 (these are the guys responsible for y)
        if self._vertical:
            sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
            vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
            theta_list = [theta * vertical_mirror for theta in theta_list]

        return theta_list


class RandomAffineFlow(nn.Module):
    def __init__(self, cfg, addnoise=True):
        super(RandomAffineFlow, self).__init__()
        self.cfg = cfg
        self._interp2 = Interp2(clamp=False)
        self._flow_interp2 = Interp2(clamp=False)
        self._meshgrid = Meshgrid()
        self._identity = _IdentityParams()
        self._random_mirror = RandomMirror(cfg.vflip) if cfg.hflip else RandomMirror(p=1)
        self._addnoise = addnoise

        self.register_buffer("_noise1", torch.FloatTensor())
        self.register_buffer("_noise2", torch.FloatTensor())
        self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
        self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
        self.register_buffer("_x", torch.IntTensor(1))
        self.register_buffer("_y", torch.IntTensor(1))

    def inverse_transform_coords(self, width, height, thetas, offset_x=None,
                                 offset_y=None):
        xx, yy = self._meshgrid(width=width, height=height)

        xx = torch.unsqueeze(xx, dim=0).float()
        yy = torch.unsqueeze(yy, dim=0).float()

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
        a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
        a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
        a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
        a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
        a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self._meshgrid(width=width, height=height)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)

        def _unsqueeze12(u):
            return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

        a1 = _unsqueeze12(thetas[:, 0])
        a2 = _unsqueeze12(thetas[:, 1])
        a3 = _unsqueeze12(thetas[:, 2])
        a4 = _unsqueeze12(thetas[:, 3])
        a5 = _unsqueeze12(thetas[:, 4])
        a6 = _unsqueeze12(thetas[:, 5])
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = xx - a3
        yhat = yy - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat

        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        x = self._xbounds
        y = self._ybounds
        #
        a1 = torch.unsqueeze(thetas[:, 0], dim=1)
        a2 = torch.unsqueeze(thetas[:, 1], dim=1)
        a3 = torch.unsqueeze(thetas[:, 2], dim=1)
        a4 = torch.unsqueeze(thetas[:, 3], dim=1)
        a5 = torch.unsqueeze(thetas[:, 4], dim=1)
        a6 = torch.unsqueeze(thetas[:, 5], dim=1)
        #
        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        #
        invalid = (
                          (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                  ).sum(dim=1, keepdim=True) > 0

        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new(batch_size, 1).zero_()
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom).byte()

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid.float() * theta_try + (1 - invalid).float() * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas)

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self._interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2):
        batch_size, channels, height, width = flow.size()
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.stack([u, v], dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        # interp2
        transformed = self._flow_interp2(new_flow, xq, yq)
        return transformed

    def forward(self, data):
        # 01234 flow 12 21 23 32
        imgs = data['imgs']
        flows_f = data['flows_f']
        masks_f = data['masks_f']

        batch_size, _, height, width = imgs[0].size()

        # identity = no transform
        theta0 = self._identity(batch_size)

        # global transform
        theta_list = [self.apply_random_transforms_to_params(
            theta0,
            max_translate=self.cfg.trans[0],
            min_zoom=self.cfg.zoom[0], max_zoom=self.cfg.zoom[1],
            min_squeeze=self.cfg.squeeze[0], max_squeeze=self.cfg.squeeze[1],
            min_rotate=self.cfg.rotate[0], max_rotate=self.cfg.rotate[1],
            validate_size=[height, width])
        ]

        # relative transform
        for i in range(len(imgs) - 1):
            theta_list.append(
                self.apply_random_transforms_to_params(
                    theta_list[-1],
                    max_translate=self.cfg.trans[1],
                    min_zoom=self.cfg.zoom[2], max_zoom=self.cfg.zoom[3],
                    min_squeeze=self.cfg.squeeze[2], max_squeeze=self.cfg.squeeze[3],
                    min_rotate=self.cfg.rotate[2], max_rotate=self.cfg.rotate[2],
                    validate_size=[height, width])
            )

        # random flip images
        theta_list = self._random_mirror(theta_list)

        # 01234
        imgs = [self.transform_image(im, theta) for im, theta in zip(imgs, theta_list)]

        if len(imgs) > 2:
            theta_list = theta_list[1:-1]
        # 12 23
        flows_f = [self.transform_flow(flo, theta1, theta2) for flo, theta1, theta2 in
                   zip(flows_f, theta_list[:-1], theta_list[1:])]

        masks_f = [self.transform_image(mask, theta) for mask, theta in
                   zip(masks_f, theta_list)]

        if self._addnoise:
            stddev = np.random.uniform(0.0, 0.04)
            for im in imgs:
                noise = torch.zeros_like(im)
                noise.normal_(std=stddev)
                im.add_(noise)
                im.clamp_(0.0, 1.0)

        data['imgs'] = imgs
        data['flows_f'] = flows_f
        data['masks_f'] = masks_f
        return data
