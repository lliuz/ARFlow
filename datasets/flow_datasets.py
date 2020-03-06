import imageio
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = [imageio.imread(self.root / p).astype(np.float32) for p in images]

        target = {}
        if 'flow' in s:
            target['flow'] = load_flow(self.root / s['flow'])
        if 'mask' in s:
            # 0~255 HxWx1
            mask = imageio.imread(self.root / s['mask']).astype(np.float32) / 255.
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])

        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, _ = self.co_transform(images, {})
        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
        data['target'] = target
        return data


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, transform=None, co_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, input_transform=transform,
                                        co_transform=co_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        samples = []
        for scene in scene_list:
            img_list = scene.files('*.png')
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                samples.append(sample)
        return samples


class Sintel(ImgSeqDataset):
    def __init__(self, root, n_frames=2, type='clean', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.dataset_type = type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(Sintel, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'

        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for flow_map in sorted((self.root / flow_dir).glob('*/*.flo')):
            info = flow_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue

            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

                if self.with_flow:
                    if self.n_frames == 3:
                        # for img1 img2 img3, only flow_23 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                    elif self.n_frames == 2:
                        # for img1 img2, flow_12 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                    else:
                        raise NotImplementedError(
                            'n_frames {} with flow or mask'.format(self.n_frames))

                    if self.with_flow:
                        assert s['flow'].isfile()
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples
