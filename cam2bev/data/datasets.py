import os
import glob
import typing as t

import numpy as np
import torch
from PIL import Image

import torch.utils.data as td
import torch.nn.functional as functional

from . import types, utils


class MultiFolderImages(td.Dataset):
    MAX_CACHE_SIZE = 1000

    def __init__(
            self,
            root: str,
            folders: t.Sequence[str],
            target_size: t.Optional[t.Tuple[int, int]] = None,
            extensions: t.Sequence[str] = ('png', 'jpg', 'jpeg'),
    ):
        self.root = root
        self.folders = folders

        self.target_size = target_size

        trial = self.folders[0]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(root, trial, f'*.{ext}')))

        self.index = [os.path.relpath(path, os.path.join(root, trial)) for path in paths]

    def __len__(self) -> int:
        return len(self.index)

    @utils.method_lru_cache(maxsize=MAX_CACHE_SIZE)
    def __getitem__(self, item) -> t.Dict[str, Image.Image]:
        chosen = self.index[item]

        if self.target_size is None:
            return {
                folder: Image.open(os.path.join(self.root, folder, chosen))
                for folder in self.folders
            }

        else:
            return {
                folder: Image.open(
                    os.path.join(self.root, folder, chosen)
                ).resize(self.target_size)
                for folder in self.folders
            }

    @property
    def spatial_size(self) -> t.Optional[t.Tuple[int, int]]:
        return self.target_size


class MultiFolderMasks(td.Dataset, types.WithIOInfo):
    MAX_CACHE_SIZE = 1000

    def __init__(
            self,
            root: str,
            folders: t.Sequence[str],
            conversion_map: t.Dict[str, t.Any],
            target_size: t.Tuple[int, int] = (400, 400),
            extensions: t.Sequence[str] = ('png', 'jpg', 'jpeg')
    ):
        super(MultiFolderMasks, self).__init__()

        self._img_dataset = MultiFolderImages(
            root=root,
            folders=folders,
            extensions=extensions,
        )
        self._conv_map = {tuple(v['color']): v['to'] for v in conversion_map.values()}

        self._seg_keys = sorted(set(self._conv_map.values()))
        self._n_classes = len(self._seg_keys)

        self.target_size = target_size

        assert all(
            map(lambda v: v[0] + 1 == v[1], zip(self._seg_keys, self._seg_keys[1:]))
        )

    @utils.method_lru_cache(maxsize=MAX_CACHE_SIZE)
    def __getitem__(self, item) -> t.Dict[str, torch.Tensor]:
        images = self._img_dataset[item]

        class_images = {}
        for k, image in images.items():
            arr = np.array(image)

            masks = {}
            for value, class_id in self._conv_map.items():
                value_arr = np.array(value)[np.newaxis, np.newaxis]
                mask = (arr == value_arr).all(axis=2)

                if class_id in masks:
                    masks[class_id] = masks[class_id] | mask[..., np.newaxis]
                else:
                    masks[class_id] = mask[..., np.newaxis]

            to_cat = []
            for key in self._seg_keys:
                to_cat.append(masks[key])

            final_np = np.concatenate(to_cat, axis=2).astype('float32')
            final_pt: torch.Tensor = torch.from_numpy(final_np).permute(2, 0, 1)
            interp_pt = functional.interpolate(
                final_pt.unsqueeze(0),
                size=self.spatial_size,
            )
            class_images[k] = interp_pt[0]

        return class_images

    def __len__(self) -> int:
        return len(self._img_dataset)

    @property
    def num_in_channels(self) -> int:
        return self._n_classes

    @property
    def num_out_channels(self) -> int:
        return self._n_classes

    @property
    def spatial_size(self) -> t.Tuple[int, int]:
        return self.target_size


def test_images_usage():
    cwd = os.path.dirname(__file__)
    root = os.path.join(cwd, '../../samples')
    folders = ['bev', 'left', 'right']

    dataset = MultiFolderImages(root, folders)

    sample = dataset[0]
    for k, im in sample.items():
        im.show()
