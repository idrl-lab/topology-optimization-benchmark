# -*- encoding: utf-8 -*-
"""Load Response Dataset.
"""
import os

import scipy.io as sio
import numpy as np
from torchvision.datasets import VisionDataset


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="layout_input",
        resp_name="xData_output",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.extensions = extensions
        self.sample_files = make_dataset_list(root, list_path, extensions, is_valid_file)

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name)
        load = load.astype(float)
        resp = resp.astype(float)

        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return len(self.sample_files)


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp


if __name__ == "__main__":
    total_num = 50000
    with open('train'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8)):
            wf.write('Example'+str(idx)+'.mat'+'\n')
    with open('val'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8), total_num):
            wf.write('Example'+str(idx)+'.mat'+'\n')