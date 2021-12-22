import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class Filelist(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix,
                        'img_info': {'filename': filename},
                        'gt_label': np.array(gt_label, dtype=np.int64)
                        }
                data_infos.append(info)
            return data_infos


@DATASETS.register_module()
class FaceAttr(MultiLabelDataset):
    CLASSES = ('female', 'male',
               'front', 'side',
               'clean', 'occlusion',
               'super_hq', 'hq', 'blur',
               'nonhuman')

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            # print(samples)
            # try:
            for filename, gt_label in samples:
                # print(filename, gt_label)
                info = {'img_prefix': self.data_prefix,
                        'img_info': {'filename': filename},
                        'gt_label': np.array(list(gt_label), dtype=np.int64)
                        }
                # print(info['gt_label'])
                data_infos.append(info)

            return data_infos
