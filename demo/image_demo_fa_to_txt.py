# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
from mmcls.apis import inference_model, init_model, show_result_pyplot
import shutil
from tqdm import tqdm
import cv2
import uuid
from cv2box import CVFile
from cv2box.utils import get_path_by_ext


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', default='', help='Image file')
    parser.add_argument('--config', default='local_config/face_attr_1221.py', help='Config file')
    parser.add_argument('--checkpoint', default='', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ('female', 'male',
                     'front', 'side',
                     'clean', 'occlusion',
                     'super_hq', 'hq', 'blur',
                     'nonhuman')
    with open('', 'a') as f1:
        for img_p in tqdm(get_path_by_ext(args.img_dir)):
            # img_name = img_p.stem
            # json_path = img_p.parent / (img_name + '.json')
            img_path_full = str(img_p)[64:]
            # print(img_path_full)
            result = inference_model(model, str(img_p), multi_label=True)
            # print('1')
            write_temp = '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}'.format(img_path_full, str(result['female']),
                                                                               str(result['male']),
                                                                               str(result['front']),
                                                                               str(result['side']),
                                                                               str(result['clean']),
                                                                               str(result['occlusion']),
                                                                               str(result['super_hq']),
                                                                               str(result['hq']), str(result['blur']),
                                                                               str(result['nonhuman']))
            f1.write(write_temp)
            f1.write('\n')


if __name__ == '__main__':
    main()
