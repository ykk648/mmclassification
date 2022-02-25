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
    parser.add_argument('--img_dir',
                        default='/workspace/85_cluster/mnt/dataset/download/multi_face_extract/multi_face_extract_5w_1',
                        help='Image file')
    parser.add_argument('--config', default='local_config/face_attr_1221.py', help='Config file')
    parser.add_argument('--checkpoint', default='/workspace/codes/mmclassification/work_dirs/face_attr_epoch_21_210106.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ('female', 'male',
                     'front', 'side',
                     'clean', 'occlusion',
                     'super_hq', 'hq', 'blur',
                     'nonhuman')

    for img_p in tqdm(get_path_by_ext(args.img_dir)):
        img_name = img_p.stem
        json_path = img_p.parent / (img_name+'.json')
        result = inference_model(model, str(img_p), multi_label=True)

        json_dict_init = {
            "version": "4.6.0",
            "flags": {
                "男": bool(result['male'] > result['female']),
                "侧脸": bool(result['side'] > result['front']),
                "遮挡": bool(result['occlusion'] > result['clean']),
                "非常清晰": bool(result['super_hq'] > result['hq'] and result['super_hq'] > result['blur']),
                "清晰": bool(result['hq'] > result['super_hq'] and result['hq'] > result['blur']),
                "模糊": bool(result['blur'] > result['hq'] and result['blur'] > result['super_hq']),
                "非人脸": bool(result['nonhuman'] > 0.5),
            },
            "shapes": [],
            "imagePath": "{}.jpg".format(img_name),
            "imageData": None,
            "imageHeight": 512,
            "imageWidth": 512
        }
        # print(json_dict_init)
        CVFile('{}'.format(json_path)).json_write(json_dict_init)


if __name__ == '__main__':
    main()
