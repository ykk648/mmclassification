# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
from mmcls.apis import inference_model, init_model, show_result_pyplot
import shutil
from tqdm import tqdm
import cv2
import uuid


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir',
                        default='/workspace/WeiBo_hq_by_BZ_1019_1_high',
                        help='Image file')
    parser.add_argument('--config', default='configs/_base_/facequality/face_gender_1102.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/face_gender_1102/epoch_20.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # root_p = '/workspace/151_cluster/nfs/cv_rsync/dataset/HifiFace/'
    # # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ['male', 'female']
    img_dir = Path(args.img_dir)
    # with open('/workspace/151_cluster/nfs/cv_rsync/dataset/HifiFace/face_quality/iqa_celebrity_0916.txt', 'a') as f:
    #     for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
    #         # test a single image
    #         img_p_str = str(img_p)
    #         result = inference_model(model, img_p_str)
    #         # f.write(img_p.strip(root_p) + ' ' + result['pred_class'])
    #         # f.write('\n')
    #
    #         if result['pred_class'] == 'high':
    #             f.write(img_p_str[53:])
    #             f.write('\n')

    out_high = '/workspace/male'
    out_low = '/workspace/female'
    count = 0
    for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
        # print(img_p)
        # count +=1
        # if count == 100:
        #     break
        # test a single image
        img_p_str = str(img_p)
        try:
            result = inference_model(model, img_p_str)
        except cv2.error:
            result = None
            print(img_p)

        if result['pred_class'] == 'male':
            shutil.copyfile(img_p, out_high + '/' + make_random_name(str(img_p)))
        elif result['pred_class'] == 'female':
            shutil.copyfile(img_p, out_low + '/' + make_random_name(str(img_p)))
            # shutil.copyfile(img_p, out_low + make_random_name(str(img_p)))
            pass
        else:
            print(result['pred_class'])


if __name__ == '__main__':
    main()
