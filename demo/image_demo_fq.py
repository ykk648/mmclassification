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
    parser.add_argument('--img_dir',default='/workspace/85_cluster/mnt/douban_spider/data_out',help='Image file')
    parser.add_argument('--config', default='configs/_base_/facequality/face_quality_0911.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/face_quality_0911/epoch_23.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ['high', 'low']
    img_dir = Path(args.img_dir)

    with open('/workspace/85_cluster/mnt/douban_spider/iqa_celebrity_clusterB_high_1119.txt', 'a') as f1:
        with open('/workspace/85_cluster/mnt/douban_spider/iqa_celebrity_clusterB_low_1119.txt', 'a') as f2:
            for img_p in tqdm(list(img_dir.rglob('*/clusterB*.jpg'))):
                # test a single image
                img_p_str = str(img_p)
                result = inference_model(model, img_p_str)
                # f.write(img_p.strip(root_p) + ' ' + result['pred_class'])
                # f.write('\n')

                if result['pred_class'] == 'high':
                    # print(img_p_str)
                    f1.write(img_p_str[48:])
                    f1.write('\n')
                else:
                    f2.write(img_p_str[48:])
                    f2.write('\n')

    # out_high = '/workspace/85_cluster/mnt/sina/sina_out_high'
    # out_low = '/workspace/85_cluster/mnt/sina/sina_out_low'
    # count = 0
    # for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
    #     # print(img_p)
    #     # count +=1
    #     # if count == 100:
    #     #     break
    #     # test a single image
    #     img_p_str = str(img_p)
    #     try:
    #         result = inference_model(model, img_p_str)
    #     except cv2.error:
    #         result = None
    #         print(img_p)
    #
    #     # print(result)
    #     # break
    #
    #     if result['pred_class'] == 'high' and result['pred_score'] > 0.9:
    #         # shutil.copyfile(img_p, out_high + '/' + make_random_name(str(img_p)))
    #
    #         pass
    #     elif result['pred_class'] == 'low' and result['pred_score'] > 0.95:
    #         shutil.copyfile(img_p, out_low + '/' + make_random_name(str(img_p)))
    #         # shutil.copyfile(img_p, out_low + make_random_name(str(img_p)))
    #         pass
    #     else:
    #         # print(result['pred_class'])
    #         print(result['pred_score'])


if __name__ == '__main__':
    main()
