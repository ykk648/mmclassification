# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/workspace/151_cluster/nfs/dataset/download/celian/2a54be190b47427ba8a746674277bc67.jpg', help='Image file')
    parser.add_argument('--config', default='configs/_base_/facequality/face_quality_0911.py', help='Config file')
    parser.add_argument('--checkpoint',default='work_dirs/face_quality_0911/epoch_23.pth',  help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    model.CLASSES = ['high', 'low']
    result = inference_model(model, args.img)
    print(result)
    # {'pred_label': 0, 'pred_score': 0.9918617010116577, 'pred_class': 'high'}
    # show the results
    show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
