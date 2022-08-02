import logging
import argparse
import torch
from lib.config import Config
from lib.experiment import Experiment
import numpy as np
import cv2
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    # 输入当前待测试图像文件路径的参数
    parser.add_argument("--imageFile", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
    parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == 'train':
        raise Exception("The `epoch` parameter should not be set when training")
    if args.view is not None and args.mode != "test":
        raise Exception('Visualization is only available during evaluation')
    if args.cpu:
        raise Exception("CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")
    return args

def main():
    # 传递参数
    args = parse_args()
    # 当前模式和当前的net
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    model = cfg.get_model()
    model_path = exp.get_checkpoint_path(args.epoch or exp.get_last_checkpoint_epoch())
    # 加载模型
    logging.getLogger(__name__).info('Loading model %s', model_path)
    model.load_state_dict(exp.get_epoch_model(args.epoch or exp.get_last_checkpoint_epoch()))
    model = model.to(device)
    # 测试模式
    model.eval()
    test_parameters = cfg.get_test_parameters()
    # 预测的点
    predictions = []
    exp.eval_start_callback(cfg)
    imageFile = args.imageFile
    # 读取测试文件
    image=cv2.imread(imageFile)
    # resize到模型需要的输入大小
    image = cv2.resize(image,(640,360))
    # 转换成0-1之间的tensor
    image=image/255.
    # 转换到float格式在cuda上处理
    image = torch.from_numpy(image).cuda().float()
    # cv2默认存储的颜色空间顺序是bgr，->rbg  增添一个batch维度
    image = torch.unsqueeze(image.permute(2, 0, 1),0)
    # 得到预测输出
    output = model(image, **test_parameters)
    # 将输出decode为lane
    prediction = model.decode(output, as_lanes=True)
    # 添加到预测list中
    predictions.extend(prediction)
    # 转换为bgr格式，转换成numpy格式待会儿存储
    img = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.resize(img, (640, 360))
    img_h, _, _ = img.shape
    pad = 0
    # 有一部分anchor超出特征图的边界，所以进行0填充
    if pad > 0:
        img_pad = np.zeros((360 + 2 * pad, 640 + 2 * pad, 3), dtype=np.uint8)
        img_pad[pad:-pad, pad:-pad, :] = img
        img = img_pad
    # 对于预测的lane的每一个点
    for i, l in enumerate(prediction[0]):
        points = l.points  #<class 'lib.lane.Lane'>
        # 缩放到每个点在原图中的位置
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        # 四舍五入
        points = points.round().astype(int)
        # 因为有填充所以有偏移，加上偏移量
        points += pad
        # 连接相邻的两个点
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                           tuple(curr_p),
                           tuple(next_p),
                           color=(255, 0, 255),
                           thickness=3)
    # 保存测试文件
    cv2.imwrite("./"+imageFile[-5:-4]+"_predict_lane_result.jpg", img)

if __name__ == '__main__':
    main()
