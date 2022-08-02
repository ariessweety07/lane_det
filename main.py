import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment
"""
cfgs: 默认/预设配置文件
lib
datasets
culane.py: culane数据集加载器
lane_dataset.py: 将来自LaneDatasetLoader中的未经过处理的 annotations 转换为模型可以使用的形式
lane_dataset_loader.py: 每个数据集加载器实现的抽象类
llamas.py: llamas数据集加载器
nolabel_dataset.py: 加载不需注释的的数据集
tusimple.py: tusimple数据集加载器
models:
laneatt.py: LaneATT模型的实现
matching.py: 用于gt和proposal匹配的效用函数
resnet.py: resnet 实现部分
nms: LaneATT模型的实现
config.py: LaneATT模型的实现
experiment.py: 跟踪和存储有关每个实验的信息
focal_loss.py: focal loss的实现
lane.py: 车道线表示
runner.py: 训练和测试循环
utils:
culane_metric.py: 非官方的CULane数据集度量实现
gen_anchor_mask.py: 计算数据集中要在锚点筛选步骤中使用的每个锚点的频率（论文提到锚点的数量会限制速度，所以挑选使用频率最大的部分锚点）
gen_video.py: 从模型预测生成视频
llamas_metric.py llamas数据集的实用程序函数
speed.py: 测量模型的效率相关指标
tusimple_metric.py: tusimple数据集图片度量的官方实现
viz_dataset.py: 显示从数据集采样的图像（增强后）
main.py: 运行实验的训练或测试阶段
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
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
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    runner = Runner(cfg, exp, device, view=args.view, resume=args.resume, deterministic=args.deterministic)
    if args.mode == 'train':
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info('Training interrupted.')
    runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)


if __name__ == '__main__':
    main()
