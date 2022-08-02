import os
import json
import random
import logging

import numpy as np

from utils.tusimple_metric import LaneEval

from .lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TuSimple(LaneDatasetLoader):
    def __init__(self, split='train', max_lanes=None, root=None):
        # 传入数据集在SPLIT_FILES中的key
        self.split = split
        # 跟目录
        self.root = root
        self.logger = logging.getLogger(__name__)

        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        # 获取对应数据集的标注文件的文件名
        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')
        # 对应tusimple数据集的尺寸
        self.img_w, self.img_h = 1280, 720
        self.annotations = []
        # 载入标注
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # 如果没有传入max_lanes就采用默认的max_lanes
        if max_lanes is not None:
            self.max_lanes = max_lanes
    # 重写父类函数，返回图像高度
    def get_img_heigth(self, _):
        return 720

    # 重写父类函数，返回图像高度
    def get_img_width(self, _):
        return 1280

    #
    def get_metrics(self, lanes, idx):
        label = self.annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        # 计算预测和gt的metrics
        _, fp, fn, matches, accs, _ = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)
        return fp, fn, matches, accs

    def pred2lanes(self, path, pred, y_samples):
        # y相对于原图的相对距离
        ys = np.array(y_samples) / self.img_h
        lanes = []
        # 遍历预测的lane
        for lane in pred:
            # 对应于于y的x
            xs = lane(ys)
            # 无效的x坐标
            invalid_mask = xs < 0
            # 映射回原图
            lane = (xs * self.get_img_width(path)).astype(int)
            # 无效的点标为-2
            lane[invalid_mask] = -2
            # 添加到lanes
            lanes.append(lane.tolist())

        return lanes

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.annotations = []
        max_lanes = 0
        # 遍历每一个json
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                # 加载对应lane的y
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                # 某一条线对应的坐标
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                # 存在一个list中
                lanes = [lane for lane in lanes if len(lane) > 0]
                # 记录当前epoch输入的最大lanes
                max_lanes = max(max_lanes, len(lanes))
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })
        # 如果是训练数据集的话，就随机打乱
        if self.split == 'train':
            random.shuffle(self.annotations)
        # 记录最大的lanes数目
        self.max_lanes = max_lanes
        # log信息记录？
        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)
    """
    """
    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    # 将预测的输出转换为对应tusimple的json
    def pred2tusimpleformat(self, idx, pred, runtime):
        # 转换为ms，所以test的runningtime都是1000
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        # 对于每一个预测都转换为json格式
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        # 存储对应的json文件
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval_predictions(self, predictions, output_basedir, runtimes=None):
        # 存储预测的json文件
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        # 加载对应预测的metric
        result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        table = {}
        # 返回metric
        for metric in result:
            table[metric['name']] = metric['value']

        return table

    # 得到对应index的标注
    def __getitem__(self, idx):
        return self.annotations[idx]

    # 返回标注的长度
    def __len__(self):
        return len(self.annotations)
