# pylint: disable-all
import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression


class LaneEval(object):
    lr = LinearRegression()
    # pixel_thresh代表点的预测正确的阈值：要在gt的20个pixel范围之内
    pixel_thresh = 20
    # pt_thresh代表lane预测正确的阈值：预测的lane的gt的预测正确的点占比>=85%
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        # 用线性回归拟合，得到斜率用三角函数求出对应的角度
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            # 权重数组的第一位
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        # 计算pred和gt的两条线的准确度，和f1组成为tusimple的metric
        # 要预测值和gt在>=0的范围内，否则算作-100，就这个点的准确度为0
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        # np的where，满足条件输出x，不满足条件输出y，
        # 此处为，计算pred和gt之间距离小于阈值的点的个数，最后除以gt的长度，算作line的准确度
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def distances(pred, gt):
        # 计算预测和gt的距离
        return np.abs(pred - gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time, get_matches=False):
        # 预测的lane的总index的长度是和gt的y取点数是一样的否则报错
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        """不太懂这个想要干嘛"""
        if running_time > 20000 or len(gt) + 2 < len(pred):
            if get_matches:
                # 第二个代表是否匹配，第三个代表每个点的准确度？第四个代表每个点的距离？
                return 0., 0., 1., [False] * len(pred), [0] * len(pred), [None] * len(pred)
            return 0., 0., 1.,
        # 得到每个index对应x和y点的角度
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        # 计算对应每个点的水平阈值的圆的半径阈值是多少
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        # 计算fp和fn
        fp, fn = 0., 0.
        matched = 0.
        my_matches = [False] * len(pred)
        my_accs = [0] * len(pred)
        my_dists = [None] * len(pred)
        # 对应gt的每个x点和对应的阈值进行遍历
        for x_gts, thresh in zip(gt, threshs):
            # 对固定的y计算x点的准确度
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            # 存储最大准确度
            my_accs = np.maximum(my_accs, accs)
            # 更新最大准确度
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            # 计算对应每个x的距离
            my_dist = [LaneEval.distances(np.array(x_preds), np.array(x_gts)) for x_preds in pred]
            if len(accs) > 0:
                # 存储具有最大准确度的g的点的坐标和距离
                my_dists[np.argmax(accs)] = {
                    'y_gts': list(np.array(y_samples)[np.array(x_gts) >= 0].astype(int)),
                    'dists': list(my_dist[np.argmax(accs)])
                }
            # 如果最大准确度小于准确度阈值，为漏报
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                # 否则，将对应index的matches设置为true
                my_matches[np.argmax(accs)] = True
                # match的num+1
                matched += 1
            # 将这条线的准确度添加
            line_accs.append(max_acc)
        # 假阳性，误检是预测的比匹配的多出来的数目
        fp = len(pred) - matched
        # 如果lanes>4  并且有漏检，酌情将漏检数目-1？因为lanes太多
        if len(gt) > 4 and fn > 0:
            fn -= 1
        # 求出所有lines的准确度之和
        s = sum(line_accs)
        # 如果gt的lines数目>4，减去一个最低准确度
        if len(gt) > 4:
            s -= min(line_accs)
        # 如果有gt和proposal匹配
        if get_matches:
            # 返回平均准确度，平均误检率，平均漏检率，对应输入的matches的bool数组，准确度数组，距离数组
            return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(
                min(len(gt), 4.), 1.), my_matches, my_accs, my_dists
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            # 读取预测的json文件
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        # 读取gt的json文件
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        # 长度不匹配，就是不匹配，没有预测
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {img['raw_file']: img for img in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        run_times = []
        for pred in json_pred:
            # 如果文件不存在，或者没有预测的线，或者run_time不在
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            # 读取文件名
            raw_file = pred['raw_file']
            # 预测的lanes
            pred_lanes = pred['lanes']
            """
            不懂
            """
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                # 计算一个epoch的一个输入的准确度等等metric
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            # 累计计算平均
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        # 返回对应这个epoch的metric计算值
        return json.dumps([{
            'name': 'Accuracy',
            'value': accuracy / num,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': fp / num,
            'order': 'asc'
        }, {
            'name': 'FN',
            'value': fn / num,
            'order': 'asc'
        }, {
            'name': 'FPS',
            'value': 1000. / np.mean(run_times)
        }])


if __name__ == '__main__':
    import sys

    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e)
        # sys.exit(e.message)
