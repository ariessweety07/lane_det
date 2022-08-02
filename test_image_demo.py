import os
import time

import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def eval(self, epoch, on_val=False, save_predictions=False):
    model = self.cfg.get_model()
    model_path = self.exp.get_checkpoint_path(epoch)
    self.logger.info('Loading model %s', model_path)
    model.load_state_dict(self.exp.get_epoch_model(epoch))
    model = model.to(self.device)
    model.eval()
    if on_val:
        dataloader = self.get_val_dataloader()
    else:
        dataloader = self.get_test_dataloader()
    test_parameters = self.cfg.get_test_parameters()
    predictions = []
    self.exp.eval_start_callback(self.cfg)
    with torch.no_grad():
        for idx, (images, _, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            output = model(images, **test_parameters)
            prediction = model.decode(output, as_lanes=True)
            predictions.extend(prediction)
            if self.view:
                img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                if self.view == 'mistakes' and fp == 0 and fn == 0:
                    continue
                cv2.imshow('pred', img)
                cv2.waitKey(0)

    if save_predictions:
        with open('predictions.pkl', 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    weights_path = "./save_weights/model.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)


    # load image
    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model.decode(output,as_lanes=True)

        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()