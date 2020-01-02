from trainlocunet import XViewLocSystem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import pytorch_lightning as pl
from glob import glob
from tqdm import tqdm
import argparse

transformimg = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406][::-1], std=[0.225, 0.224, 0.225][::-1]
        ),
    ]
)


def getlocmodel(ckpt, tagscsv="workspace/loc/meta_tags.csv", cuda=True):
    pretrained_model = XViewLocSystem.load_from_metrics(
        weights_path=ckpt, tags_csv=tagscsv,
    )
    pretrained_model.eval()
    pretrained_model.freeze()
    if cuda:
        pretrained_model = pretrained_model.cuda()
    return pretrained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate localization predictions for XView2 submission"
    )
    parser.add_argument(
        "--lckpt", required=True, metavar="/path/to/localization/checkpoint"
    )
    args = parser.parse_args()
    pretrained_model = getlocmodel(args.lckpt)

    print("Generating localization predictions: ")

    PATTERN = "data/test/images/*pre*.png"
    pres = glob(PATTERN)
    locs = [
        "results/predictions/test_localization_{}_prediction.png".format(
            fn.split(".")[-2].split("_")[-1]
        )
        for fn in pres
    ]
    vizlocs = [
        "results/vizpredictions/test_localization_{}_prediction.png".format(
            fn.split(".")[-2].split("_")[-1]
        )
        for fn in pres
    ]

    for i in tqdm(range(len(pres))):
        pre = transformimg(cv2.imread(pres[i])).unsqueeze(0)
        locfn = locs[i]
        vizlocfn = vizlocs[i]
        loc = pretrained_model(pre.cuda())[:, 0:2, :, :]
        loc = torch.argmax(loc, dim=1).squeeze()
        loc = loc.cpu().numpy().astype(np.uint8)
        cv2.imwrite(locfn, loc)
        cv2.imwrite(vizlocfn, loc * 255)
