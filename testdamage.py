from traindamageunet import XViewSystem
from testloc import getlocmodel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import pytorch_lightning as pl
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import argparse


transformimg = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406][::-1], std=[0.225, 0.224, 0.225][::-1]
        ),
    ]
)


def getdamagemodel(ckpt, tagscsv="workspace/damage/meta_tags.csv", cuda=True):
    pretrained_model = XViewSystem.load_from_metrics(
        weights_path=ckpt, tags_csv=tagscsv,
    )
    pretrained_model.eval()
    pretrained_model.freeze()
    if cuda:
        pretrained_model = pretrained_model.cuda()
    return pretrained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate damage and localization predictions for XView2"
    )
    parser.add_argument("--dckpt", metavar="/path/to/damagecheckpoint", required=True)
    parser.add_argument("--lckpt", metavar="/path/to/loccheckpoint", required=True)
    args = parser.parse_args()

    dmgmodel = getdamagemodel(args.dckpt)
    locmodel = getlocmodel(args.lckpt)
    PATTERN = "data/test/images/*post*.png"

    print("generate loc and damage predictions: ")

    posts = glob(PATTERN)
    pres = [fn.replace("post", "pre") for fn in posts]
    dmgs = [
        "results/predictions/test_damage_{}_prediction.png".format(
            fn.split(".")[-2].split("_")[-1]
        )
        for fn in posts
    ]
    vizdmgs = [
        "results/vizpredictions/test_damage_{}_prediction.png".format(
            fn.split(".")[-2].split("_")[-1]
        )
        for fn in posts
    ]
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

    for i in tqdm(range(len(posts))):
        post = transformimg(cv2.imread(posts[i])).unsqueeze(0)
        pre = transformimg(cv2.imread(pres[i])).unsqueeze(0)
        locfn = locs[i]
        dmgfn = dmgs[i]
        vizdmgfn = vizdmgs[i]
        vizlocfn = vizlocs[i]

        loc = locmodel(pre.cuda())[:, 0:2, :, :]
        loc = torch.argmax(loc, dim=1).squeeze().detach().cpu().numpy().astype(np.uint8)

        mask = deepcopy(loc)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)

        allmask = cv2.copyMakeBorder(erosion, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        polygons = cv2.findContours(
            erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(-1, -1)
        )
        polygons = polygons[0] if len(polygons) == 2 else polygons[1]
        cv2.fillPoly(loc, polygons, (1, 1, 1))
        polygons = [polygon.flatten() for polygon in polygons]
        points = [np.array(point).reshape(-1, 2).astype(int) for point in polygons]

        _, dmg = dmgmodel([pre.cuda(), post.cuda()])
        dmg = torch.argmax(dmg[:, 1:, :, :], dim=1).squeeze() + 1
        dmg = dmg.cpu().numpy().astype(np.uint8)

        outblank = np.zeros_like(dmg)
        kernel = np.ones((5, 5), np.uint8)
        for point in points:
            if len(point) < 2:
                continue
            pblank = np.zeros_like(mask)
            cv2.fillPoly(pblank, [point], (1, 1, 1))
            pblank = cv2.dilate(pblank, kernel, iterations=2)
            pblank[pblank > 0] = dmg[pblank > 0]
            outblank[pblank > 0] = dmg[pblank > 0]
            phist, pcounts = np.unique(pblank, return_counts=True)
            if len(phist) < 3:
                continue
            phist = phist[1:]
            pcounts = pcounts[1:]
            area = sum(pcounts) * 1.0
            colorcount = defaultdict(int)
            for i in range(len(phist)):
                colorcount[phist[i]] = pcounts[i] / area
            color = max(colorcount, key=colorcount.get)
            if colorcount[4] > 0.5:
                color = 4
            elif colorcount[3] > 0.20 or colorcount[4] > 0.15:
                color = 3
            elif colorcount[2] > 0.10 or colorcount[3] > 0.1 or colorcount[4] > 0.05:
                color = 2
            print(phist, pcounts, color)
            outblank[pblank > 0] = color

        dmg[outblank > 0] = outblank[outblank > 0]

        loc = cv2.dilate(loc, kernel, iterations=2)
        cv2.imwrite(locfn, loc)
        cv2.imwrite(vizlocfn, loc * 63)
        blank = np.zeros_like(dmg)
        blank[loc.astype(np.bool)] = dmg[loc.astype(np.bool)]
        cv2.imwrite(dmgfn, blank)
        vizblank = 63 * blank
        cv2.imwrite(vizdmgfn, vizblank)
