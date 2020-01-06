import glob
import json
import numpy as np
import cv2
from shapely import wkt
from tqdm import tqdm
import shutil


def shrinkpolygon(polygon, border=2):
    if polygon.area < 37:
        border = 0
    cx, cy = polygon.centroid.coords[0]
    pts = polygon.exterior.coords

    def shrink(pt):
        (x, y) = pt
        if x < cx:
            x += border
        elif x > cx:
            x -= border

        if y < cy:
            y += border
        elif y > cy:
            y -= border
        return [int(x), int(y)]

    return np.array([shrink(pt) for pt in pts])


def getjsons(pattern):
    return glob.glob(pattern)


def json2dict(filename):
    with open(filename) as f:
        j = json.load(f)
    return j


def prejson2img(filename):
    j = json2dict(filename)
    polygons = []
    for feat in j["features"]["xy"]:
        polygon = wkt.loads(feat["wkt"])
        coords = shrinkpolygon(polygon)
        polygons.append(coords)
    blank = np.zeros((1024, 1024, 1))
    cv2.fillPoly(blank, polygons, (255, 255, 255))
    edgelines = np.zeros((1024, 1024, 1))
    cv2.polylines(edgelines, polygons, True, (255, 255, 255), thickness=2)
    edgeheatmap = np.zeros((1024, 1024, 1))
    cv2.polylines(edgeheatmap, polygons, True, (63, 63, 63), thickness=13)
    cv2.polylines(edgeheatmap, polygons, True, (126, 126, 126), thickness=9)
    cv2.polylines(edgeheatmap, polygons, True, (189, 189, 189), thickness=5)
    cv2.polylines(edgeheatmap, polygons, True, (252, 252, 252), thickness=3)
    return blank, edgelines, edgeheatmap, len(polygons)


totalarea, area1, area2, area3, area4 = 0, 0, 0, 0, 0


def postjson2img(filename):
    global totalarea, area1, area2, area3, area4
    j = json2dict(filename)
    polygons = []
    damages = []
    for feat in j["features"]["xy"]:
        polygon = wkt.loads(feat["wkt"])
        # coords = shrinkpolygon(polygon)
        coords = polygon
        polygons.append(coords)
        damages.append(feat["properties"]["subtype"])

    polytypes0 = [polygons[i] for i, d in enumerate(damages) if d == "un-classified"]
    polytypes1 = [polygons[i] for i, d in enumerate(damages) if d == "no-damage"]
    polytypes2 = [polygons[i] for i, d in enumerate(damages) if d == "minor-damage"]
    polytypes3 = [polygons[i] for i, d in enumerate(damages) if d == "major-damage"]
    polytypes4 = [polygons[i] for i, d in enumerate(damages) if d == "destroyed"]

    totalarea += 1024 * 1024
    area1 += sum(p.area for p in polytypes1)
    area2 += sum(p.area for p in polytypes2)
    area3 += sum(p.area for p in polytypes3)
    area4 += sum(p.area for p in polytypes4)

    blank = np.zeros((1024, 1024, 1))
    color = (63, 63, 63)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes1], color)
    color = (126, 126, 126)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes2], color)
    color = (189, 189, 189)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes3], color)
    color = (252, 252, 252)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes4], color)

    polygons = [
        shrinkpolygon(p) for p in polytypes1 + polytypes2 + polytypes3 + polytypes4
    ]

    edgelines = np.zeros((1024, 1024, 1))
    cv2.polylines(edgelines, polygons, True, (255, 255, 255), thickness=9)

    edgeheatmap = np.zeros((1024, 1024, 1))
    cv2.polylines(edgeheatmap, polygons, True, (63, 63, 63), thickness=13)
    cv2.polylines(edgeheatmap, polygons, True, (126, 126, 126), thickness=9)
    cv2.polylines(edgeheatmap, polygons, True, (189, 189, 189), thickness=5)
    cv2.polylines(edgeheatmap, polygons, True, (252, 252, 252), thickness=3)
    return blank, edgelines, edgeheatmap, len(polygons)


def prepostjson2img(prefilename):
    postfilename = prefilename.replace("pre_disaster", "post_disaster")
    preloc, preedge, preweight, precount = prejson2img(prefilename)
    postdmg, postedge, postweight, postcount = postjson2img(postfilename)
    prefilename = (
        prefilename.replace("pre_disaster", "pre_mask_disaster")
        .replace(".json", ".png")
        .replace("labels", "targets")
    )
    postfilename = (
        postfilename.replace("post_disaster", "post_mask_disaster")
        .replace(".json", ".png")
        .replace("labels", "targets")
    )
    premask = np.concatenate([preloc, preedge, preweight], axis=2)
    cv2.imwrite(prefilename, premask)
    postmask = np.concatenate([postdmg, postedge, postweight], axis=2)
    cv2.imwrite(postfilename, postmask)
    return precount, postcount


def moveempty(prefn, prelabelsfn):
    postfn = prefn.replace("pre", "post")
    postlabelsfn = prelabelsfn.replace("pre", "post")
    emptyprefn = prefn.replace("images1024", "emptyimages1024")
    emptypostfn = postfn.replace("images1024", "emptyimages1024")
    emptyprelabelsfn = prelabelsfn.replace("labels1024", "emptylabels1024")
    emptypostlabelsfn = postlabelsfn.replace("labels1024", "emptylabels1024")
    shutil.move(prefn, emptyprefn)
    shutil.move(postfn, emptypostfn)
    shutil.move(prelabelsfn, emptyprelabelsfn)
    shutil.move(postlabelsfn, emptypostlabelsfn)


if __name__ == "__main__":
    pattern = "data/train/images1024/*pre*"
    emptyimages = []
    for f in tqdm(getjsons(pattern)):
        fn = f.replace("data/train/images1024", "data/train/labels1024").replace(
            "png", "json"
        )
        precount, postcount = prepostjson2img(fn)
        if not precount or not postcount:
            emptyimages.append(fn + "\n")
            moveempty(f,fn) 

    with open("data/train/emptyimages.txt", "w") as f:
        f.writelines(emptyimages)

    print(
        "Total area: {} , Pixel class fractions: no-damage {} , minor-damage {} , major-damage {} , destroyed {} ".format(
            totalarea,
            area1 / totalarea,
            area2 / totalarea,
            area3 / totalarea,
            area4 / totalarea,
        )
    )
