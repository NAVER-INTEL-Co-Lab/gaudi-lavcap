#%%
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

predict_json = "results/mmcap_float/epoch1_coco_result.json"
gt_json = "audiocaps/test_coco.json"


coco = COCO(gt_json)
coco_result = coco.loadRes(predict_json)
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.params['image_id'] = coco_result.getImgIds()
coco_eval.evaluate()

stats = {}
for metric, score in coco_eval.eval.items():
    stats[metric] = score

print(stats)