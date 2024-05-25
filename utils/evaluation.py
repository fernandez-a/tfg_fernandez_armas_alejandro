from shapely.geometry import box
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils.leafDataset import LeafDataset
import albumentations as A
import numpy as np
from utils.transformations import get_valid_transforms
from utils.train_eval import collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc,average_precision_score, precision_score, recall_score

import sys
import cv2
from torchvision.ops import nms

def calculate_iou(gt_bbox, pred_bbox):
    gt_box = box(gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
    pred_box = box(pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3])
     
    intersection = gt_box.intersection(pred_box).area
    union = gt_box.area + pred_box.area - intersection
 
    iou = intersection / union if union != 0 else 0
    return iou

def pr_curve(true_l, conf_l):
    precisions, recalls, thresholds = precision_recall_curve(true_l, conf_l)

    pr_auc = auc(recalls, precisions)
    ap = average_precision_score(true_l, conf_l)
    
    print(f'PR AUC: {pr_auc}')
    print(f'MAP: {ap}')
    
    fig = plt.figure(figsize=(9, 6))
    plt.plot(recalls, precisions, label=f'PR curve MAP = {ap:.2f}',color="blue")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='upper left')
    fig.savefig('pr_curve.png')
    plt.show()


def eval_test(df_test, model, device, marking, BATCH_SIZE):
    
    valid_dataset = LeafDataset(image_ids=df_test.index.values,
                                dataframe=df_test,
                                transforms=get_valid_transforms()
                                )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)


    rec_l = []
    conf_l = []
    true_l = []
    for i, (images, targets, image_ids) in enumerate(tqdm(valid_data_loader)):
        _, h, w = images[0].shape
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        boxes = targets[0]['boxes'].cpu().numpy()
        labels = targets[0]['labels'].cpu().numpy()

        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")
    
        with torch.no_grad():
            outputs = model(images)
            
            oboxes = outputs['pred_boxes'][0].detach().cpu().numpy()
            prob = outputs['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
            
            keep = nms(torch.tensor(oboxes), torch.tensor(prob), iou_threshold=0.6)

            oboxes = oboxes[keep]
            prob = prob[keep]
            
            for box in zip(oboxes, prob):
                ious = [calculate_iou(box[0], gt_box) for gt_box in boxes]
                true_l.append(1 if max(ious) >= 0.5 else 0)
                
                conf_l.append(round(box[1],5))
    
    pr_curve(true_l, conf_l)
    prc =precision_score(true_l, [1 if conf > 0.6 else 0 for conf in conf_l])
    rec_l = recall_score(true_l, [1 if conf > 0.6 else 0 for conf in conf_l])
    print(f'Precision: {prc}, Recall: {rec_l}')



def view_sample(df_test, model, device, marking, BATCH_SIZE):
    valid_dataset = LeafDataset(image_ids=df_test.index.values,
                                 dataframe=df_test,
                                 transforms=get_valid_transforms()
                                )
     
    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=collate_fn)
    
    images, targets, image_ids = next(iter(valid_data_loader))
    _, h, w = images[0].shape
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    labels = targets[0]['labels'].cpu().numpy()
    boxes = targets[0]['boxes'].cpu().numpy()
    boxes = [np.array(box).astype(np.int32) for box in A.denormalize_bboxes(boxes, h, w)]

    sample = images[0].permute(1, 2, 0).cpu().numpy()
    sample = (sample * 255).astype(np.uint8)
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")
    
    with torch.no_grad():
        outputs = model(images)
        
    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2]+box[0], box[3]+box[1]),
                  (0,0,220), 1)
    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in  A.denormalize_bboxes(oboxes, h, w)]
    prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
    
    

    for box,p, label in zip(oboxes,prob, labels):
        if p > 0.25:
            color = (220,0,0)
            cv2.rectangle(sample,
                    (box[0], box[1]),
                    (box[2]+box[0], box[3]+box[1]),
                    color, 1)
    ax.set_axis_off()
    ax.imshow(sample)

    plt.show()