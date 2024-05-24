import torch
from utils.average_meter import AverageMeter
from tqdm import tqdm
import numpy as np
from shapely.geometry import box

def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch, batch_size):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()
    summary_bbox_loss = AverageMeter()
    summar_class_loss = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for step, (images, targets, image_ids) in enumerate(tk0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        bbox_loss = loss_dict['loss_bbox']
        class_loss = loss_dict['loss_ce']
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), batch_size)
        summary_bbox_loss.update(bbox_loss.item(), batch_size)
        summar_class_loss.update(class_loss.item(), batch_size)


        tk0.set_postfix(loss=summary_loss.avg)


    return summary_loss, summary_bbox_loss, summar_class_loss



def eval_fn(data_loader, model, criterion, device, batch_size):
    model.eval()
    criterion.eval()

    summary_loss = AverageMeter()
    summary_bbox_loss = AverageMeter()
    summar_class_loss = AverageMeter()


    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            bbox_loss = loss_dict['loss_bbox']
            class_loss = loss_dict['loss_ce']
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(), batch_size)
            summary_bbox_loss.update(bbox_loss.item(), batch_size)
            summar_class_loss.update(class_loss.item(), batch_size)


            tk0.set_postfix(loss=summary_loss.avg)


    return summary_loss, summary_bbox_loss, summar_class_loss



def collate_fn(batch):
    return tuple(zip(*batch))
