import sys
sys.path.append('./detr/')
from detr.models.detr import SetCriterion
from detr.models.matcher import HungarianMatcher
from utils.leafDataset import LeafDataset
from utils.transformation import get_train_transforms, get_valid_transforms
from utils.train_eval import collate_fn, train_fn, eval_fn
from utils.prepare_data import prepare_data
from model import DETRModel
import torch
from torch.utils.data import DataLoader


class Runner:
    def __init__(self, num_classes, num_queries, null_class_coef, BATCH_SIZE, LR, EPOCHS):
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.null_class_coef = null_class_coef
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPOCHS = EPOCHS
        self.marking, self.df_train, self.df_valid = prepare_data('input/train.csv')
        self.matcher = HungarianMatcher()
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
        self.losses = ['labels', 'boxes', 'cardinality']

    def run(self):
        train_dataset = LeafDataset(
            image_ids=self.df_train.index.values,
            dataframe=self.marking,
            transforms=get_train_transforms()
        )

        valid_dataset = LeafDataset(
            image_ids=self.df_valid.index.values,
            dataframe=self.marking,
            transforms=get_valid_transforms()
        )
        
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        device = torch.device('cuda')
        model = DETRModel(num_classes=self.num_classes,num_queries=self.num_queries)
        model = model.to(device)
        criterion = SetCriterion(self.num_classes -1, self.matcher, self.weight_dict, eos_coef = self.null_class_coef, losses=self.losses)
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.LR)
        
        best_loss = 10**5
        epochs_no_improve = 0
        n_epochs_stop = 5

        for epoch in range(self.EPOCHS):
            train_loss = train_fn(train_data_loader, model, criterion, optimizer, device, scheduler=None, epoch=epoch, batch_size=self.BATCH_SIZE)
            valid_loss = eval_fn(valid_data_loader, model, criterion, device, batch_size=self.BATCH_SIZE)

            print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1, train_loss.avg, valid_loss.avg))

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                print('Best model found in Epoch {}........Saving Model'.format(epoch+1))
                torch.save(model.state_dict(), f'models/detr_best_{epoch}.pth')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.n_epochs_stop:
                    print('Early stopping!')
                    break

        torch.save(model.state_dict(), 'models/detr_final.pth')
        self.writer.close()