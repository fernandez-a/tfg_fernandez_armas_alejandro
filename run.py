import sys
sys.path.append('./detr')
from detr.models.detr import SetCriterion
from detr.models.matcher import HungarianMatcher
from utils.leafDataset import LeafDataset
from utils.transformation import get_train_transforms, get_valid_transforms
from utils.train_eval import collate_fn, train_fn, eval_fn
from utils.prepare_data import prepare_data
from utils.model import DETRModel
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, num_classes, num_queries, null_class_coef, BATCH_SIZE, LR, EPOCHS):
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.null_class_coef = null_class_coef
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPOCHS = EPOCHS
        self.marking, self.df_train, self.df_valid , _ = prepare_data('input/train_powdery.csv')

        self.matcher = HungarianMatcher()
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
        self.losses = ['labels', 'boxes', 'cardinality']

    def run(self):
        writer = SummaryWriter() 

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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model = DETRModel(num_classes=self.num_classes,num_queries=self.num_queries)
        model = model.to(device)
        criterion = SetCriterion(self.num_classes -1, self.matcher, self.weight_dict, eos_coef = self.null_class_coef, losses=self.losses)
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.LR)
        
        best_loss = 10**5
        num_epochs_no_improve = 0
        for epoch in range(self.EPOCHS):
            train_loss, train_loss_bbox, train_class_loss = train_fn(train_data_loader, model, criterion, optimizer, device, scheduler=None, epoch=epoch, batch_size=self.BATCH_SIZE)
            valid_loss, valid_loss_bbox , valid_class_loss= eval_fn(valid_data_loader, model, criterion, device, batch_size=self.BATCH_SIZE)
            writer.add_scalar('Loss/train', train_loss.avg, epoch)
            writer.add_scalar('Loss/valid', valid_loss.avg, epoch)
            print('Num epoch no improve: {}'.format(num_epochs_no_improve))
            print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}| TRAIN_BBOX_LOSS {}| VALID_BBOX_LOSS {}| TRAIN_CLASS_LOSS {}| VALID_CLASS_LOSS|'.format(epoch+1, train_loss.avg, valid_loss.avg, train_loss_bbox.avg, valid_loss_bbox.avg, train_class_loss.avg, valid_class_loss.avg))

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                print('Best loss {}'.format(best_loss))
                print('Best model found in Epoch {}........Saving Model'.format(epoch+1))
                torch.save(model.state_dict(), f'models/detr.pth')
                num_epochs_no_improve = 0
            else:
                num_epochs_no_improve += 1
                if num_epochs_no_improve >= 3:
                    print('Early stopping triggered. No improvement in validation loss for 3 epochs.')
                    early_stopping = True
                    break
        writer.close()
        
        if not early_stopping:
            print('Training completed without early stopping.')