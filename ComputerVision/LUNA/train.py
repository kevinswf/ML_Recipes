import argparse
import sys
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import LunaModel
from dataset import LunaDataset
from util.util import iterate_with_estimate
from util.loggingconf import logging

log = logging.getLogger(__name__)

# for each sample, keep track of three metrics
METRICS_LABEL_IDX = 0   # groundtruth class labels
METRICS_PRED_IDX = 1    # predicted class probabilities
METRICS_LOSS_IDX = 2    # loss
METRICS_SIZE = 3

class LunaTrainingApp:

    def __init__(self, sys_argv=None):
        
        # if arguments is not specified during class init, get arguments from command line
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', help='Number of epochs to train', default=1, type=int)
        parser.add_argument('--batch-size', help='Batch size', default=32, type=int)
        parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
        parser.add_argument('--momentum', help='SGD momentum', default=0.9, type=float)
        parser.add_argument('--num-workers', help='dataloader workers', default=6, type=int)
        self.args = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.total_train_samples_count = 0              # how many samples have we trained

        # GPU?
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.model = self.init_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')  # reduction=none will give loss per sample, i.e. a tensor of loss, instead of averaged over batch


    def init_model(self):
        # init model and send to GPU if any
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA, {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model
    
    def init_dataloader(self):
        batch_size = self.args.batch_size
        # if multiple cuda
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_ds = LunaDataset(val_stride=10, is_val_set=False)
        val_ds = LunaDataset(val_stride=10, is_val_set=True)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=self.use_cuda, num_workers=self.args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=self.use_cuda, num_workers=self.args.num_workers)

        return train_dl, val_dl
    
    def train(self, epoch_idx, train_dl):
        # set model to train mode
        self.model.train()

        # init metrics to track each individual sample's label, predicted probability, and loss
        train_metrics = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        # create a batch iterator with time estimation
        batch_iter = iterate_with_estimate(train_dl, f"Epoch {epoch_idx} training", start_idx=train_dl.num_workers)

        # for each batch
        for batch_idx, batch in batch_iter:
            # reset grad
            self.optimizer.zero_grad()

            # forward and compute loss
            loss = self.compute_batch_loss(batch_idx, batch, train_dl.batch_size, train_metrics)

            # backward compute grad
            loss.backward()

            # update params
            self.optimizer.step()

        self.total_train_samples_count += len(train_dl)

        # return metrics, and to cpu
        return train_metrics.to('cpu')
    
    def eval(self, epoch_idx, val_dl):
        # set model to eval mode
        self.model.eval()

        with torch.no_grad():
            # init metrics to track each individual sample's label, predicted probability, and loss
            val_metrics = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            # create a batch iterator with time estimation
            batch_iter = iterate_with_estimate(val_dl, f"Epoch {epoch_idx} eval", start_idx=val_dl.num_workers)

            for batch_idx, batch in batch_iter:
                # forward and compute loss
                loss = self.compute_batch_loss(batch_idx, batch, val_dl.batch_size, val_metrics)

        # return metrics, and to cpu
        return val_metrics.to('cpu')

    
    def compute_batch_loss(self, batch_idx, batch, batch_size, metrics):
        # each sample is CT_chunk, class_label, uid, center_irc
        input, label, uid, center_irc = batch

        # move input and label to GPU
        input = input.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        # forward pass
        logits, proba = self.model(input)

        # calculate loss
        loss = self.loss_func(logits, label[:, 1])  # index 1 is is_nodule class

        # calculate the batch's start and end index in all samples
        start_idx = batch_idx * batch_size
        end_idx = start_idx + label.size(0)

        # log each sample's label, predicted proba, and loss (no need gradients)
        metrics[METRICS_LABEL_IDX, start_idx:end_idx] = label[:, 1].detach()
        metrics[METRICS_PRED_IDX, start_idx:end_idx] = proba[:, 1].detach()     # probability of is_nodule
        metrics[METRICS_LOSS_IDX, start_idx:end_idx] = loss.detach()

        # return the loss average over the batch
        return loss.mean()
    
    # function to print logs for each epoch
    def log_metrics(self, epoch_idx, mode, metrics, classification_threshold=0.5):
        # get the indices of positive ground truth and predictions, i.e. predicted to be is nodule
        pos_labels_idx = metrics[METRICS_LABEL_IDX] > classification_threshold             # label is either 0 or 1, so > threshold is ok
        pos_preds_idx = metrics[METRICS_PRED_IDX] > classification_threshold

        # the negative samples
        neg_labels_idx = ~pos_labels_idx
        neg_preds_idx = ~pos_preds_idx

        # total pos and neg ground truth samples
        pos_count = int(pos_labels_idx.sum())
        neg_count = int(neg_labels_idx.sum())

        # number of correct predictions
        pos_correct = int((pos_labels_idx & pos_preds_idx).sum())
        neg_correct = int((neg_labels_idx & neg_preds_idx).sum())

        metrics_log = {}
        metrics_log['loss_avg'] = metrics[METRICS_LOSS_IDX].mean()                                      # average loss on all samples
        metrics_log['loss_avg_pos'] = metrics[METRICS_LOSS_IDX, pos_labels_idx].mean()                  # average loss on positive samples, i.e. labled nodule samples
        metrics_log['loss_avg_neg'] = metrics[METRICS_LOSS_IDX, neg_labels_idx].mean()                  # average loss on negative samples
        metrics_log['correct_all'] = (pos_correct + neg_correct) / np.float32(metrics.shape[1]) * 100   # correct prediction percentage on all samples
        metrics_log['correct_pos'] = pos_correct / np.float32(pos_count) * 100                          # correct prediction percentage on all samples
        metrics_log['correct_neg'] = neg_correct / np.float32(neg_count) * 100                          # correct prediction percentage on all samples

        # print logs
        log.info(f"Epoch {epoch_idx} {mode} {metrics_log['loss_avg']:.4f} loss, {metrics_log['correct_all']:.1f}% correct")
        log.info(f"Epoch {epoch_idx} {mode} Positives {metrics_log['loss_avg_pos']:.4f} loss, {metrics_log['correct_pos']:.1f}% correct")
        log.info(f"Epoch {epoch_idx} {mode} Negatives {metrics_log['loss_avg_neg']:.4f} loss, {metrics_log['correct_neg']:.1f}% correct")


    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.args}")

        # construct data loader
        train_dl, val_dl = self.init_dataloader()

        # train and eval each epoch
        for epoch in range(1, self.args.epochs + 1):
            log.info(f'Epoch {epoch} of {self.args.epochs}')

            train_metrics = self.train(epoch, train_dl)
            self.log_metrics(epoch, 'train', train_metrics)

            val_metrics = self.eval(epoch, val_dl)
            self.log_metrics(epoch, 'eval', val_metrics)

        print("Done")


if __name__ == '__main__':
    LunaTrainingApp().main()