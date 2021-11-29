import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, results_folder='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc_min = np.Inf
        self.delta = delta
        self.path = results_folder
        self.trace_func = trace_func
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, acc, model, epoch):

        score = acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, acc, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.acc_min:.6f} --> {acc:.6f}).  Saving model ...')
        filename = 'model_val_acc_{0:04d}_epoch_{1:02d}_checkpoint.pth.tar'.format(int(acc), epoch)  # assuming score=acc
        torch.save(model.state_dict(), os.path.join(self.path, filename))
        self.acc_min = acc

