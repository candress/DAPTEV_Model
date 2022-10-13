# -*- coding: utf-8 -*-
# Code adapted from:
# Shubham Patel
# https://github.com/bayeslabs/genmol
# https://blog.bayeslabs.co/2019/06/04/All-you-need-to-know-about-Vae.html
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import math


# 30 epochs seems to be more than enough to train the VAE and reduce the loss function.
# Any more epochs just results in variations of the 3 loss components and doesn't reduce much more.
n_batch = 32
score_threshold = 0
kl_start = 0
kl_w_start = 0.001
kl_w_end = 0.1
n_workers = 0

clip_grad  = 50
lr_start = 0.003
lr_n_period = 100
lr_n_mult = 1
lr_end = 3 * 1e-4

d = None
writer = None

class Trainer():
    def __init__(self, data_module, num_epoch, batch_amnt, score_threshold_to_binary, 
                starting_learn_rate = 0.003, ending_learn_rate = 3 * 1e-4, starting_KL_weight = 0.001, 
                ending_KL_weight = 0.1, regress_choice = 0):
        global d
        global n_batch
        global score_threshold
        global lr_n_period
        global lr_start
        global lr_end
        global kl_w_start
        global kl_w_end

        d = data_module
        self.n_epoch = num_epoch
        n_batch = batch_amnt
        score_threshold = score_threshold_to_binary
        lr_n_period = num_epoch
        lr_start = starting_learn_rate
        lr_end = ending_learn_rate
        kl_w_start = starting_KL_weight
        kl_w_end = ending_KL_weight
        self.regression_choice = regress_choice

  
    def _train_epoch(self, model, epoch, train_loader, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()
        
        # Create 2D numpy arrays to store the following values.
        # Has add, last, and mean methods
        kl_loss_values = Running_Data()
        recon_loss_values = Running_Data()
        score_predict_loss_values = Running_Data()
        loss_values = Running_Data()

        # For batch data and associated score in training loader (train_loader is a reference to the collate function [generator])
        # Note: the train_loader is actually a Torch.DataLoader which is pulling from the custom collate function
        for input_batch, score_batch in train_loader:
            # Iterate through each tensor in batch and send them to their device.
            # Then convert the entire input batch from a list of tensors to a tuple of tensors.
            input_batch = tuple(data.to(d.device) for data in input_batch)
            if self.regression_choice == 0:
                score_batch = [score.fill_(1) if score.item() <= score_threshold else score.fill_(0) for score in score_batch]
            else:
                score_batch = [score.fill_(score.item()/input_batch[i].size(0)) for i, score in enumerate(score_batch)]

        #forward
            # Note, calling "model(input_batch)" is a pytorch implimentation that uses the model's forward method
            kl_loss, prediction_loss, recon_loss = model(input_batch, score_batch)
            loss = kl_weight * kl_loss + recon_loss + prediction_loss
            # loss = kl_weight * kl_loss + recon_loss
            
        #backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(get_optim_params(model), clip_grad)
                optimizer.step()

        # update running log
            kl_loss_values.add(kl_loss.item())
            score_predict_loss_values.add(prediction_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())

    # Calculate the epoch's average values
        lr = (optimizer.param_groups[0]['lr'] if optimizer is not None else None)
        kl_loss_value = kl_loss_values.mean()
        reg_loss_value = score_predict_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'regularizer_loss': reg_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix
    

    def _train(self, model, train_loader, summary_writer):
        global lr_start
        global kl_w_start

        optimizer = optim.Adam(get_optim_params(model), lr = lr_start)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer)
        model.zero_grad()

        for epoch in tqdm(range(self.n_epoch), desc="VAE Epoch"):
            kl_annealer = KLAnnealer(self.n_epoch)
            kl_weight = kl_annealer(epoch)
            self.postfix = self._train_epoch(model, epoch, train_loader, kl_weight, optimizer)
            fill_tensorboard(self.postfix, summary_writer)
            # print({key:(f'{value:.7f}' if isinstance(value, (float, int)) and key != 'epoch' else value) for key, value in postfix.items()})
            lr_annealer.step()
        
        # setting these values again at the end to guarantee they are set to be static for next generation (gen, not epoch)
        lr_start = self.postfix['lr']
        kl_w_start = self.postfix['kl_weight']


    def fit(self, model, train_data, summary_writer):
        train_loader = self.get_dataloader(model, train_data)
        self._train(model, train_loader, summary_writer)
        return model


    def get_dataloader(self, model, train_data, collate_fn=None, shuffle=True):
        if collate_fn is None:
            # variable that holds a reference to the function that creates tensors and batches data
            collate_fn = self.get_collate_fn(model)
        return DataLoader(train_data, batch_size = n_batch, shuffle = shuffle, num_workers = n_workers, collate_fn = collate_fn)


    def get_collate_fn(self, model):
        device = get_collate_device(model)

        def collate(train_data):
            # sort the training date according to sequence length in descending order/largest first (reverse)
            train_data.sort(key=lambda x: len(x[0]), reverse=True)
            scores = [d.target_to_tensor(seq_score[1], device=device) for seq_score in train_data]
            # build a list of tensors with device option already specified (but not sent)
            tensors = [d.string_to_tensor(seq_score[0], device=device) for seq_score in train_data]
            return tensors, scores

        return collate


def get_collate_device(model):
    return model.device


def get_optim_params(model):
    return (p for p in model.parameters() if p.requires_grad)



def fill_tensorboard(postfix, writer):
    epoch = postfix['epoch']
    title_prefix = 'AB'
    writer.add_scalar(f'{title_prefix}_VAE/A_Encoder_KL_Loss', postfix['kl_loss'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/B_Regularization_Loss', postfix['regularizer_loss'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/C_Decoder_Reconstruction_Loss', postfix['recon_loss'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/D_Total_Loss', postfix['loss'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/Z_Encoder_KL_Loss_x_Weight', postfix['kl_loss'] * postfix['kl_weight'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/Z_LR', postfix['lr'], epoch)
    writer.add_scalar(f'{title_prefix}_VAE/Z_KL_weight', postfix['kl_weight'], epoch)



class KLAnnealer:
    def __init__(self,n_epoch):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch
        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc



class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self , optimizer):
        self.n_period = lr_n_period
        self.n_mult = lr_n_mult
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end
            
            
            
class Running_Data:
    def __init__(self):
        self.data = []
        self.size = 0

    def add(self, element):
        self.data.append(element)
        self.size = len(self.data)
        return element

    def last(self):
        assert len(self.data) != 0, "Can't get an element from an empty list!"
        return self.data[-1]

    def mean(self):
        assert len(self.data) != 0, "Can't calculate mean from an empty list!"
        return sum(self.data)/len(self.data)
      


