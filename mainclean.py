#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

"""

import os
import time
import torch
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
from torch import nn
import torch.optim
import numpy as np

# from utils.dataset import DataSetSeg
from utils.TFRDataset import tfr_data_loader

from utils.transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint
from statistics import mean
from utils.opts import parser
from utils import presets
from utils import engine
from utils.earlystopping import EarlyStopping
import matplotlib
# import imageio
from torch._six import inf
from torchvideotransforms import video_transforms, volume_transforms


torch.backends.cudnn.benchmark = True

global best_prec1
best_prec1 = 0
args = parser.parse_args()
video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False
disentangle_channels = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def debug_plot(img):
    from matplotlib import pyplot as plt
    import tkinter
    import matplotlib
    matplotlib.use('TkAgg')
    pl = img.cpu().numpy().transpose(0, 2, 3, 4, 1)
    plt.subplot(131);plt.imshow(pl[0, 0]);plt.subplot(132);plt.imshow(pl[0, 5]);plt.subplot(133);plt.imshow(pl[0, 10]);plt.show()


def validate(val_loader, model, criterion, device, logiters=None):
    batch_timev = AverageMeter()
    lossesv = AverageMeter()
    top1v = AverageMeter()
    precisionv = AverageMeter()
    recallv = AverageMeter()
    f1scorev = AverageMeter()


    end = time.time()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
            imgs, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels)  # noqa

            # debug_plot(imgs)
            output, jv_penalty = engine.model_step(model, imgs, model_name=args.model)
            loss = criterion(output, target.float().reshape(-1, 1))
            prec1, preci, rec, f1s = acc_scores(target, output.data)
            
            lossesv.update(loss.data.item(), 1)
            top1v.update(prec1.item(), 1)
            precisionv.update(preci.item(), 1)
            recallv.update(rec.item(), 1)
            f1scorev.update(f1s.item(), 1)
            
            batch_timev.update(time.time() - end)
            end = time.time()

            # if (i % args.print_freq == 0 or (i == len(val_loader) - 1)) and logiters is None:
            if (i % args.print_freq == 0 or (i == len_val_loader - 1)) and logiters is None:

                print_string = 'Test: [{0}/{1}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.8f} ({loss.avg: .8f})\t'\
                               'Bal_acc: {balacc:.8f} preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f}'\
                               '({rec.avg:.5f}) f1: {f1s.val:.5f} ({f1s.avg:.5f})'\
                               .format(i * args.batch_size, len_val_loader, batch_time=batch_timev, loss=lossesv, balacc=top1v.avg,
                                       preci=precisionv, rec=recallv, f1s=f1scorev)
                print(print_string)
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')

            elif logiters is not None:
                if i > logiters:
                    break
    model.train()
    return top1v.avg, precisionv.avg, recallv.avg, f1scorev.avg, lossesv.avg


def save_npz(epoch, log_dict, results_folder, savename='train'):

    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)


if __name__ == '__main__':
    
    assert args.dist is not None, "You must pass a PT distance."
    assert args.speed is not None, "You must pass a PT speed."
    assert args.length is not None, "You must pass a PT length."
    stem = "{}_{}_{}".format(args.length, args.speed, args.dist)
    pf_root, timesteps, len_train_loader, len_val_loader = engine.dataset_selector(dist=args.dist, speed=args.speed, length=args.length, optical_flow=args.optical_flow)  # 14, 1, 64

    print("Loading training dataset")
    train_loader = tfr_data_loader(data_dir=pf_root+'train-*', batch_size=args.batch_size, drop_remainder=True, timesteps=args.length)  # , optical_flow=args.optical_flow)

    print("Loading validation dataset")
    val_loader = tfr_data_loader(data_dir=pf_root+'test-*', batch_size=args.batch_size, drop_remainder=True, timesteps=args.length)  # , optical_flow=args.optical_flow)

    if args.optical_flow:
        stem = "_{}".format(stem, "flow")
    # results_folder = os.path.join('results', stem, '{0}'.format(args.name))
    results_folder = os.path.join("/cifs/data/tserre_lrs/projects/prj_tracking/pytorch_hGRU/cifs_results", stem, '{0}'.format(args.name))
    ES = EarlyStopping(patience=200, results_folder=results_folder)
    os.makedirs(results_folder, exist_ok=True)
    exp_logging = args.log
    jacobian_penalty = args.penalty

    model = engine.model_selector(args=args, timesteps=timesteps, device=device)
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    # noqa Save timesteps/kernel_size/dimensions/learning rate/epochs/exp_name/algo/penalty to a dict for reloading in the future
    param_names_shapes = {k: v.shape for k, v in model.named_parameters()}
    hp_dict = {
        "penalty": jacobian_penalty,
        "start_epoch": args.start_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "loaded_ckpt": args.ckpt,
        "results_dir": results_folder,
        "exp_name": args.name,
        "algo": args.algo,
        "dimensions": args.dimensions,
        "fb_kernel_size": args.fb_kernel_size,
        "param_names_shapes": param_names_shapes,
        "timesteps": timesteps
    }
    np.savez(os.path.join(results_folder, "hp_dict"), **hp_dict)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Including parameters {}".format([k for k, v in model.named_parameters()]))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)
    lr_init = args.lr

    val_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
    train_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 'jvpen': [], 'scaled_loss': []}

    if args.ckpt is not None:
        model = engine.load_ckpt(model, args.ckpt)
    scale = torch.Tensor([1.0]).to(device)
    for epoch in range(args.start_epoch, args.epochs):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        f1score = AverageMeter()

        time_since_last = time.time()
        model.train()
        end = time.perf_counter()

        for idx, (imgs, target) in enumerate(train_loader):
            data_time.update(time.perf_counter() - end)
            
            imgs, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels)  # noqa

            # Run training
            output, jv_penalty = engine.model_step(model, imgs, model_name=args.model)
            loss = criterion(output, target.float().reshape(-1, 1))
            losses.update(loss.data.item(), 1)
            jv_penalty = jv_penalty.mean()
            train_log_dict['jvpen'].append(jv_penalty.item())

            if jacobian_penalty:
                loss = loss + jv_penalty * 1e1
            
            prec1, preci, rec, f1s = acc_scores(target[:], output.data[:])
            top1.update(prec1.item(), 1)
            precision.update(preci.item(), 1)
            recall.update(rec.item(), 1)
            f1score.update(f1s.item(), 1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.perf_counter() - end)
            
            end = time.perf_counter()
            if idx % (args.print_freq) == 0:
                time_now = time.time()
                print_string = 'Epoch: [{0}][{1}/{2}]  lr: {lr:g}  Time: {batch_time.val:.3f} (itavg:{timeiteravg:.3f}) '\
                               '({batch_time.avg:.3f})  Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                               'Loss: {loss.val:.8f} ({lossprint:.8f}) ({loss.avg:.8f})  bal_acc: {top1.val:.5f} '\
                               '({top1.avg:.5f}) preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f} '\
                               '({rec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) jvpen: {jpena:.12f} {timeprint:.3f} losscale:{losscale:.5f}'\
                               .format(epoch, idx, len_train_loader, batch_time=batch_time, data_time=data_time, loss=losses,
                                       lossprint=mean(losses.history[-args.print_freq:]), lr=optimizer.param_groups[0]['lr'],
                                       top1=top1, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                                       timeprint=time_now - time_since_last, preci=precision, rec=recall,
                                       f1s=f1score, jpena=jv_penalty.item(), losscale=scale.item())
                print(print_string)
                time_since_last = time_now
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')
        #lr_scheduler.step()
        print(epoch)
        train_log_dict['loss'].extend(losses.history)
        train_log_dict['balacc'].extend(top1.history)
        train_log_dict['precision'].extend(precision.history)
        train_log_dict['recall'].extend(recall.history)
        train_log_dict['f1score'].extend(f1score.history)
        save_npz(epoch, train_log_dict, results_folder, 'train')
        save_npz(epoch, val_log_dict, results_folder, 'val')

        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            model.eval()
            accv, precv, recv, f1sv, losv = validate(val_loader, model, criterion, device, logiters=3)
            model.train()
            print_string = 'val f {} val loss {}'.format(f1sv, losv)
            print(print_string)
            val_log_dict['loss'].append(losv)
            val_log_dict['balacc'].append(accv)
            val_log_dict['precision'].append(precv)
            val_log_dict['recall'].append(recv)
            val_log_dict['f1score'].append(f1sv)
            with open(results_folder + args.name + '.txt', 'a+') as log_file:
                log_file.write(print_string + '\n')
            # save_checkpoint({
            #     'epoch': epoch,
            #     'state_dict': model.state_dict(),
            #     'best_acc': accv}, True, results_folder)
            ES(accv, model, epoch)
        if ES.early_stop:
            print("Early stopping triggered. Quitting.")
            os._exit(1)


