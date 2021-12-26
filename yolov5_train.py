# -*- coding:utf-8 -*-
import time
import math
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler

from models.yolov5_model import Model
from datasets.yolov5_datasets import LoadImagesAndLabels
from utils.yolov5_utils import colorstr,one_cycle,de_parallel,labels_to_class_weights,ComputeLoss,valid_model,fitness

def train():
    # Config
    epochs=100
    plots = True
    # Image size
    imgsz=416
    # Batch size
    batch_size = 16
    evaluation_interval=1
    augment=False  # 四张拼成一张
    rect=False
    adam=True
    multi_scale=True   # 多尺度训练
    quad=False  # 四个 四张拼成一张 再拼成一张
    linear_lr=False
    cfg='configs/yolov5m.yaml'
    hyp='configs/hyp.scratch.yaml'
    images_path='data/images'
    train_cache_path=Path('data/train_labels.cache')
    valid_cache_path=Path('data/vaild_labels.cache')
    workers=2
    songs=['可爱女人', '星晴', '黑色幽默', '龙卷风', '屋顶', '爱在西元前', '简单爱', '开不了口', '上海一九四三', '双截棍', '安静', '蜗牛', '你比从前快乐', '世界末日', '半岛铁盒', '暗号', '分裂', '爷爷泡的茶', '回到过去', '最后的战役', '晴天', '三年二班', '东风破', '你听得到', '她的睫毛', '轨迹', '断了的弦', '七里香', '借口', '搁浅', '园游会', '夜曲', '发如雪', '黑色毛衣', '枫', '浪漫手机', '麦芽糖', '珊瑚海', '一路向北', '听妈妈的话', '千里之外', '退后', '心雨', '白色风车', '千山万水', '不能说的秘密', '牛仔很忙', '彩虹', '青花瓷', '阳光宅男', '蒲公英的约定', '我不配', '甜甜的', '最长的电影', '周大侠', '给我一首歌的时间', '花海', '魔术先生', '说好的幸福呢', '時光機', '乔克叔叔', '稻香', '说了再见', '好久不见', '愛的飛行日記', '超人不会飞', 'Mine Mine', '公主病', '你好吗', '疗伤烧肉粽', '水手怕水', '世界未末日', '超跑女神', '明明就', '爱你没差', '夢想啟動', '大笨钟', '傻笑', '手语', '乌克丽丽', '哪裡都是你', '算什么男人', '怎么了', '我要夏天', '手写的从前', '听爸爸的话', '美人魚', '听见下雨的声音', '说走就走', '一点点', '前世情人', '不该', '告白气球', '愛情廢柴', '等你下课', '不爱我就拉倒', '说好不哭', '我是如此相信', 'Mojito', '瓦解']
    song2label={song:i for i,song in enumerate(songs)}
    label2song={i:song for i,song in enumerate(songs)}
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device != 'cpu'
    device = torch.device('cuda:0' if cuda else 'cpu')
    nc = 100 # number of classes
    names = songs
    #model
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    # Optimizer
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=0.01, momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    print(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2
    # Scheduler
    if linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    start_epoch, best_fitness = 0, 0.0
    # Trainloader
    train_dataset=LoadImagesAndLabels(
        images_path,
        img_size=imgsz,
        batch_size=batch_size,
        is_train=True,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache_path=train_cache_path
    )
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn
    )
    valid_dataset=LoadImagesAndLabels(
        images_path,
        img_size=imgsz,
        batch_size=8,
        is_train=False,
        augment=False,
        hyp=hyp,
        cache_path=valid_cache_path
    )
    valid_loader=DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=workers,
        collate_fn=LoadImagesAndLabels.collate_fn
    )
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 100 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    t0 = time.time()
    start_epoch=0
    best_epoch=0
    nw = max(round(hyp['warmup_epochs'] * len(train_loader)), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class

    train_loss=[]
    val_loss=[]
    val_metrics=[]

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + len(train_loader) * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
        
        train_loss.append(dict(zip(['box loss', 'obj loss', 'cls loss'],mloss.cpu().data.tolist())))
        # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if (epoch % evaluation_interval == 0) or (epoch+1 == epochs):
            save_dir=Path(f'{epoch}')
            save_dir.mkdir(parents=True, exist_ok=True)
            results, maps, _ = valid_model(model,valid_loader,conf_thres=0.5,iou_thres=0.5,verbose=True,save_dir=save_dir,compute_loss=compute_loss)
            val_loss.append(dict(zip(['box loss', 'obj loss', 'cls loss'],results[4:])))
            val_metrics.append(dict(zip(['val_precision', 'val_recall', 'val_mAP@.5','val_mAP@.5:.95'],results[:4])))
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                print(f'---- fi={fi[0]}')
                best_epoch=epoch+1
                best_fitness = fi
                torch.save(model.state_dict(), f"pths/{cfg.split('/')[-1].split('.')[0]}.pth")
                
    print(f'---- best_fitness={best_fitness[0]}\n---- best_epoch={best_epoch}')
    pd.DataFrame(train_loss).to_pickle('train_loss.pkl')
    pd.DataFrame(val_loss).to_pickle('val_loss.pkl')
    pd.DataFrame(val_metrics).to_pickle('val_metrics.pkl')

if __name__ == '__main__':
    train()