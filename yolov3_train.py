# -*- coding:utf-8 -*-
import datetime
import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from terminaltables import AsciiTable

from utils.yolov3_utils import weights_init_normal,evaluate
from datasets.yolov3_datasets import myDataset
from models.yolov3_model import Darknet

model_def='configs/yolov3.cfg'
data_path='./data'
epochs=100
batch_size=16
gradient_accumulations=5
workers=2
img_size=416
evaluation_interval=1
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ['可爱女人', '星晴', '黑色幽默', '龙卷风', '屋顶', '爱在西元前', '简单爱', '开不了口', '上海一九四三', '双截棍', '安静', '蜗牛', '你比从前快乐', '世界末日', '半岛铁盒', '暗号', '分裂', '爷爷泡的茶', '回到过去', '最后的战役', '晴天', '三年二班', '东风破', '你听得到', '她的睫毛', '轨迹', '断了的弦', '七里香', '借口', '搁浅', '园游会', '夜曲', '发如雪', '黑色毛衣', '枫', '浪漫手机', '麦芽糖', '珊瑚海', '一路向北', '听妈妈的话', '千里之外', '退后', '心雨', '白色风车', '千山万水', '不能说的秘密', '牛仔很忙', '彩虹', '青花瓷', '阳光宅男', '蒲公英的约定', '我不配', '甜甜的', '最长的电影', '周大侠', '给我一首歌的时间', '花海', '魔术先生', '说好的幸福呢', '時光機', '乔克叔叔', '稻香', '说了再见', '好久不见', '愛的飛行日記', '超人不会飞', 'Mine Mine', '公主病', '你好吗', '疗伤烧肉粽', '水手怕水', '世界未末日', '超跑女神', '明明就', '爱你没差', '夢想啟動', '大笨钟', '傻笑', '手语', '乌克丽丽', '哪裡都是你', '算什么男人', '怎么了', '我要夏天', '手写的从前', '听爸爸的话', '美人魚', '听见下雨的声音', '说走就走', '一点点', '前世情人', '不该', '告白气球', '愛情廢柴', '等你下课', '不爱我就拉倒', '说好不哭', '我是如此相信', 'Mojito', '瓦解']
# Initiate model
model = Darknet(model_def,img_size).to(device)
model.apply(weights_init_normal)

dataset = myDataset(img_size=img_size,data_path=data_path,is_train=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
)
optimizer = optim.Adam(model.parameters())
metrics = ["grid_size","loss","x","y","w","h","conf","cls","cls_acc","recall50","recall75","precision","conf_obj","conf_noobj"]
train_loss=[] # to plot yolo layer loss
val_loss=[] # to plot yolo layer loss
val_metrics=[] # to plot metrics
best_map=0
best_epoch=0
for epoch in range(epochs):
    yolo0_loss=np.zeros(len(dataloader))
    yolo1_loss=np.zeros(len(dataloader))
    yolo2_loss=np.zeros(len(dataloader))
    model.train()
    start_time = time.time()
    for batch_i, (imgs, targets) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        loss.backward()

        if batches_done % gradient_accumulations:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        # ----------------
        #   Log progress
        # ----------------

        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        for i,yolo in enumerate(model.yolo_layers):
            eval(f'yolo{i}_loss')[batch_i] = yolo.metrics.get('loss', 0)

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)

    print('Epoch', epoch, evaluation_interval)

    if (epoch % evaluation_interval == 0) or (epoch+1 == epochs):
        print("\n---- Evaluating Model ----")
        # Evaluate the model on the validation set
        result = evaluate(
            model,
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=img_size,
            data_path=data_path,
            batch_size=16,
            device=device
        )
        if result:
            precision, recall, AP, f1, ap_class, losses = result
            val_metric = {
                "val_precision": precision.mean(),
                "val_recall": recall.mean(),
                "val_mAP": AP.mean(),
                "val_f1": f1.mean(),
            }
            val_metrics.append(val_metric)
            val_loss.append(losses)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            now_map=AP.mean()
            print(f"---- now_map {now_map}")
            if now_map > best_map:
                best_map=now_map
                best_epoch=epoch+1
                torch.save(model.state_dict(), f"pths/yolov3.pth")

print(f"---- best_map {best_map}\n---- best_epoch {best_epoch}")
pd.DataFrame(train_loss).to_pickle('train_loss.pkl')
pd.DataFrame(val_loss).to_pickle('val_loss.pkl')
pd.DataFrame(val_metrics).to_pickle('val_metrics.pkl')