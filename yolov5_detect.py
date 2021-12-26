# -*- coding:utf-8 -*-
import os
import sys
import time

import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps

from models.yolov5_model import Model
from utils.yolov5_utils import non_max_suppression


songs=['可爱女人', '星晴', '黑色幽默', '龙卷风', '屋顶', '爱在西元前', '简单爱', '开不了口', '上海一九四三', '双截棍', '安静', '蜗牛', '你比从前快乐', '世界末日', '半岛铁盒', '暗号', '分裂', '爷爷泡的茶', '回到过去', '最后的战役', '晴天', '三年二班', '东风破', '你听得到', '她的睫毛', '轨迹', '断了的弦', '七里香', '借口', '搁浅', '园游会', '夜曲', '发如雪', '黑色毛衣', '枫', '浪漫手机', '麦芽糖', '珊瑚海', '一路向北', '听妈妈的话', '千里之外', '退后', '心雨', '白色风车', '千山万水', '不能说的秘密', '牛仔很忙', '彩虹', '青花瓷', '阳光宅男', '蒲公英的约定', '我不配', '甜甜的', '最长的电影', '周大侠', '给我一首歌的时间', '花海', '魔术先生', '说好的幸福呢', '時光機', '乔克叔叔', '稻香', '说了再见', '好久不见', '愛的飛行日記', '超人不会飞', 'Mine Mine', '公主病', '你好吗', '疗伤烧肉粽', '水手怕水', '世界未末日', '超跑女神', '明明就', '爱你没差', '夢想啟動', '大笨钟', '傻笑', '手语', '乌克丽丽', '哪裡都是你', '算什么男人', '怎么了', '我要夏天', '手写的从前', '听爸爸的话', '美人魚', '听见下雨的声音', '说走就走', '一点点', '前世情人', '不该', '告白气球', '愛情廢柴', '等你下课', '不爱我就拉倒', '说好不哭', '我是如此相信', 'Mojito', '瓦解']
song2label={song:i for i,song in enumerate(songs)}
label2song={i:song for i,song in enumerate(songs)}

# Config
hyp='configs/hyp.scratch.yaml'
with open(hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)  # load hyps dict
cfg='configs/yolov5m.yaml'
argv=sys.argv
if len(argv) > 1:
    cfg=f'configs/{argv[1]}.yaml'
weights_path=f"pths/{cfg.split('/')[-1].split('.')[0]}.pth"
file_path='data/test/'
font_path='simhei.ttf'
device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = device != 'cpu'
device = torch.device('cuda:0' if cuda else 'cpu')
nc = 100 # number of classes

# load model
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# detect
files=[i for i in os.listdir(file_path) if 'png' in i]
conf_thres=0.5
iou_thres=0.5
font = ImageFont.truetype(font_path, 18)
for img_i,file in enumerate(files):
    print("(%d) Image: '%s'" % (img_i, file))
    image=Image.open(file_path+file)
    try:
        true_label=np.loadtxt(file_path+file.replace('png','txt'))[:,0].tolist()
    except:
        true_label=np.loadtxt(file_path+file.replace('png','txt'))[:1].tolist()
    image=image.resize((416,416))
    image=transforms.ToTensor()(image)
    image = image.to(device, non_blocking=True).float()
    pred=model(image.unsqueeze(0))
    pred=non_max_suppression(pred[0], conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True, agnostic=False)
    img=Image.open(file_path+file)
    scare=520/416
    image_draw=ImageDraw.Draw(img)
    pred=pred[0]
    for x1, y1, x2, y2, conf, class_ in pred.cpu().data:
#         if int(class_) in true_label:
        x1, y1, x2, y2=x1*scare, y1*scare, x2*scare, y2*scare
        pred_label=f'{label2song[int(class_)]} {conf:.2f}'
        print(pred_label)
        random_color=tuple(np.random.randint(0,256,3))
        w, h = font.getsize(pred_label)  # text width, height
        outside = y1 - h >= 0  # label fits outside box
        if not outside:
            print('hehe')
        image_draw.rectangle([x1,
                            y1 - h if outside else y1,
                            x1 + w + 1,
                            y1 + 1 if outside else y1 + h + 1], fill=random_color)
        image_draw.rectangle((x1, y1, x2, y2),width=3,outline=random_color)
        image_draw.text((x1, y1 - h if outside else y1), pred_label, fill=(255, 255, 255), font=font)
    img.save(file)