# -*- coding:utf-8 -*-
import time
import datetime
from pathlib import Path

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from datasets.yolov3_datasets import ImageFolder
from utils.yolov3_utils import non_max_suppression
from models.yolov3_model import Darknet


songs=['可爱女人', '星晴', '黑色幽默', '龙卷风', '屋顶', '爱在西元前', '简单爱', '开不了口', '上海一九四三', '双截棍', '安静', '蜗牛', '你比从前快乐', '世界末日', '半岛铁盒', '暗号', '分裂', '爷爷泡的茶', '回到过去', '最后的战役', '晴天', '三年二班', '东风破', '你听得到', '她的睫毛', '轨迹', '断了的弦', '七里香', '借口', '搁浅', '园游会', '夜曲', '发如雪', '黑色毛衣', '枫', '浪漫手机', '麦芽糖', '珊瑚海', '一路向北', '听妈妈的话', '千里之外', '退后', '心雨', '白色风车', '千山万水', '不能说的秘密', '牛仔很忙', '彩虹', '青花瓷', '阳光宅男', '蒲公英的约定', '我不配', '甜甜的', '最长的电影', '周大侠', '给我一首歌的时间', '花海', '魔术先生', '说好的幸福呢', '時光機', '乔克叔叔', '稻香', '说了再见', '好久不见', '愛的飛行日記', '超人不会飞', 'Mine Mine', '公主病', '你好吗', '疗伤烧肉粽', '水手怕水', '世界未末日', '超跑女神', '明明就', '爱你没差', '夢想啟動', '大笨钟', '傻笑', '手语', '乌克丽丽', '哪裡都是你', '算什么男人', '怎么了', '我要夏天', '手写的从前', '听爸爸的话', '美人魚', '听见下雨的声音', '说走就走', '一点点', '前世情人', '不该', '告白气球', '愛情廢柴', '等你下课', '不爱我就拉倒', '说好不哭', '我是如此相信', 'Mojito', '瓦解']
song2label={song:i for i,song in enumerate(songs)}
label2song={i:song for i,song in enumerate(songs)}

conf_thres=0.5
nms_thres=0.5
cls_conf_thres=0.5
img_size=416
workers=2
device = "cuda" if torch.cuda.is_available() else "cpu"
model_def='configs/yolov3.cfg'
weights_path='pths/yolov3.pth'
font_path='simhei.ttf'
folder_path='data/test'

model = Darknet(model_def,img_size).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()  # Set in evaluation mode
dataloader = DataLoader(
    ImageFolder(img_size=img_size,folder_path=folder_path),
    batch_size=1,
    shuffle=False,
    num_workers=workers
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
sizes=[]
imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print("\nPerforming object detection:")
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    sizes.append(input_imgs[0].shape[-1])

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres, cls_conf_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

for img_i, (size,path, detections) in enumerate(zip(sizes,imgs, img_detections)):
    print("(%d) Image: '%s'" % (img_i, path))
    # Create plot
    img = Image.open(path)
    font = ImageFont.truetype(font_path, 18)
    image_draw=ImageDraw.Draw(img)
    scare=520/size
    try:
        true_label=np.loadtxt(path.replace('png','txt'))[:,0].tolist()
    except:
        true_label=np.loadtxt(path.replace('png','txt'))[:1].tolist()

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
#             if int(cls_pred) in true_label:
            x1,y1,x2,y2=x1*scare,y1*scare,x2*scare,y2*scare
            random_color=tuple(np.random.randint(0,256,3))
            pred_label=f'{label2song[int(cls_pred)]} {cls_conf:.2f}'
            print(pred_label)
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
        filename = Path(path).stem
        img.save(f"{filename}.png")