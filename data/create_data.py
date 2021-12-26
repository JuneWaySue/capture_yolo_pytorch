from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
import shutil,os
import numpy as np

class CreateData:
    def __init__(self,create_num):
        self.jay_img_paths=['JAY/' + i for i in os.listdir('JAY/')] # 背景图片路径
        self.font_path='../simhei.ttf' # 字体路径
        self.img_save_path='images/' # 生成训练集图片的路径
        self.label_save_path='labels/' # 生成图片对应label的路径
        self.test_path='test/' # 生成测试集图片的路径
        # 100首周杰伦歌曲名称
        self.songs=['可爱女人', '星晴', '黑色幽默', '龙卷风', '屋顶', '爱在西元前', '简单爱', '开不了口', '上海一九四三', '双截棍', '安静', '蜗牛', '你比从前快乐', '世界末日', '半岛铁盒', '暗号', '分裂', '爷爷泡的茶', '回到过去', '最后的战役', '晴天', '三年二班', '东风破', '你听得到', '她的睫毛', '轨迹', '断了的弦', '七里香', '借口', '搁浅', '园游会', '夜曲', '发如雪', '黑色毛衣', '枫', '浪漫手机', '麦芽糖', '珊瑚海', '一路向北', '听妈妈的话', '千里之外', '退后', '心雨', '白色风车', '千山万水', '不能说的秘密', '牛仔很忙', '彩虹', '青花瓷', '阳光宅男', '蒲公英的约定', '我不配', '甜甜的', '最长的电影', '周大侠', '给我一首歌的时间', '花海', '魔术先生', '说好的幸福呢', '時光機', '乔克叔叔', '稻香', '说了再见', '好久不见', '愛的飛行日記', '超人不会飞', 'Mine Mine', '公主病', '你好吗', '疗伤烧肉粽', '水手怕水', '世界未末日', '超跑女神', '明明就', '爱你没差', '夢想啟動', '大笨钟', '傻笑', '手语', '乌克丽丽', '哪裡都是你', '算什么男人', '怎么了', '我要夏天', '手写的从前', '听爸爸的话', '美人魚', '听见下雨的声音', '说走就走', '一点点', '前世情人', '不该', '告白气球', '愛情廢柴', '等你下课', '不爱我就拉倒', '说好不哭', '我是如此相信', 'Mojito', '瓦解']
        self.song2label={song:i for i,song in enumerate(self.songs)}
        self.label2song={i:song for i,song in enumerate(self.songs)}
        self.create_num=create_num
        self.image_w=520
        self.image_h=520
        self.max_iou=0.5  # 一张图中每首歌名之间的boxes的iou不能超过0.5
        
    def create_folder(self):
        while True:
            try:
                for path in [self.img_save_path,self.label_save_path,self.test_path]:
                    shutil.rmtree(path,ignore_errors=True)
                    os.makedirs(path,exist_ok=True)
                break
            except:
                pass
        
    def bbox_iou(self,box2):
        '''两两计算iou'''
        for box1 in self.tmp_boxes:
            inter_x1=max([box1[0],box2[0]])
            inter_y1=max([box1[1],box2[1]])
            inter_x2=min([box1[2],box2[2]])
            inter_y2=min([box1[3],box2[3]])
            inter_area=(inter_x2-inter_x1+1) * (inter_y2-inter_y1+1)  # +1是为了避免等于0
            box1_area=(box1[2]-box1[0]+1) * (box1[3]-box1[1]+1)  # +1是为了避免等于0
            box2_area=(box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)  # +1是为了避免等于0
            iou=inter_area / (box1_area + box2_area - inter_area + 1e-16)  # +1e-16是为了避免除以0
            if iou > self.max_iou:
                # 只要有一个与之的iou大于阈值则重新来过
                return iou
        else:
            return 0

    def draw_text(self,image,image_draw,song):
        iou=np.inf
        num=0
        while iou > self.max_iou:
            if num >= 100:
                # 为了避免陷进死循环，如果循环100次都没有找到合适的位置，则iou>0.5的阈值失效
                break
            random_font_size=np.random.randint(40,50) # 随机字号
            random_rotate=np.random.randint(-90,90) # 随机旋转角度
            random_color=np.random.randint(0,256,3) # 随机字体颜色
            random_x,random_y=np.random.randint(1,520,2) # 随机xmin,ymin
            
            font = ImageFont.truetype(self.font_path, random_font_size)
            label=self.song2label[song]
            size_wh=font.getsize(song)
            
            img = Image.new('L', size_wh)
            img_draw = ImageDraw.Draw(img)
            img_draw.text((0, 0), song, font=font, fill=255)
            img_rotate = img.rotate(random_rotate, resample=2, expand=True)
            img_color = ImageOps.colorize(img_rotate, (0,0,0), random_color)
            w,h=img_color.size
            xmin=random_x
            ymin=random_y
            # 为了避免超出520x520，修正xmin,ymin
            if random_x+w > self.image_w:
                xmin=self.image_w - w - 2
            if random_y+h > self.image_h:
                ymin=self.image_h - h - 2
            xmax=xmin+w
            ymax=ymin+h
            boxes=(xmin,ymin,xmax,ymax)
            
            iou=self.bbox_iou(boxes)
            num+=1
        image.paste(img_color, box=(xmin,ymin), mask=img_rotate)
#         image_draw.rectangle(boxes,outline=tuple(random_color))  # 把矩形框label画出来
        return image,boxes,label
    
    def process(self,boxes):
        '''
        将xmin,ymin,xmax,ymax转为x,y,w,h
        以及归一化坐标，生成label
        '''
        x1,y1,x2,y2=boxes
        x=((x1+x2)/2)/self.image_w
        y=((y1+y2)/2)/self.image_h
        w=(x2-x1)/self.image_w
        h=(y2-y1)/self.image_h
        return [x,y,w,h]
        
    def main(self):
        '''主函数'''
        self.create_folder() # 重置所需文件夹
        for i in tqdm(range(self.create_num+3)):
            random_song_num=np.random.randint(1,5) # 随机1~4首
            random_jay_img_path=np.random.choice(self.jay_img_paths) # 随机背景
            image=Image.open(random_jay_img_path).convert('RGB').resize((self.image_w,self.image_h))
            image_draw=ImageDraw.Draw(image)
            boxes_list=[]
            label_list=[]
            self.tmp_boxes=[] # 用于计算两两boxes的iou
            for j in range(random_song_num):
                song=np.random.choice(self.songs)
                image,boxes,label=self.draw_text(image,image_draw,song)
                self.tmp_boxes.append(boxes)
                boxes_list.append(self.process(boxes))
                label_list.append(label)
                
            # save image and label
            image_filename=self.img_save_path+f'image{i}.png' if i < self.create_num else self.test_path+f'test{i}.png'
            label_filename=self.label_save_path+f'image{i}.txt' if i < self.create_num else self.test_path+f'test{i}.txt'
            image.save(image_filename)
            with open(label_filename,'w') as f:
                for k in range(len(label_list)):
                    # label x y w h
                    f.write(f'{label_list[k]} {boxes_list[k][0]} {boxes_list[k][1]} {boxes_list[k][2]} {boxes_list[k][3]}\n')
                        
if __name__ == '__main__':
    creator=CreateData(5000)
    creator.main()