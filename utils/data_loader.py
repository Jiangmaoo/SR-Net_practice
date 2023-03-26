import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import istd_transforms


#获取训练集、验证机、测试集的路径
def make_data_path_list(phase="train",rate=0.8):
    random.seed(44)

    #去噪数据集
    root_path='dataset/'+phase+'/'
    #图片的路径
    files_name=os.listdir(root_path+phase+'_A')

    if phase=="train":
        random.shuffle(files_name)  #打乱训练数据
    elif phase=="test":
        files_name.sort()   #排序测试数据

    path_a=[]   #噪声图像
    path_c=[]   #gt

    #取出  噪声图像/gt
    if phase=="train":
        for name in files_name:
            path_a.append(root_path+phase+'_A/'+name)
            #path_c.append(root_path+phase+'_2/'+name.split("_")[0]+"_"+name.split("_")[1]+".png")
            #path_c.append(root_path+phase+'_2/'+name.split("_")[0]+".png")
            path_c.append(root_path + phase + '_c/' + name)
    elif phase=='test':
        for name in files_name:
            path_a.append(root_path+phase+'_A/'+name)
            path_c.append(root_path + phase + '_c/' + name)
    num=len(path_a)

    #分出训练集和验证集
    if phase=="train":
        path_a,path_a_val=path_a[:int(num*rate)],path_a[int(num*rate):]#前80%训练。后20%训练
        path_c,path_c_val=path_c[:int(num*rate)],path_c[int(num*rate):]

        path_list={'path_A':path_a,'path_C':path_c,}
        path_list_val={'path_A':path_a_val,'path_C':path_c_val}

        #返回测试集（路径）和验证集（路径）
        return path_list,path_list_val

    elif phase=='test':
        path_list={'path_A':path_a,'path_C':path_c}

        #返回测试机（路径）
        return path_list

class ImageTransformOwn:
    def __init__(self,size=246,mean=(0.5,),std=(0.5,)):
        self.data_transform=transforms.Compose(
            [transforms.ToTensor,transforms.Normalize(mean,std)]
        )
    def __call__(self,img):
        return self.data_transform(img)

#图片转换
class ImageTransform:
    def __init__(self,size=286,crop_size=256,mean=(0.5,),std=(0.5,)):
        self.data_transform={
            #训练集
            "train":istd_transforms.Compose([
                istd_transforms.Scale(size=size),#缩放
                istd_transforms.RandomCrop(size=crop_size),#随机裁剪
                istd_transforms.RandomHorizontalFlip(p=0.5),#水平翻转
                istd_transforms.RandomVerticalFlip(p=0.5),#竖直翻转
                istd_transforms.ToTensor(),#
                istd_transforms.Normalize(mean,std)#归一化
            ]),
            #验证集
            "val":istd_transforms.Compose([
                istd_transforms.Scale(size=size),
                istd_transforms.RandomCrop(size=crop_size),
                istd_transforms.ToTensor(),
                istd_transforms.Normalize(mean,std)
            ]),
            #测试集
            "test": istd_transforms.Compose([
                istd_transforms.Scale(size=size),
                istd_transforms.RandomCrop(size=crop_size),
                istd_transforms.ToTensor(),
                istd_transforms.Normalize(mean, std)
            ]),

            "test_no_crop": istd_transforms.Compose([
                istd_transforms.Resize([crop_size, crop_size]),
                istd_transforms.ToTensor(),
                istd_transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, phase, img):
        return self.data_transform[phase](img)

#加载图片
class ImageDataset(data.Dataset):
    def __init__(self,img_list,img_transform,phase):
        self.img_list=img_list
        self.img_transform=img_transform
        self.phase=phase

    def __len__(self):
        return len(self.img_list["path_A"])
    def __getitem__(self, index):
        img=Image.open(self.img_list["path_A"][index]).convert("RGB")
        #gt_shadow=Image.open(self.img_list["path_B"](index))
        gt=Image.open(self.img_list["path_C"][index]).convert("RGB")
        img,gt=self.img_transform(self.phase,[img,gt])

        return img,gt
if __name__=='__main__':
    img=Image.open('../dataset/train/train_A/test.png').convert('RGB')
    gt = Image.open('../dataset/train/train_c/test.png').convert('RGB')

    print(img.size)
    print(gt.size)

    f=plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img)
    f.add_subplot(1, 3, 2)
    # plt.imshow(gt_shadow, cmap='gray')
    # f.add_subplot(1, 3, 3)
    plt.imshow(gt)

    img_transforms=ImageTransform(size=286,crop_size=256,mean=(0.5,),std=(0.5,))
    img,gt=img_transforms[img,gt]

    print(img.shape)
    print(gt.img)

    f.add_subplot(2, 3, 4)
    plt.imshow(transforms.ToPILImage()(img).convert('RGB'))
    f.add_subplot(2, 3, 5)
    # plt.imshow(transforms.ToPILImage()(gt_shadow).convert('L'), cmap='gray')
    # f.add_subplot(2, 3, 6)
    plt.imshow(transforms.ToPILImage()(gt).convert('RGB'))
    f.tight_layout()
    plt.show()





