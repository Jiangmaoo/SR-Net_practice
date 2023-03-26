import numbers
from collections.abc import Sequence
from typing import Tuple

import torch.nn
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor


#图像预处理组合
class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img):
        #对图片进行所有的预处理
        for t in self.transforms:
            img=t(img)
        return img

    #自定义打印类信息
    def __repr__(self):
        format_string=self.__class__.__name__+"("
        for t in self.transforms:
            format_string+="\n"
            format_string+="    {0}".format(t)
        format_string+="\n)"
        return format_string

#格式转换
class ToTensor(object):
    def __call__(self,img):
        return F.to_tensor(img[0]),F.to_tensor(img[1])

    def __repr__(self):
        return self.__class__.__name__+"()"
#缩放
class Scale(object):

    # interpolation=Image.BILINEAR时，resize()方法时使用双线性插值算法进行插值处理
    # 由于原图像和新图像像素点位置一般不完全重合，因为需要插值处理，以便在新图像上获取一个合适的像素值
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size=size
        self.interpolation=interpolation  #该实例对象的属性值
    def __call__(self,img):
        output=[]
        for i in img:
            w,h=i.size
            if(w<=h and w==self.size) or (h<=w and h==self.size):
                output.append(i)
                continue
            if w<h:
                ow=self.size
                oh=int(self.size*h/w)
                output.append(i.resize((ow,oh),self.interpolation))
                continue
            else:
                oh=self.size
                ow=int(self.size*w/h)
            output.append(i.resize((ow,oh),self.interpolation))

        return output[0],output[1]

#归一化
class Normalize(object):
    def __init__(self,mean,std,inplace=False):
        self.mean=mean #均值
        self.std=std    #方差
        self.inplace=inplace    #是否就地计算

    def __call__(self, tensor):
        return F.normalize(tensor[0],self.mean,self.std,self.inplace),\
            F.normalize(tensor[1],self.mean,self.std,self.inplace)

    def __repr__(self):
        return self.__class__.__name__+"(mean={0},std={1}".format(self.mean,self.std)

class Resize(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.size=size
    def forward(self,img):
        return F.resize(img[0],self.size),\
            F.resize(img[1],self.size)

#裁剪图像中心区域
class CenterCrop(torch.nn.Module):
    def __init__(self,size):
        super().__init__()

        if isinstance(size,numbers.Number):
            self.size=(int(size),int(size))
        elif isinstance(size,Sequence) and len(size)==1:
            self.size=(size[0],size[0])
        else:
            if len(size)!=2:
                raise ValueError("Please provide only two dimensions(h,w) for size.")
            self.size=size

    def forward(self,img):
        return F.center_crop(img[0], self.size), \
               F.center_crop(img[1], self.size)
    def __repr__(self):
        return self.__class__.__name__+"(size={0}".format(self.size)

#随机裁剪
class RandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img:Tensor,output_size:Tuple[int,int])->Tuple[int,int,int,int]:
        w,h=img.size
        th,tw=output_size

        if w==tw and h==th:
            return 0,0,h,w
        i=torch.randint(0,h-th+1,size=(1,)).item()#?
        j=torch.randint(0,w-tw+1,size=(1,)).item()

        return i,j,th,tw

    def __init__(self,size,padding=None,pad_if_needed=False,fill=1,padding_mode="constant"):
        super().__init__()

        if isinstance(size,numbers.Number):
            self.size=(int(size),int(size))
        elif isinstance(size,Sequence) and len(size)==1:
            self.size=(size[0],size[0])
        else:
            if len(size)!=2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size=tuple(size)
        self.padding=padding
        self.pad_if_needed=pad_if_needed
        self.fill=fill
        self.padding_mode=padding_mode
    def forward(self,img):
        if self.padding is not None:
            img[0]=F.pad(img[0],self.padding,self.fill,self.padding_mode)
        width,height=img[0].size

        #填充宽
        if self.pad_if_needed and width<self.size[1]:
            padding=[self.size[1]-width,0]
            img[0]=F.pad(img[0],padding,self.fill,self.padding_mode)

        if self.pad_if_needed and height<self.size[0]:
            padding=[0,self.size[1]-height]
            img[0]=F.pad(img[0],padding,self.fill,self.padding_mode)

        i,j,h,w=self.get_params(img[0],self.size)
        return F.crop(img[0],i,j,h,w),F.crop(img[1],i,j,h,w)
    def __repr__(self):
        return self.__class__.__name__+"(size={0}, padding={1})".format(self.size, self.padding)

#随机水平翻转
class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def forward(self,img):
        if torch.rand(1)<self.p:
            return F.hflip(img[0]),F.hflip(img[1])
        return img[0],img[1]
    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)

class RandomVerticalFlip(torch.nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p
    def forward(self,img):
        if torch.rand(1)<self.p:
            return F.vflip(img[0]),F.vflip(img[1])
        return img[0],img[1]
    def __repr__(self):
        return self.__class__.__name__+'(p={})'.format(self.p)

