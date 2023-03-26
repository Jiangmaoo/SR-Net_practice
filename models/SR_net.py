import math

import torch
from torch import nn


#初始化权重
def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname=m.__class__.__name__
        if (classname.find("Conv")==0 or classname.find("Linear")==0 or hasattr(m,"weight")):
            if init_type=="gaussian":
                nn.init.normal_(m.weight,0.0,0.02)
            elif init_type=="xavier":
                nn.init.xavier_normal_(m.weight,gain=math.sqrt(2))
            elif init_type=="kaiming":
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
            elif init_type=="orthogonal":
                nn.init.orthogonal_(m.weight,gain=math.sqrt(2))
            elif init_type=="default":
                pass
            else:
                assert 0,"Unsupported initialization:{}".format(init_type)
            if hasattr(m,"bias") and m.bias is not None:
                nn.init.constant_(m.bias,0.0)
    return init_fun

#卷积
class Cvi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(Cvi,self).__init__()

        #初始化卷积
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

        #初始化卷积参数
        self.conv.apply(weights_init("gaussian"))

        #卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)   #归一化
        elif after=="Tanh":
            self.after=torch.tanh #tanh激活函数（-1到1S型）
        elif after=="sigmoid":
            self.after=torch.sigmoid    #sigmoid激活函数（0到1S型）

        #卷积前进行的操作
        if before=="ReLU":
            self.after=nn.ReLU(inplace=True)  #ReLU激活函数（<0时=0；>0时等于自身)(inplace=True,节省反复申请与释放内存的空间和时间)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=False)  #LeakyReLu激活函数（<0时斜率为0.2）

    def forward(self,x):
        if hasattr(self,"before"):
            x=self.before(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x
#逆卷积
class CvTi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(CvTi, self).__init__()

        #初始化逆卷积
        self.conv=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                     tride=stride,padding=padding)
        #初始化逆卷积权重
        self.conv.apply(weights_init("gaussian"))

        # 卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)
        elif after=="Tanh":
            self.after=torch.tanh
        elif after=="sigmoid":
            self.after=torch.sigmoid

        #卷积前进行的操作
        if before=="ReLU":
            self.before=nn.ReLU(inplace=True)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        if hasattr(self,"before"):
            x=self.conv(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x

#编码器
class Encoder(nn.Module):
    def __init__(self,input_channels=3):
        super(Encoder, self).__init__()
        self.cv0=Cvi(input_channels,64)
        self.cv1=Cvi(64,128,before="LReLU",after="BN")
        self.cv2=Cvi(128,256,before="LReLu",after="BN")
        self.cv3=Cvi(256,512,before="LReLU",after="BN")
        self.cv4 = Cvi(512, 512, before="LReLU", after="BN")
        self.cv5 = Cvi(512, 512, before="LReLU")

    def forward(self,x):
        x0=self.cv0(x)
        x1=self.cv1(x0)
        x2=self.cv2(x1)
        x3=self.cv3(x2)
        x4_1=self.cv4(x3)
        x4_2=self.cv4(x4_1)
        x4_3=self.cv4(x4_2)
        x5=self.cv5(x4_3)

        feature_dic={
            "x0":x0,
            "x1":x1,
            "x2":x2,
            "x3":x3,
            "x4_1":x4_1,
            "x4_2":x4_2,
            "x4_3":x4_3,
            "x5":x5,
        }

        return feature_dic

#解码器
class Decoder(nn.Module):
    def __init__(self,output_channels=1):
        super(Decoder, self).__init__()
        self.cvt6=CvTi(512,512,before="ReLU",after="BN")
        self.cvt7 = CvTi(1024, 512, before="ReLU", after="BN")
        self.cvt8 = CvTi(1024, 256, before="ReLU", after="BN")
        self.cvt9 = CvTi(512, 128, before="ReLU", after="BN")
        self.cvt10 = CvTi(256, 64, before="ReLU", after="BN")
        self.cvt11 = CvTi(128,output_channels, before="ReLU", after="Tanh")

    def forward(self,feature_dic):
        x6=self.cvt6(feature_dic["x5"])

        cat1_1=torch.cat([x6,feature_dic["x4_3"]],dim=1)
        x7_1=self.cvt7(cat1_1)
        cat1_2=torch.cat([x7_1,feature_dic["x4_2"]],dim=1)
        x7_2=self.cvt7(cat1_2)
        cat1_3 = torch.cat([x7_2, feature_dic["x4_1"]], dim=1)
        x7_3 = self.cvt7(cat1_3)

        cat2 = torch.cat([x7_3, feature_dic["x3"]], dim=1)
        x8 = self.cvt8(cat2)

        cat3 = torch.cat([x8, feature_dic["x2"]], dim=1)
        x9 = self.cvt9(cat3)

        cat4 = torch.cat([x9, feature_dic["x1"]], dim=1)
        x10 = self.cvt10(cat4)

        cat5 = torch.cat([x10, feature_dic["x0"]], dim=1)
        out = self.cvt11(cat5)

        return out

#联合解码器
class JointDecoder(nn.Module):
    def __init__(self,input_channels=3,output_channels=3):
        super(JointDecoder, self).__init__()
        self.cvt6=CvTi(1024,1024,before="ReLu",after="BN")
        self.cvt7 = CvTi(2048, 1024, before="ReLu", after="BN")
        self.cvt8 = CvTi(2048, 512, before="ReLu", after="BN")
        self.cvt9 = CvTi(1024, 256, before="ReLu", after="BN")
        self.cvt10 = CvTi(512, 128, before="ReLu", after="BN")
        self.cvt11 = CvTi(256, output_channels, before="ReLu", after="Tanh")

    def forward(self,f_dic1,f_dic2):
        cat_0=torch.cat([f_dic1["x5"],f_dic2["x5"]],1)
        x6=self.cvt6(cat_0) #channel=1024

        cat1_1=torch.cat([x6,f_dic1["x4_3"],f_dic2["x4_3"]],dim=1)
        x7_1=self.cvt7(cat1_1)
        cat1_2 = torch.cat([x7_1, f_dic1["x4_2"], f_dic2["x4_2"]], dim=1)
        x7_2 = self.cvt7(cat1_2)
        cat1_3 = torch.cat([x7_2, f_dic1["x4_1"], f_dic2["x4_1"]], dim=1)
        x7_3 = self.cvt7(cat1_3)

        cat2 = torch.cat([x7_3, f_dic1["x3"], f_dic2["x3"]], dim=1)
        x8 = self.cvt8(cat2)

        cat3 = torch.cat([x8, f_dic1["x2"], f_dic2["x2"]], dim=1)
        x9 = self.cvt9(cat3)

        cat4 = torch.cat([x9, f_dic1["x1"], f_dic2["x1"]], dim=1)
        x10 = self.cvt10(cat4)

        cat5 = torch.cat([x10, f_dic1["x0"], f_dic2["x0"]], dim=1)
        out = self.cvt11(cat5)

        return out
#SR-Net
class SRNet(nn.Module):
    def __init__(self,input_channels=3,output_channels=3):
        super(SRNet, self).__init__()

        #阴影编码器Es 和无阴影编码器Esf
        self.domain1_encoder=Encoder(input_channels)
        self.domain2_encoder=Encoder(input_channels)

        #共同特征编码器Ec
        self.general_encoder=Encoder(input_channels)

        #阴影-共同特征解码器Js/无阴影-共同特征解码器Jsf
        #一个解码器
        self.joint_decoder=JointDecoder(output_channels)

        #阴影移除联合解码器Jssf
        self.joint_decoderT=JointDecoder(output_channels)

        self.placeholder=None

    def forward(self,input_img,gt):
        #分别用Es和Esf对输入图片和GT进行编码
        feature_dic1=self.domain1_encoder(input_img)  #阴影编码器Es
        feature_dic2=self.domain2_encoder(gt)   #无阴影编码器Esf

        #使用Ec对输入图片和GT进行编码
        general_dic1=self.general_encoder(input_img)
        general_dic2=self.general_encoder(gt)

        #使用Js重构阴影图片
        reconstruct_input=self.joint_decoder(feature_dic1,general_dic1)

        #使用Jsf重构无阴影图片
        reconstruct_gt = self.joint_decoder(feature_dic2, general_dic2)

        #使用Js2sf将阴影图片重构为无阴影图片
        reconstruct_tf=self.joint_decoderT(feature_dic1,general_dic1)

        #返回  重构阴影图像  重构阴影去除图像  重构无阴影图像
        return reconstruct_input,reconstruct_tf,reconstruct_gt

    #模型测试
    def test(self,input_img):
        feature_dic1=self.domain1_encoder(input_img)    #阴影编码器
        general_dic1=self.general_encoder(input_img)    #共同特征编码器
        reconstruct_input=self.joint_decoderT(feature_dic1,general_dic1)    #阴影移除联合解码器Js2sf

        return reconstruct_input

    def test_pair(self,input_img):
        feature_dic1=self.domain1_encoder(input_img)
        general_dic1=self.general_encoder(input_img)

        if self.placeholder is None or self.placeholder["x1"].size(0)!=feature_dic1["x1"].size(0):
            self.placeholder={}
            for key in feature_dic1.keys():
                self.placeholder[key]=torch.zeros(feature_dic1[key].shape,requires_grad=False).to(
                    torch.device(feature_dic1["x1"].device)
                )
        # rec_by_tg1=self.joint_decoderT(self.placeholder,general_dic1)
        # gan里面的解码器
        # rec_by_td1=self.joint_decoderT(feature_dic1,self.placeholder)
        rec_by_tg1 = self.joint_decoder(self.placeholder, general_dic1)
        #重建里面那个解码器
        rec_by_td1 = self.joint_decoder(feature_dic1, self.placeholder)

        reconstruct_tf=self.joint_decoderT(feature_dic1,general_dic1)

        return reconstruct_tf,rec_by_tg1,rec_by_td1

#鉴别器
class Discriminator(nn.Module):
    def __init__(self,input_channels=4):
        super(Discriminator, self).__init__()
        self.cv0=Cvi(input_channels,64)
        self.cv1=Cvi(64,128,before="LReLU",after="BN")
        self.cv2 = Cvi(128, 256, before="LReLU", after="BN")
        self.cv3 = Cvi(256, 512, before="LReLU", after="BN")
        self.cv4 = Cvi(512, 1, before="LReLU", after="sigmoid")

    def forward(self,x):
        x0=self.cv0(x)
        x1=self.cv1(x0)
        x2=self.cv2(x1)
        x3=self.cv3(x2)
        out=self.cv4(x3)

        return out

if __name__=='__main__':
    #BCHW
    size=(3,3,256,256)
    input1=torch.ones(size)
    input2=torch.ones(size)
    l1=nn.L1Loss()

    size(3,3,256,256)
    input=torch.ones(size)
