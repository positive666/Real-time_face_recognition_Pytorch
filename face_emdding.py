from facenet_pytorch import  InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from centerface import CenterFace, FaceExtract
import argparse

# creat a new database
# 要保证数据库中的图片只有当前人脸
def CreatNewDB(weights='./centerface.onnx',source='./data_DB/', feature_path='./data_DB/', cropped_save_pth=None, device='', conf_thres=0.5):
    # 定义了一个数据集
    # 读取路径下的所有文件夹，每一个文件夹为一个类别，有对应的标签index
    # 所有文件夹下的图片按顺序都存储在dataset中，dataset为可遍历对象,每张图片的格式为元祖（data,label）
    dataset = datasets.ImageFolder(source)
    print(device)
    # dataset.class_to_idx将类别转化为数字索引
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    # 定义一个数据加载器,一张图片表示为（data，label）的元祖
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    names = []
    save_names=[]
    for name in dataset.class_to_idx.keys():
        names.append(name)
    # 保存所有剪裁过后的人脸，用于求特征向量
    print(names)
    cropped_faces = []
    # 每个类别的人脸数
    num_each_class =5
    root_num=[]
    # image为图片，index为name对应的标签
    j = 0
    for image, index in loader:
        centerface = CenterFace("./centerface.onnx")
        w, h = image.size
        boxes, lms = centerface(image, h, w, threshold=0.5)
        print(f"{names[index]} 检测人脸数：{len(boxes)}")
        extractFace = FaceExtract()
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        #save_names.append(names[index])
        # j表示一个name下的第几张图片

        # face number id 
        for i, box in enumerate(boxes):
            box, score = box[:4], box[4]
            # save cropped face,创建文件夹
            path = cropped_save_pth + dataset.idx_to_class[index]
            if not os.path.exists(path):
                os.makedirs(path)
            save_path = path + f'/{j}.jpg'
            # face为tensor
            face = extractFace(image, box=box, save_path=save_path)
            cropped_faces.append(face)
            # tensor([[...]])
            #face_embedding = resnet(face.unsqueeze(0)).detach()
            # 每张图片只保存一个人脸向量
            #embedings.append(face_embedding[0])
        j += 1
        
        # break
        root_num.append(j) 
        print(j)
        if j % num_each_class == 0:
            j = 0 
        #num_each_class=j 
        
       #j=0        
        #print(j)        
    aligned = torch.stack(cropped_faces).to(device)
    # 返回所有人脸的特征向量，每个向量的长度是512维
    embedings = resnet(aligned).detach().cpu()
    #num_each_class=j
    #print(num_each_class)
    # all nums
    # [tensor([...])]
    mean_embedings = []
    print("all face number s：",len(dataset.idx_to_class))
    for i in range (len(dataset.idx_to_class)):
        emd = embedings[i * num_each_class:(i + 1) * num_each_class].mean(0)    # 
        mean_embedings.append(emd)
    
    # dicts = [[(e1 - e2).norm().item() for e2 in mean_embedings] for e1 in mean_embedings]
    # print(pd.DataFrame(dicts, columns=names, index=names))
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    # 将人脸特征向量和标签保存下来
    # 人脸特征向量和标签的index保持一致
    # names中的顺序应该和数据集类别的顺序相同
    torch.save(mean_embedings, feature_path + 'features.pt')
    print(names)
    torch.save(names, feature_path + 'names.pt')


# add a new face to existing database
def AddNewFace2DB():
    a = 1


# 将一个list的sample组成一个mini-batch的函数
def collate_fn(x):
    return x[0]

def args(known=False):
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=r'centerface.onnx', type=str)
    parser.add_argument('--source',  default='./data_DB/',  type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--feature_path', default=f"./data_DB/emd/", type=str, help='img path for output')
    parser.add_argument('--cropped_save_pth', default='./data_DB/', help='save result to path_corrped')
    #parser.add_argument('--names', default=f"./data/DB{DB}_features/names.pt", type=str, help='img path for output')
   # parser.add_argument('--names', default=f"../data/DB{DB}_features/names.pt", type=str, help='save_embdding_name of path')
    
    #parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
   # parser.add_argument('--save_txt', action='store_true', help='path for save results to *.txt')
    #parser.add_argument('--show', action='store_true', help='show result')
    #parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf_thres', default=0.5,type=float, help='the thresh of post_processing')
   # parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
   # parser.add_argument('--onnxruntime', action='store_true', help='load modle for onnxruntime')
    #parser.add_argument('--yolov5_face',action='store_true', help='load yolov5  for pytorch model')
    #parser.add_argument('--h264_flag',action='store_true', help='h264 support for python ')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
if __name__ == "__main__":
    # os.name返回当前操作系统名字,nt为Windows,posix为linux
    # thread numbers
    opt=args(True)
    workers = 0 if os.name == 'nt' else 4
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Running on {device}")

    # DB = 25
    # DB_path = f"/home/project/球赛人员头像/"
    # cropped_path = f"./data/DB{DB}_images_cropped/"
    # feature_path = f"./data/DB{DB}_features/"

    #print(DB_path)
    CreatNewDB(**vars(opt))
