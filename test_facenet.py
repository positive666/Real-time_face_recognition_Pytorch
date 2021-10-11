 # coding: utf-8
from models.inception_resnet_v1 import  InceptionResnetV1
import torch
from PIL import ImageDraw, Image, ImageFont
from torchvision import datasets
from torch.utils.data import DataLoader
import imageio

import sys, os
import threading
import skimage.io
import time
import cv2
from torch.nn import functional as F
import av
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from models.yolov5_face import *
from models.experimental import attempt_load
from models.centerface import CenterFace, FaceExtract
from utils.torch_utils import select_device, time_sync,colorstr
from utils.google_utils import attempt_download
import argparse
# mtcnn = MTCNN(keep_all=True)
from pathlib import Path


"""
    author github:https://github.com/positive666?tab=repositories
    
    example:
       
    cv_dnn read onnx---    run: 
       python test_facenet.py  --conf_thres 0.45 --weights ./centerface.onnx    --source (your test data path)  --embeddings_face  (your face_embedding path)  
    onnx read centerface---- run:
       python test_facenet.py  --conf_thres 0.45 --weights ./new.onnx    --source (your test data path)  --embeddings_face  (your face_embedding path)  --onnxruntime 
    yolov5  :
       python test_facenet.py  --conf_thres 0.45 --weights ./yolov5.onnx    --source (your test data path)  --embeddings_face  (your face_embedding path)  --yolov5
""" 

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = 'mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
# with torch.no_grad():
    # def get_score(x, T : int = 100,
                            # alpha : float = 130.0,
                            # r : float = 0.88):
       
            # np.repeat(d)
            # face_embedding = models['resnet'](face.unsqueeze(0).to(device))     
            # norm = normalize(x.detach().cpu(), axis=1)
            # print(norm.shape)
            # # Only get the upper triangle of the distance matrix
            # eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(T, k=1)]
           
            # # Calculate score as given in the paper
            # score = 2*(1/(1+np.exp(np.mean(eucl_dist))))
            # # Normalize value based on alpha and r
            # return 1 / (1+np.exp(-(alpha * (score - r))))

def cosin_metric(x1, x2):
    x1=x1.detach().cpu()
    x2=x2.detach().cpu()
    return np.dot(x1, x2) / ((x1.norm() * x2.norm()))

def compare(face1, face2):

    face1_norm = F.normalize(face1)            #normalize
    face2_norm = F.normalize(face2)
    cos_sim = torch.matmul(face1_norm, face2_norm.T)                                       
    # cos_Sim = torch.dot(face1_norm.reshape(-1), face2_norm.reshape(-1))
    return cos_sim
            
def show_results(img, xywh, landmarks):
    img=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    h,w,c = img.shape
    #tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
   # cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
    box=x1,y1,x2,y2
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        #cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)
        lms=point_x,point_y
    
    #tf = max(tl - 1, 1)  # font thickness
    #label = str(conf)[:5]
    #cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return box,lms
    
def detect_img(img, names, models,embeddings,device=None,threshold=0.55,save_json=False,onnxruntime=False,yolov5=False,save_unreg_flag=False,FIQ=True):
    #init model
    
    #print('check img type:',type(img))
    iou_thres=0.3
    
    #mtcnn = MTCNN(keep_all=True, device=device)
    extractFace = FaceExtract()
  
    # /init font style 
    fontStyle = ImageFont.truetype("./SIMYOU.TTF", 26)
    
    w, h = img.size
    
    #print(w, h)
    boxes, lms = detect_yolov5(models['detect_face'],img, iou_thres,threshold,device)if yolov5 else models['detect_face'](img, h ,w , threshold)  #input img type is BGR(array)
   # boxes, lms = mtcnn.detect(img)  # 
    # face = mtcnn(img)
    # boxes, prob = mtcnn.detect(img)  # MTCNN

    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    
    print("detect face mumbers：", len(boxes))
    for i, box in enumerate(boxes):
        if yolov5:
            box,lms[i]=show_results(img[i],box,lms[i])
        # get box,score
        #print("box：", len(boxes))
        else:
            box, score = box[:4], box[4]
        x1, y1, x2, y2 = box
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        
        # get norm crop face  tensor
        face = extractFace(frame_draw, box)
        
        
        # resnet输入参数为4维
        face_embedding = models['resnet'](face.unsqueeze(0).to(device))
        # if FIQ:
            # score=get_score(face)
            # print("人脸质量分数：",score)
        #print(embeddings)
        #print((face_embedding))
        #face_embedding=face_embedding.cuda()
        # l2 sim compute 
        #probs = [(compare(face_embedding,embeddings[i])).item() for i in range(len(embeddings))]
        probs = [(face_embedding - embeddings[i]).norm().item() for i in range(len(embeddings))]
       # print('pronbs:',probs)
        #  后面换成faiss库
        dis = min(probs)

       # dis = max(probs)
        
        if dis >0.78:
            name = 'unknow'
            if save_unreg_flag:
                unkown_path = "./Unknow/"
                if not os.path.exists(unkown_path):
                    os.makedirs(unkown_path)
                frame_draw.save(unkown_path+ name+f'---{dis}.jpg')
        else:
            index = probs.index(dis)
            
            name = names[index]
            
            print("name:",name,index)
        # 
            if save_json:
          
                js={'image_id': index,
                      'face_id': name,
                      'bbox': [x1,y1,x2,y2],
                            }
                jdict.append(js)          
                print('json_result:',js)          
           # show result 
        draw.text((x1 + 10, y1 + 5), str(name) + str(i + 1), fill='red', font=fontStyle, align='center')
            # show distance
        draw.text((0, i * 24), f"({i + 1}: {dis:.3f})", fill='red', font=fontStyle)
        
    return frame_draw,lms


def collate_fn(x):
    return x[0]


# detect a face dataset and save its result
def detect_multi(dataset_path, names,model, embeddings, save_path=None,device=None,threshold=0.5,save_json=False,onnxruntime=False,yolov5_face=False):
    dataset = datasets.ImageFolder(dataset_path)
    
    total_nums = len(dataset)
    # class nums
    class_nums = len(dataset.class_to_idx.items())
   
    num_one_class = total_nums / class_nums

    # dic类型
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)

    i = 0
    # iamge类型->>>'PIL.Image.Image'>
    for image, index in loader:
        name = dataset.idx_to_class[index]
        path = save_path + name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

     
        img = detect_img(image, names,model,embeddings=embeddings,device=device)
        img.save(path + f"{name}_{i}.jpg")
        i += 1
        if i % num_one_class == 0:
            i = 0
        # img.show()
        # break


def detect_videos(video_path, names, model,embeddings, video_save_path=None,device=None,threshold=0.5,save_json=False,onnxruntime=False,yolov5_face=False,h264_flag=False,per_frame=8,Writer_Video=True,save_unknown_flag=False):
    cap = cv2.VideoCapture(video_path)
    # frame是每一帧的图像,ret的类型为bool,表示是否读到图片
    ret, frame = cap.read()
    if(ret==False or frame is None ) or (h264_flag):
        h264_flag=True
        # a = mp4_to_H264()
        # from_path = from_path
        print('convert h264')
        # to_path = from_path[:-3]+"h264"+".mp4"
        # print('to path',to_path)
        # a.convert_byfile(from_path, to_path)
        # cap = cv2.VideoCapture(to_path)
        # ret, frame = cap.read()
        
        reader = imageio.get_reader(video_path,'ffmpeg')
        print(len(reader))
        for i,im in enumerate(reader): 
            image = reader.get_data(i) #读取图片
            print(image)
            #print(im)
            frame=skimage.img_as_float(image).astype(np.float64)
            #print(frame)
            break
            #frame = skimage.img_as_float(im).astype(np.float64)
        ret=True    
    # if(frame is None):
        # print("data decode fail")
        # return  -1
    # 获取原视频帧数
    
    fps = cap.get(propId=cv2.CAP_PROP_FPS)
    print(fps)
    indx=(0,0)
    h, w = frame.shape[:2] 
    count=0
    if(Writer_Video):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(video_save_path, fourcc, fps, (w, h), True)
    if(h264_flag):
  
           
            # a = mp4_to_H264()
            # from_path = from_path
            print('convert h264 decode method......')
            # to_path = from_path[:-3]+"h264"+".mp4"
            # print('to path',to_path)
            # a.convert_byfile(from_path, to_path)
            # cap = cv2.VideoCapture(to_path)
            # ret, frame = cap.read()
            #reader = imageio.get_reader(video_path)
            container = av.open(video_path)
            print("container:", container)
            print("container.streams:", container.streams)
            print("container.format:", container.format)

            for img in container.decode(video = 0):
                #print("process frame: %04d (width: %d, height: %d)" % (img.index, img.width, img.height))
               # print('check im type:',type(img))
                img = img.to_ndarray(format='bgr24')

                #frame.to_image().save("output/frame-%04d.jpg" % frame.index)
            # for i,im in enumerate(reader): 
                # #im = reader.get_data(i)
                # #print(i))
                # #print('1',type(im))
                #img = skimage.img_as_float(im).astype(np.float64)  # convert ->array
                #print('2',type(img))
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #print()
                img,lms = detect_img(img, names,model,embeddings=embeddings,device=device,threshold=threshold,save_json=save_json,onnxruntime=onnxruntime,yolov5=yolov5_face,save_unknown_flag=save_unknown_flag)
            # Image转Opencv
                frame= cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                for lm in lms:
                    for i in range(0, 5):
                        cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 6, (0, 255, 0), -1)
                out.write(frame)if Writer_Video else print('>> <<')
                print('h264 detect one' )
    else:
        while (ret) :
            # opencv转Image
           
            count+=1
            
            if(count ==per_frame):
                
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img,lms = detect_img(img, names,model,embeddings=embeddings,device=device,threshold=threshold,save_json=save_json,onnxruntime=onnxruntime,yolov5=yolov5_face)
                # Image转Opencv
                frame= cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
               
                #for lm in lms:
                    #for i in range(0, 5):
                        #cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 6, (0, 255, 0), -1)
                if Writer_Video:   
                    out.write(frame) 
                ret, frame = cap.read() 
                
                #print('detect once per {per_frame} image'.format(per_frame=per_frame))
                count=0   
            
                
            # if(frame is None):
               # print("decode failed.....")
               # break 
    
    cap.release()
    if Writer_Video:   
                    out.release()
    #out.release()





# another way to read images dataset


    
def read_files(dataset_path, save_path):
    set_dirs = os.listdir(dataset_path)
    print(set_dirs)
    for class_name in set_dirs:
        dir_path = data_path + class_name + '/'
        imgs_dir = os.listdir(dir_path)
        for img_name in imgs_dir:
            img_path = dir_path + img_name
            img = Image.open(img_path)
            img = detect_img(img)
            path = save_path + class_name + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            # img.save(path + f"{class_name}_{img_name}")
            print(path + f"{class_name}_{img_name}")
        break

def args(known=False):
    #import argparse
    DB=25
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=r'centerface.onnx', type=str)
   # parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--embeddings_face', default=f"./data_DB/emd/features.pt", type=str, help='img path for output')
    parser.add_argument('--source',  default='../../img_face.mp4',  type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--names', default=f"./data_DB/emd/names.pt", type=str, help='img path for output')
   # parser.add_argument('--names', default=f"../data/DB{DB}_features/names.pt", type=str, help='save_embdding_name of path')
    parser.add_argument('--conf_thres', default=0.5,type=float, help='the thresh of post_processing')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save_txt', action='store_true', help='path for save results to *.txt')
    #parser.add_argument('--show', action='store_true', help='show result')
    #parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='./runs/result', help='save result to project/names')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--onnxruntime', action='store_true', help='load modle for onnxruntime')
    parser.add_argument('--yolov5_face',action='store_true', help='load yolov5  for pytorch model')
    parser.add_argument('--h264_flag',action='store_true', help='h264 support for python ')
    parser.add_argument('--per_frame',default=8,type=int, help='the thresh of post_processing')
    parser.add_argument('--save_unknown_flag',action='store_true', help='save unkonwn image ')
    parser.add_argument('--Writer_Video',action='store_true', help='Writer_Video ')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
    
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model
  
#@torch.no_grad()
def run(weights=r'centerface.onnx',  # model.pt path(s)
        embeddings_face="./data/DB25_features/features.pt",
        source= '../../img_face.mp4',  # file/dir/URL/glob, 0 for webcam
        names="./data/DB{DB}_features/names.pt",
        conf_thres=0.5,  # confidence threshold
        nosave=False,
        save_txt=False,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project='runs/result',  # save results to project/name
        save_json=False,
        onnxruntime=False,
        yolov5_face=False,
        h264_flag=False,
        per_frame=8,
        Writer_Video=True,
        save_unknown_flag=False
        ):
    print("Running on device:{}".format(device))
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')) or  source.endswith(VID_FORMATS)
     # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    device = select_device(device)    
    models={'detect_face':None,'resnet':None}
    models['detect_face'] = load_model(weights, device) if yolov5_face else CenterFace(weights,onnxruntime,landmarks=True)
    models['resnet']= InceptionResnetV1(pretrained='vggface2').eval().to(device)
   # models.append(model)
   # models.append(resnet)
    # if not models:
        # print("model dict is None!")
        
    embeddings_face = torch.load(embeddings_face)
    embeddings_face=torch.stack(embeddings_face,0).to(device)
    names = torch.load(names)
    
    
    # Dataloader
    if webcam:
        
        # #ssert ret, f'Camera Error'
        project=project+"_face.mp4"
        print("statrt docoding face reg..")
        start_time = time.time()
        detect_videos(source, names, models,embeddings_face, project,device,conf_thres,save_json,onnxruntime,yolov5_face,h264_flag,per_frame,Writer_Video,save_unknown_flag)
        end_time = time.time()
        print(f"time(s): {end_time - start_time:.2f}")
    else:
       
        detect_multi(source, names, models,embeddings_face, project,device,conf_thres,save_json,onnxruntime,yolov5_face)
   
 
       
    
    
    # if save_json:
                # save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # callbacks.on_val_image_end(pred, predn, path, names, img[si])
    
    
    
if __name__ == '__main__':

   
    # list
    opt=args(True)
    print(colorstr('face_reg: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    
    run(**vars(opt))
    
   


