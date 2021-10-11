import numpy as np
import cv2
import datetime
import torch
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate
from PIL import Image
import io
import onnx
import onnxruntime
import time

#
class CenterFace(object):
    def __init__(self, weights,onnxruntime_flag=False,landmarks=True):
        self.landmarks = landmarks
        self.onnxruntime=onnxruntime_flag
        # # load model 
        # self.session=[] 
        # self.inputs = []
        # self.outputs=[]
        # self.net=[]
        if self.onnxruntime:
            # onnx rumtime read modify onxx file 
            self.session = onnxruntime.InferenceSession(weights)
            self.inputs = self.session.get_inputs()[0].name
            self.outputs = ["537", "538", "539", "540"]
            #cv dnn read onnx
        else : 
            
            self.net = cv2.dnn.readNetFromONNX(weights)
            
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 1, 1

    def __call__(self, img, height, width, threshold=0.5):
    
        if onnxruntime:
            self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 480, 640 , 480/height, 640/width  
        else:
            self.img_h_new, self.img_w_new, self.scale_h, self.scale_w=self.transform(height, width)

        # conver type:Image->>ndarray
        if not isinstance(img, np.ndarray):
             img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        #img =cv2.resize(img,(self.img_w_new, self.img_h_new))
        return self.inference_opencv(img, threshold)

    # 3
    def inference_opencv(self, img, threshold):
        # 对图像进行预处理：1 减均值；2 缩放；3 通道交换
        
        #BGR->RGB
      
        #img =cv2.resize(img,(self.img_w_new, self.img_h_new))
        if self.onnxruntime:
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img =cv2.resize(img,(self.img_w_new, self.img_h_new))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
            input_image = np.expand_dims(np.swapaxes(np.swapaxes(img,0,2),1,2),0).astype(np.float32)

            heatmap, scale , offset ,lms = self.session.run(None, {self.inputs: input_image})
           # return self.postprocess(heatmap, lms, offset, scale, threshold)
        #begin = datetime.datetime.now()
        else:
        
            blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)

            if self.landmarks:
                # 都是ndarray类型
                heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
                #return self.postprocess(heatmap, lms, offset, scale, threshold)
            else:
                heatmap, scale, offset = self.net.forward(["535", "536", "537"])
                #return self.postprocess(heatmap, lms, offset, scale, threshold)
       # end = datetime.datetime.now()
        # print("cpu times = ", end - begin)
     
        return self.postprocess(heatmap, lms, offset, scale, threshold)

   
    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    # 4
    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets


    # 5
    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    # 6
    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep


# 提取人脸/FaceExtract
class FaceExtract(object):
    def __init__(self):
    	# 无意义，只是为了让构造函数非空
        self.empty = 1

    def __call__(self, img, box, target_size=160, margin=0, save_path=None, post_process=True):
        return self.extract_face(img, box, target_size=target_size, margin=margin, save_path=save_path,
                                 post_process=post_process)

    def extract_face(self, img, box, target_size, margin, save_path, post_process):
        """
        Extract face + margin from PIL Image given bounding box.
            Arguments:
                img {PIL.Image} -- A PIL Image.
                box {numpy.ndarray} -- Four-element bounding box.
                target_size {int} -- Output image size in pixels. The image will be square.
                margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
                    Note that the application of the margin differs slightly from the davidsandberg/facenet
                    repo, which applies the margin to the original image before resizing, making the margin
                    dependent on the original image size.
                save_path {str} -- Save path for extracted face image. (default: {None})

            Returns:
                torch.tensor -- tensor representing the extracted face.
            """

        # face = extract_face(img, box, target_size, margin, save_path)
        margin = [
            margin * (box[2] - box[0]) / (target_size - margin),
            margin * (box[3] - box[1]) / (target_size - margin),
        ]

        raw_img_size = self.get_image_size(img)

        # ???
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_img_size[0])),
            int(min(box[3] + margin[1] / 2, raw_img_size[1])),
        ]

        # cv2 input :ndarray type ,(160,160,3)
        # PIL input:Image tyoe 
        face = self.crop_resize(img, box, target_size)

        if save_path is not None:
            self.save_img(face, save_path)

        # tensor
        face = F.to_tensor(np.float32(face))

        # 标准化/norm
        if post_process:
            face = (face - 127.5) / 128.0

        return face

    def get_image_size(self, img):
        if isinstance(img, (np.ndarray, torch.Tensor)):
            return img.shape[1::-1]
        else:
            return img.size

    def crop_resize(self, img, box, target_size):
        """
        resize the shape of a face
        """
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = cv2.resize(
                img,
                (target_size, target_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        elif isinstance(img, torch.Tensor):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = self.imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(),
                (target_size, target_size)
            ).byte().squeeze(0).permute(1, 2, 0)
        else:
            out = img.crop(box).copy().resize((target_size, target_size), Image.BILINEAR)
        return out

    def save_img(self, img, path):
        
        
        if isinstance(img, np.ndarray):
            cv2.imwrite(path, img)
            # cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            #cv2.imwrite(path, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
           # path=path+str(time.time())+'.jpg'
            img.save((path + f"{str(time.time())}.jpg"))

    def imresample(self, img, sz):
        im_data = interpolate(img, size=sz, mode="area")
        return im_data
        