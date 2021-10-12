## Real-time Face Recognition Using Pytorch 



### Provides a complete face recognition project, reasoning part of the script provided/提供一个完整的人脸项目，目前上传一部分可直接使用，后续待补充模型训练代码以及优化,链接Deepstream的C++ sdk参考使用

deepstream face:[https://github.com/positive666/Deepstream_Project/tree/main/Deepstream_Face]


## News


  1.add Face Detection (cebterface/yolov5_face) 
  
  2.add facenet
  
  
  3.add av to fix video format questions
  
  4.Feature Library scripting and options for storing unrecognized images
  




## Quick start

1. Install:
    
    ``` need python >=3.5 with general python libary...
    # With pip:
    pip install requirements.txt
    
    ``` 
    
2. prepare models and  embeddings_face_data

     centerface models alreay exist this repo 
	 or 
     you can go repo :https://github.com/Star-Clouds/CenterFace 
	 
	 prepare your data -----create embeddings_face 
	 
	 note!!!  Keep the same number of images for each face category in your feature library!!!!!!!!
	 
	 ``` 
	 python face_embedding.py   
	 ``` 
	 
     Generating a feature library in your  Program Root
	 
3. How to run
    ``` 
     python test_facenet.py  --conf_thres 0.45 --weights ./new.onnx    --source (your test data path)  --embeddings_face  (your face_embedding path)  
```



## References

1. references facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

2. yolov5_face repo: [https://github.com/deepcam-cn/yolov5-face.git]

3. centerface repo:  [https://github.com/Star-Clouds/CenterFace]