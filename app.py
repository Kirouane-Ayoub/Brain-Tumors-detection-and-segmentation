import streamlit as st
import cv2
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode , Visualizer
from detectron2 import model_zoo
import cvzone
import math
import uuid
uuidOne = uuid.uuid1()

st.sidebar.image('logo.png' , width=250)

tab1 , tab2, tab3 = st.tabs(["Home", 
                            "Detection" , 
                            "Segmentation"])
with st.sidebar:    
    device = st.selectbox("Select Device name " , 
                            ("cpu" , "GPU") , key = "<uniquevalueofsomesort1>")
    acc_thr = st.slider( "select threshold accuracy : " ,
                        min_value=0.0 ,
                        max_value=1.0 ,
                        step=0.01 , 
                        value=0.5)
    iou = st.slider("Select Intersection over union (iou) value : " ,
                     min_value=0.1 , 
                     max_value=1.0 , 
                     value=0.5)
    save = st.selectbox("Do you want to save the Result ? " , 
                        ["YES" , "NO"]) 
with tab1:
    st.header("About Project : ")
    st.image("img_home/detect.png")
    st.write("""Brain Tumors are complex. There are a lot of abnormalities in the sizes and location of the brain tumor(s). 
    This makes it really difficult for complete understanding of the nature of the tumor. 
    Also, a professional Neurosurgeon is required for MRI analysis. 
    Often times in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI.
    So an automated system on Cloud can solve this problem.
    """)
    st.text("""
    This model can detect and segment brain tumors
    """)
    st.image("img_home/segmentation.jpg")
    st.header("About The Dataset : ")
    st.write("This dataset is part of RF100, an Intel-sponsored initiative to create a new object detection benchmark for model generalizability.")

def segment(zoo_conf , impath , weight_path , device , conf_thres , save) : 
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_conf))
    cfg.MODEL.WEIGHTS = weight_path
    im = cv2.imread(impath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    predictor = DefaultPredictor(cfg)
    predictions =predictor(im)
    viz = Visualizer(im[:,:,::-1] , 
    instance_mode=ColorMode.IMAGE)
    output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
    st.image(output.get_image()[:,:,::-1])
    if save == "YES" :
        cv2.imwrite(f"results/{uuidOne}.jpg", output.get_image()[:,:,::-1])
    else : 
        pass

def detection(acc_thr , iou , save , name) : 
    class_names = ["tumor"]
    img = cv2.imread(name)
    model = YOLO("tumor-detection-j9mqsv2.pt")
    results = model.predict(source=name, conf=acc_thr , iou=iou) # , save=True
    for result in  results: 
        bboxs = result.boxes
        for box in bboxs : 
                # bboxes
            x1  , y1 , x2 , y2 = box.xyxy[0]
            x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
            clsi = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100 ))/100
            w,h = int(x2 - x1) , int(y2 - y1)
            cvzone.cornerRect(img , (x1 , y1 , w , h) , l=7 , rt=1)
            #(wt, ht), _ = cv2.getTextSize(f"class:{clsi} {conf}", cv2.FONT_HERSHEY_PLAIN,1, 1)
            #cv2.rectangle(img, (x1, y1 - 18), (x1 + wt, y1), (0,0,255), -1)
            #cv2.putText(img, f"class:{clsi} {conf}", (max(0,x1),max(20 , y1)),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255), 1)
            cvzone.putTextRect(img , f"{class_names[clsi]} {conf}" , (max(0,x1) , max(20 , y1)),thickness=1 , colorR=(0,0,255) , scale=1 , offset=3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        st.image(img)
        if save == "YES" :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(f"results/{uuidOne}.jpg", img)
        else : 
            pass

with tab2:     
    file_uploader = st.file_uploader("Select your file : " ,
                                      type=["jpg" , "png" , "jpeg"])
    if file_uploader : 
        name = file_uploader.name
        if st.button("start Detection") :
            detection(acc_thr=acc_thr , iou=iou , save=save , name=name)

with tab3 :
    image_upload = st.file_uploader("Upload your image" ,
                                     type=["png" , "jpg" , "jpeg"] , 
                                     key = "<uniquevalueofsomesort13>")
    if image_upload:
        image_name = image_upload.name 
    if st.button("start segmentation") :
        segment(impath=image_name , 
        zoo_conf="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" , 
        weight_path="tumor_segment.pth" , 
        conf_thres=acc_thr , device=device)




