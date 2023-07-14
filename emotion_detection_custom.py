# python3 emotion_detection_custom.py -i ./customDataset/sohrabi/train/neutral -o ./customDataset/train/ --model output/model.pth   --prototxt model/deploy.prototxt.txt   --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel


# import the necessary libraries
from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from spike import config as cfg
from torchvision import transforms
from spike import EmotionNet
import torch.nn.functional as nnf
from spike import utils
import numpy as np
import argparse
import torch
import cv2
import os
import pickle


# initialize the argument parser and establish the arguments required
# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image_folder", type=str, required=True,
                    # help="path to the image folder")
# parser.add_argument("-o", "--output_folder", type=str, required=True,
                    # help="path to the output folder")
# parser.add_argument("-m", "--model", type=str, required=True,
#                     help="path to the trained model")
# parser.add_argument('-p', '--prototxt', type=str, required=True,
#                     help='Path to deployed prototxt.txt model architecture file')
# parser.add_argument('-c', '--caffemodel', type=str, required=True,
#                     help='Path to Caffe model containing the weights')
# parser.add_argument("-conf", "--confidence", type=int, default=0.9,
#                     help="the minimum probability to filter out weak detection")
# args = vars(parser.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['caffemodel'])
print(cfg.prototxt, cfg.caffe_model)
net = cv2.dnn.readNetFromCaffe(cfg.prototxt, cfg.caffe_model)

 
# check if gpu is available or not
device = "cuda" if torch.cuda.is_available() else "cpu"
 
# dictionary mapping for different outputs
emotion_dict = {0: "angry", 1: "fear", 2: "happy", 3: "neutral",
                4: "sad", 5: "surprise"}
 
# load the emotionNet weights
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))
# model_weights = torch.load(args["model"])
model_weights = torch.load(cfg.model)
model.load_state_dict(model_weights)
model.to(device)
model.eval()
 
# initialize a list of preprocessing steps to apply on each image during runtime
data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    Resize((48, 48)),
    ToTensor()
])

# initialize the video stream

# os.makedirs(args['output_folder'],exist_ok=True)
os.makedirs(cfg.output_folder,exist_ok=True)

# image_list = os.listdir(args['image_folder']) 
image_list = os.listdir(cfg.image_folder) 

print(len(image_list))
# iterate over frames from the video file stream
count = 0
emo_dict = {}
for i in  image_list:
    # image_path = os.path.join(args['image_folder'],i)
    image_path = os.path.join(cfg.image_folder,i)
    # print("image_path",image_path)
    # print("count",count)
    # count += 1
    # print(image_path)
    frame = cv2.imread(image_path)
    # read the next frame from the input stream
    # (grabbed, frame) = vs.read()
 
    # check there's any frame to be grabbed from the steam
    # if not grabbed:
    #     break
 
    # clone the current frame, convert it from BGR into RGB
    frame = utils.resize_image(frame, width=720, height=720)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # initialize an empty canvas to output the probability distributions
    canvas = np.zeros((300, 300, 3), dtype="uint8")
 
    # get the frame dimension, resize it and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
 
    # infer the blog through the network to get the detections and predictions
    net.setInput(blob)
    detections = net.forward()
      
    # iterate over the detections
    for i in range(0, detections.shape[2]):
 
        # grab the confidence associated with the model's prediction
        confidence = detections[0, 0, i, 2]
 
        # eliminate weak detections, ensuring the confidence is greater
        # than the minimum confidence pre-defined
        # if confidence > args['confidence']:
        if confidence > cfg.confidence:
 
            # compute the (x,y) coordinates (int) of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # grab the region of interest within the image (the face),
            # apply a data transform to fit the exact method our network was trained,
            # add a new dimension (C, H, W) => (N, C, H, W) and send it to the device
            # print(start_y,end_y, start_x,end_x)
            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)
 
            # infer the face (roi) into our pretrained model and compute the
            # probability score and class for each face and grab the readable
            # emotion detection
            predictions = model(face)
            prob = nnf.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            top_p, top_class = top_p.item(), top_class.item()
 
            # grab the list of predictions along with their associated labels
            emotion_prob = [p.item() for p in prob[0]]
            emotion_value = emotion_dict.values()
            # draw the bounding box of the face along with the associated emotion
            # and probability
            face_emotion = emotion_dict[top_class]


            # crop_path = os.path.join(args['output_folder'],f"{face_emotion}_face_{count}.jpg")
            crop_path = os.path.join(cfg.output_folder,f"face_{count}_{face_emotion}.jpg")
            
            emo2 = {}
            emo2["path"] = crop_path
            emo2["angry"] = emotion_prob[0]
            emo2["fear"] = emotion_prob[1]
            emo2["happy"] = emotion_prob[2]
            emo2["neutral"] = emotion_prob[3]
            emo2["sad"] = emotion_prob[4]
            emo2["surprise"] = emotion_prob[5]
            emo_dict[count] = emo2
            count += 1
            # print(crop_path)
            # break
            if cfg.save_crop:
                cv2.imwrite(crop_path,output[start_y:end_y , start_x:end_x ])
            
            # draw the probability distribution on an empty canvas initialized
            for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                prob_text = f"{emotion}: {prob * 100:.2f}%"
                width = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
                              (0, 0, 255), -1)
                cv2.putText(canvas, prob_text, (5, (i * 50) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
            
            # print(os.path.basename(args['image_folder']))
            # print(args['output_folder'])


            face_text = f"{face_emotion}: {top_p * 100:.2f}%"
            cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(output, face_text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.05, (0, 255, 0), 2)
 
    # display the output to our screen
    if cfg.show_detection:
        cv2.imshow("Face", output)
    if cfg.show_emotion:
        cv2.imshow("Emotion probability distribution", canvas)
 
    # break the loop if the `q` key is pressed
    key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(500)#pauses for 3 seconds before fetching next image
    
    if key == ord("q"):
        break
 

with open(cfg.dict_path, 'wb') as fp:
    pickle.dump(emo_dict, fp)
    print('dictionary saved successfully to file')