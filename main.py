import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import os
import json
import yaml
from datetime import datetime


def objectDetection(img_path:str, model) -> list:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(img_path)
    image_name = image_name.split('.')[0]

    result = model(image)
    result.crop(save_dir=os.path.join("YOLO", image_name))
    detectedObjects = result.render()[0]
    path = os.path.join("YOLO", image_name, 'crops', '**', '*.jpg')

    listOfObjects = []
    for filename in glob(path):
        obj = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        listOfObjects.append(obj)     

    return listOfObjects, detectedObjects


def similarity_top(similarity_list:list, listOfObjects:list, N) -> tuple():
    results = zip(range(len(similarity_list)), similarity_list)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    images = []
    scores=[]
    scores2=[]
    for index, score in results[:N]:
        scores.append(score)
        images.append(listOfObjects[index])

    return scores, images


def findObjects(listOfObjects:list, query:list, model, preprocess, device:str, N) -> tuple():
    similarity=[]

    for i in listOfObjects:
        objects = preprocess(text=query, images=Image.fromarray(i), return_tensors="pt", padding=True)

        outputs = model(**objects)
        logits_per_image = outputs.logits_per_image
        similarity.append(logits_per_image[0][0].item()+logits_per_image[0][1].item())

    scores, images = similarity_top(similarity, listOfObjects, N=N)
        
    return scores, images


def plotResults(scores: list, images: list, n):
    plt.figure(figsize=(20,5))
    for index, img in enumerate(images):
        if index < n:
            plt.subplot(1, n, index+1)
            plt.imshow(img)
            plt.title(scores[index])
            plt.axis('off')
    if not os.path.exists("resultFolder"):
        os.makedirs("resultFolder")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"image_{timestamp}.png"
    plt.savefig(os.path.join("resultFolder", filename))


with open('ObjectDetection.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

OBJDETECTIONMODEL = config['object_detection_config']['model']
OBJDETECTIONREPO = config['object_detection_config']['repository']

objectDetectorModel = torch.hub.load(OBJDETECTIONREPO, OBJDETECTIONMODEL)

DEVICE = 'cpu'
N = 5

objectFinderModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
preProcess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def pipeline(video_path, prompt1, prompt2):
    data={}
    cap = cv2.VideoCapture(video_path)

    output_dir = 'FootageFrames'
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = 'result_data.json'
        
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        timestamp_seconds = frame_count / fps
        
        if frame_count % int(fps) == 0:
            frame_file = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_file, frame)

            listOfObjects, detectedObjects = objectDetection(frame_file, objectDetectorModel)
            scores, images = findObjects(listOfObjects, [prompt1,prompt2], objectFinderModel, preProcess, DEVICE, N)

            correctImg=0

            for i in scores:
                if i>46:
                    correctImg+=1

            if correctImg > 0:
                data['timestamp'] = timestamp_seconds
                data['Number of detected objects'] = correctImg

                with open(output_file, 'a') as f:  # Open JSON file in append mode here
                    json.dump(data, f, indent=4)

                plt.figure(figsize=(20,5))
                plt.axis('off')
                plotResults(scores, images, correctImg)

        frame_count += 1

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = input("Enter the video path: ")
    prompt_1 = input("Enter prompt 1: ")
    prompt_2 = input("Enter prompt 2: ")

    pipeline(video_path, prompt_1, prompt_2)

if __name__ == "__main__":
    main()