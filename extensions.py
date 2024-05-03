# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To test a pre-trained ResNet50 network on images capured from a webcam
# Tasks: 
# 1. Extensions: The script performs image classification using a pre-trained ResNet50 network
# Dataset: ImageNet (https://image-net.org/)

import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests
import sys
import cv2

# Main function to capture frames from the webcam and predict the class results
def main(argv):
    # Load the pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download the ImageNet class index
    class_index = json.loads(requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').text)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Show the frame in a window
        cv2.imshow('Press "s" to save a frame or "q" to quit', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # If 's' is pressed, save the frame and predict the class results
        if key == ord('s'):
            # Save the frame
            cv2.imwrite('saved_frame.jpg', frame)

            # Convert the frame to PIL Image and apply the transformations
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            # Pass the image through the model
            with torch.no_grad():
                out = model(batch_t)

            # Print the top 5 predictions
            _, indices = torch.topk(out, 5)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            for idx in indices[0]:
                print('Label:', class_index[idx.item()], ', Confidence:', percentage[idx].item())

        # If 'q' is pressed, break the loop
        elif key == ord('q'):
            break

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)