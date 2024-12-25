import streamlit as st
import random
import argparse
import logging
import webbrowser
from clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
from get_faces_from_camera import TrainingDataCollector
from face_embedding import GenerateFaceEmbedding
from facePredictor import FacePredictor
from ai_training.train_softmax import TrainFaceRecogModel

# Setup logging
logFileName = "ProceduralLog.txt"
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=logFileName,
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Streamlit App
def main():
    st.title("Student Monitoring Registration")
    
    # Student Details Input
    studentID = st.text_input("Student ID")
    rollNo = st.text_input("Roll No")
    studentName = st.text_input("Student Name")
    emailID = st.text_input("Email ID")
    mobileNo = st.text_input("Mobile No")
    
    # Notification message placeholder
    message = st.empty()
    
    # Collect Images Button
    if st.button("Take Images"):
        collectUserImageForRegistration(studentID, rollNo, studentName, message)
    
    # Train Model Button
    if st.button("Train Images"):
        trainModel(message)
    
    # Prediction Button
    if st.button("Predict"):
        makePrediction()

    # Quit Button (doesn't really apply in Streamlit as it's not a traditional GUI)
    if st.button("Quit"):
        st.stop()

# Collect User Image Function
def collectUserImageForRegistration(studentID, rollNo, name, message):
    if not studentID or not rollNo or not name:
        message.warning("Please fill all the fields!")
        return
    
    # Arguments and logic
    args = {"faces": 50, "output": f"datasets/train/{name}"}
    trnngDataCollctrObj = TrainingDataCollector(args)
    trnngDataCollctrObj.collectImagesFromCamera()
    
    message.success(f"Collected {args['faces']} images for training.")

# Train Model Function
def trainModel(message):
    # Train model logic here
    args = {
        "embeddings": "faceEmbeddingModels/embeddings.pickle",
        "model": "faceEmbeddingModels/my_model.h5",
        "le": "faceEmbeddingModels/le.pickle"
    }
    
    # Generate embeddings and train model
    getFaceEmbedding()
    faceRecogModel = TrainFaceRecogModel(args)
    faceRecogModel.trainKerasModelForFaceRecognition()
    
    message.success("Model training is successful. Now you can go for prediction.")

# Prediction Function
def makePrediction():
    faceDetector = FacePredictor()
    faceDetector.detectFace()

# Main execution
if __name__ == "__main__":
    main()
