# @title Predicting from new image
from keras_facenet import FaceNet
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
# from google.colab.patches import cv2_imshow
# Load the embedding and filenames
feature_list = np.array(pickle.load(open('C:/Alpesh/1stopai/Celebrity Image Classifier Project/extracted_features.pkl','rb')))
filenames = pickle.load(open('C:/Alpesh/1stopai/Celebrity Image Classifier Project/filenames.pkl','rb'))

# Initialize the FaceNet model
facenet_model = FaceNet()

# Initialize the MTCNN face detector
detector = MTCNN()

# Load the sample image and detect the face
sample_img = cv2.imread('C:/Alpesh/1stopai/Celebrity Image Classifier Project/ben-cornish-leonardo-dicaprio-lookalike.jpg')
results = detector.detect_faces(sample_img)

# Extract the bounding box from the face detection results
x, y, width, height = results[0]['box']
face = sample_img[y:y+height, x:x+width]

# Resize the face to the required input size for FaceNet (160x160)
image = Image.fromarray(face)
image = image.resize((160, 160))

# Convert the image to an array and preprocess it
face_array = np.asarray(image)
face_array = face_array.astype('float32')

# Expand dimensions to match FaceNet's input requirements and extract features
expanded_img = np.expand_dims(face_array, axis=0)
# preprocessed_img = facenet_model.preprocess(expanded_img)

# Use FaceNet model to predict the embedding (feature extraction)
result = facenet_model.embeddings(expanded_img).flatten()

# Compute cosine similarity between the new image embedding and all the stored embeddings
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

# Find the index of the most similar image
index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

# # Load and display the most similar image
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)
