import streamlit as st
from keras_facenet import FaceNet
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

# Load the embeddings and filenames
feature_list = np.array(pickle.load(open('C:/Alpesh/1stopai/Celebrity Image Classifier Project/extracted_features.pkl','rb')))
filenames = pickle.load(open('C:/Alpesh/1stopai/Celebrity Image Classifier Project/filenames.pkl','rb'))

# Initialize the FaceNet model and MTCNN detector
facenet_model = FaceNet()
detector = MTCNN()

# Streamlit app title
st.title("Celebrity Look-Alike Finder")

# Upload image section
uploaded_file = st.file_uploader("Upload an image of a person", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to an array
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Detect faces in the uploaded image
    results = detector.detect_faces(image)

    if results:
        # Extract the bounding box from the face detection results
        x, y, width, height = results[0]['box']
        face = image[y:y + height, x:x + width]

        # Resize the face to the required input size for FaceNet (160x160)
        face = Image.fromarray(face)
        face = face.resize((160, 160))

        # Convert the image to an array and preprocess it
        face_array = np.asarray(face)
        face_array = face_array.astype('float32')

        # Expand dimensions to match FaceNet's input requirements and extract features
        expanded_img = np.expand_dims(face_array, axis=0)

        # Use FaceNet model to predict the embedding (feature extraction)
        result = facenet_model.embeddings(expanded_img).flatten()

        # Compute cosine similarity between the new image embedding and all the stored embeddings
        similarity = []
        for i in range(len(feature_list)):
            similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

        # Find the index of the most similar image
        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

        # Load and display the most similar image
        temp_img = cv2.imread(filenames[index_pos])

        # Convert BGR to RGB format for display in Streamlit
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

        # Create two columns for input and output
        col1, col2 = st.columns(2)
        predicted_actor = filenames[index_pos].split('/')[5]
        with col1:
            st.header("Uploaded Image")
            st.image(image, caption="Your Uploaded Image", use_column_width=True)

        with col2:
            st.header(predicted_actor)
            st.image(temp_img, caption="Seems like "+predicted_actor+" stole your glory.", use_column_width=True)

    else:
        st.error("No face detected in the uploaded image. Please try another image.")
