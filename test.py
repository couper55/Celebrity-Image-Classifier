# # @title Dataset Loading (optional) and Preprocessing
# import os
# import cv2
# from mtcnn import MTCNN

# # Initialize MTCNN detector
# detector = MTCNN()

# # Paths
# input_dir = '/content/drive/MyDrive/Celebrity Faces Dataset'           # Root directory containing actor images
# output_dir = '/content/output_faces'      # Directory to save cropped faces

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# def process_and_save_faces(image_path, output_path):
#     """Detect and save faces from an image."""
#     # Load and convert image to RGB
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Detect faces in the image
#     faces = detector.detect_faces(image_rgb)

#     if len(faces) > 0:
#         # Extract the first detected face (modify if multiple faces needed)
#         x, y, width, height = faces[0]['box']
#         face = image_rgb[y:y + height, x:x + width]

#         # Resize face to 224x224
#         face_resized = cv2.resize(face, (224, 224))

#         # Save the cropped face
#         cv2.imwrite(output_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
#         print(f'Saved: {output_path}')
#     else:
#         print(f'No face detected in {image_path}')

# def traverse_and_process(input_dir, output_dir):
#     """Traverse directories and process images."""
#     for root, _, files in os.walk(input_dir):
#         # Create a corresponding output subdirectory
#         relative_path = os.path.relpath(root, input_dir)
#         output_subdir = os.path.join(output_dir, relative_path)
#         os.makedirs(output_subdir, exist_ok=True)

#         # Process each image in the current directory
#         for file in files:
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 input_path = os.path.join(root, file)
#                 output_path = os.path.join(output_subdir, file)
#                 process_and_save_faces(input_path, output_path)

# # Run the processing function
# traverse_and_process(input_dir, output_dir)

# @title Dataset Loading (optional)
# import os
# import pickle
# actors = os.listdir('/content/drive/MyDrive/output_faces')
# print(actors)

# filenames = []
# for actor in actors:
#     actor_path = os.path.join('/content/drive/MyDrive/output_faces', actor)
#     actor_files = os.listdir(actor_path)
#     for file in actor_files:
#         filenames.append(os.path.join(actor_path, file))

# pickle.dump(filenames, open('filenames.pkl', 'wb'))

# @title Feature Extraction
# from keras_facenet import FaceNet  # Use keras_facenet package for FaceNet
# from keras.preprocessing import image
# import numpy as np
# import pickle
# from tqdm import tqdm

# # Load the filenames
# filenames = pickle.load(open('/content/filenames.pkl', 'rb'))

# # Load the FaceNet model
# facenet_model = FaceNet()

# # Function to extract features using FaceNet
# def feature_extractor(img_path, model):
#     # Load and preprocess image
#     img = image.load_img(img_path, target_size=(160, 160))  # FaceNet typically uses 160x160 image input
#     img_array = image.img_to_array(img)
#     expanded_img = np.expand_dims(img_array, axis=0)

#     # Get the embedding (512-dimensional vector)
#     embeddings = facenet_model.embeddings(expanded_img)

#     # Return the flattened embeddings
#     return embeddings.flatten()

# # Extract features from the images
# features = []
# for file in tqdm(filenames):
#     features.append(feature_extractor(file, facenet_model))

# # Save the embeddings to a file
# pickle.dump(features, open('extracted_features.pkl', 'wb'))

"""Optional 👆"""


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
