# Detect-name-movie-from-image

## Python Version and Required Libraries

```plaintext
python==3.10.15
insightface==0.7.3
onnxruntime==1.13.1
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
opencv-python-headless==4.7.0.72
albumentations==1.3.0
fastapi==0.95.2
uvicorn==0.22.0
numpy==1.24.4

File Descriptions
Data Files
all_combined_embeddings.zip: Saves the combined embeddings of face and scene features for each movie in the training dataset.
test_all_combine_embeddings.npy: Saves the combined embeddings of face and scene features for each image from movies in the test dataset.
metadata_test.json: Saves metadata information for each image in the test dataset. It maps the image data stored in test_all_combine_embeddings.npy.
metadata.json: Saves metadata information for each vector in the training dataset. It maps the vector data stored in all_combined_embeddings.zip.

Code Files
concat_embedding.ipynb: Used to calculate the accuracy of the test dataset.
app.py: Provides an API to predict the name of a movie based on a single scene image that contains an actor's face.
test_no_api.py: Predicts the name of a movie based on a single scene image that contains an actor's face, without using the API.
