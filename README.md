# Detect-name-movie-from-image

## Python Version and Required Libraries
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

## File Descriptions

### data
all_combined_embeddings.zip            : Save the combined embeddings of face and scene features for each movie in the training dataset. 
test_all_combinecombine_embeddings.npy : Save the combined embeddings of face and scene features for each image from movies in the test dataset.
metadata_test.json                     : Save metadata information for each image in the test dataset. It maps the image data stored in test_all_combine_embeddings.npy 
metadata.json                          : Save metadata information for each vecto in the train dataset. It maps the vecto data stored in all_combine_embeddings.zip 

### code
concat_embedding.ipynb                 : is used to calculate the accuracy of the test dataset
app.py                                 : api for predict the name of a movie based on a single scene image that contains actor's face.
test_no_api.py                         : predict the name of a movie based on a single scene image that contains actor's face with no api.
