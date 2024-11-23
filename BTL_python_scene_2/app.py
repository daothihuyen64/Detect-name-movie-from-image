from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import insightface
import json
import os
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],  # Thay đổi theo địa chỉ Vue.js của bạn
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả header
)
# --- Khởi tạo mô hình ---
face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=-1)

scene_model = models.efficientnet_b0(pretrained=True)
scene_model.classifier = nn.Identity()
scene_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Đường dẫn tệp dữ liệu ---
all_embeddings_path = "all_combined_embeddings.npy"
metadata_path = "metadata.json"

# Tải dữ liệu metadata và embedding
if not os.path.exists(all_embeddings_path) or not os.path.exists(metadata_path):
    raise FileNotFoundError("Metadata or embedding file not found.")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
all_embeddings = np.load(all_embeddings_path)


# --- Hàm tiện ích ---
def extract_face_embedding(image_array, face_model):
    faces = face_model.get(image_array)
    return [face.embedding for face in faces]


def mask_faces(image_array, face_model):
    faces = face_model.get(image_array)
    masked_img = image_array.copy()
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return masked_img


def extract_scene_embedding_from_array(image_array, scene_model, preprocess):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        scene_embedding = scene_model(img_tensor).numpy().squeeze()
    return scene_embedding


def combine_face_and_scene_embeddings(face_embeddings, scene_embedding):
    return [np.concatenate([face_embedding, scene_embedding]) for face_embedding in face_embeddings]


def find_top_similar_images(query_vectors, all_embeddings, metadata, top_n=10):
    all_top_results = []
    for q_vec in query_vectors:
        similarities = cosine_similarity([q_vec], all_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_n]
        for idx in top_indices:
            all_top_results.append((idx, similarities[idx]))

    all_top_results = sorted(all_top_results, key=lambda x: x[1], reverse=True)
    final_top_n = all_top_results[:top_n]
    top_indices = [idx for idx, _ in final_top_n]
    top_metadata = [metadata[i] for i in top_indices]
    imdb_ids = [meta["imdb_id"] for meta in top_metadata]
    predict_id = max(set(imdb_ids), key=imdb_ids.count)
    return final_top_n, predict_id


# --- API ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Lưu và đọc ảnh người dùng upload
    try:
        file_bytes = await file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        image_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError("Could not decode the image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the image: {str(e)}")

    # Trích xuất embedding khuôn mặt
    face_embeddings = extract_face_embedding(image_array, face_model)
    if not face_embeddings:
        return JSONResponse(content={"message": "No face detected in the image."}, status_code=400)

    # Che mặt và trích xuất embedding cảnh
    masked_image = mask_faces(image_array, face_model)
    scene_embedding = extract_scene_embedding_from_array(masked_image, scene_model, preprocess)

    # Kết hợp embedding khuôn mặt và cảnh
    query_vectors = combine_face_and_scene_embeddings(face_embeddings, scene_embedding)

    # Tìm kiếm phim tương tự
    try:
        _, predicted_id = find_top_similar_images(query_vectors, all_embeddings, metadata)
        link = f'https://www.imdb.com/title/tt{predicted_id}/'
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {str(e)}")

    return {
        "predicted_imdb_id": str(predicted_id),
         "link" : link
    }
