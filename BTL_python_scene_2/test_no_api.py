import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import insightface
import json

# --- 1. Trích xuất embedding khuôn mặt ---
def extract_face_embedding(image_path, face_model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []

    # Nhận diện khuôn mặt trong ảnh
    faces = face_model.get(img)
    face_embeddings = [face.embedding for face in faces]

    return face_embeddings

def mask_faces(image_path, face_model):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Phát hiện khuôn mặt
    faces = face_model.get(img)

    # Tạo bản sao để che khuôn mặt
    masked_img = img.copy()
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return masked_img

def extract_scene_embedding_from_array(image_array, scene_model, preprocess):

    # Chuyển đổi mảng NumPy thành đối tượng Image của PIL
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0)

    # Trích xuất embedding
    with torch.no_grad():
        scene_embedding = scene_model(img_tensor).numpy().squeeze()

    return scene_embedding

# --- 3. Kết hợp embedding ---
def combine_face_and_scene_embeddings(face_embeddings, scene_embedding):
    combined_embeddings = [np.concatenate([face_embedding, scene_embedding]) for face_embedding in face_embeddings]
    return np.array(combined_embeddings)

# --- 4. Tìm kiếm vector giống nhất ---
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

# --- Main Function ---
def main(image_path, all_embeddings_path, metadata_path, top_n=10):
    # Load dữ liệu
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    all_embeddings = np.load(all_embeddings_path)

    # Chuẩn bị các mô hình
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

    # 1. Trích xuất embedding khuôn mặt
    face_embeddings = extract_face_embedding(image_path, face_model)

    # 2. Che mặt và trích xuất đặc trưng cảnh
    masked_image = mask_faces(image_path, face_model)
    scene_embedding = extract_scene_embedding_from_array(masked_image, scene_model, preprocess)

    # 3. Kết hợp embedding khuôn mặt và cảnh
    query_vectors = combine_face_and_scene_embeddings(face_embeddings, scene_embedding)

    # 4. Tìm kiếm vector giống nhất
    top_results, predicted_id = find_top_similar_images(query_vectors, all_embeddings, metadata, top_n)

    print(f"Predicted IMDb ID: {predicted_id}")
    for result in top_results:
        idx, similarity = result
        print(f"Index: {idx}, Similarity: {similarity:.4f}")

# --- Run ---
if __name__ == "__main__":
    # Đường dẫn dữ liệu
    image_path = "image5_2/0068646/Al_Pacino_Diane_Keaton_MV5BNjVjMjg3OGQtMTBiOS00MjMxLTk2ZmUtNDgxZDFiMTVjYmNmXkEyXkFqcGc@._V1_QL75_UX2000_.jpg"
    all_embeddings_path = "all_combined_embeddings.npy"
    metadata_path = "metadata.json"

    main(image_path, all_embeddings_path, metadata_path)
