from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from face_utils import detect_faces
from inference import predict_emotion

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    faces = detect_faces(image)
    results = []

    for idx, face in enumerate(faces):
        label, probs = predict_emotion(face)
        results.append({
            "face_number": idx + 1,
            "predicted_emotion": label,
            "probabilities": probs
        })

    return {"results": results}
