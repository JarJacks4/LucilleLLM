import torch
from model_loader import model, feature_extractor

def predict_emotion(image_pil):
    inputs = feature_extractor(images=image_pil.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_id = torch.argmax(probs).item()
    predicted_label = model.config.id2label[predicted_id]

    return predicted_label, {
        model.config.id2label[i]: float(prob) for i, prob in enumerate(probs)
    }
