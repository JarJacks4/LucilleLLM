from transformers import AutoModelForImageClassification, AutoFeatureExtractor

model_name = "trpakov/vit-face-expression"

model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
