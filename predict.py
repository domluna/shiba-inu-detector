import fire
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


def load_model():
    pass

def predict(image_file):
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    img = Image.open(image_file)

    inputs = feature_extractor(img, return_tensors='pt')
    outputs = model(**inputs)

    print(outputs.logits)
    preds = outputs.logits.argmax(-1)

    print(preds.item())

if __name__ == "__main__":
    fire.Fire(predict)
