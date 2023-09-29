import cv2
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import pipeline

image = Image.open("thispersondoesnotexist.jpg")


def load_model(model_name: str):
    model = AutoModelForImageClassification.from_pretrained(model_name, from_tf=True)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return pipeline("image-classification", model=model, feature_extractor=extractor)


def take_picture(camera):
    r, i = camera.read()
    return Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))


def main():
    camera = cv2.VideoCapture(0)
    classifier = load_model("CynthiaCR/emotions_classifier")
    while input("Aby przerwać naciśnij X") != "X":
        img = take_picture(camera)
        img.show("Test")
        emotion = classifier(img)
        print(emotion)


if __name__ == "__main__":
    main()