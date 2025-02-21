import cv2
import numpy as np
import os

def load_dataset(dataset, img_size=(200, 200)):
    images, labels = [], []
    label_map = {}
    label_counter = 0

    for subdir in os.listdir(dataset):
        subject_path = os.path.join(dataset, subdir)
        if not os.path.isdir(subject_path):
            continue

        if subdir not in label_map:
            label_map[subdir] = label_counter
            label_counter += 1

        for filename in os.listdir(subject_path):
            filepath = os.path.join(subject_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipping unreadable file: {filepath}")
                continue

            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label_map[subdir])

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32), label_map

def recognize_faces(model, dataset_path):
    names = ["Person1", "Person2"]

    images, labels, _ = load_dataset(dataset_path, img_size=(200, 200))
    model.train(images, labels)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                label_id, confidence = model.predict(face_roi)
                label = names[label_id] if label_id < len(names) else "Unknown"
                color = (255, 0, 0) if label == "Person2" else (0, 0, 255)

                text = "Acknowledged Person" if label == "Person2" else "Unidentified Person"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            except:
                continue

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


model1 = cv2.face.EigenFaceRecognizer_create()
model2 = cv2.face.FisherFaceRecognizer_create()
model3 = cv2.face.LBPHFaceRecognizer_create()

recognize_faces(model3, "dataset")