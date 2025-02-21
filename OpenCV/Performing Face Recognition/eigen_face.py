import numpy as np
import os
import sys
import cv2

def read_images(dataset, sz=(200, 200)):
    X, y = [], []
    label = 0

    for dirname, dirnames, _ in os.walk(dataset):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    if im is None:
                        print(f"Skipping unreadable file: {filepath}")
                        continue

                    if sz:
                        im = cv2.resize(im, sz)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(label)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

            label += 1
    return [X, y]

def face_rec():
    names = ['friend1', 'friend2']  # Adjust based on your dataset

    [X, y] = read_images("dataset", sz=(200, 200))
    y = np.asarray(y, dtype=np.int32)

    try:
        model = cv2.face.EigenFaceRecognizer_create()
        model.train(X, y)
    except AttributeError:
        print("Error: EigenFaceRecognizer not found. Install OpenCV-contrib.")
        return

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Error: Haar cascade file not found.")
        return

    while True:
        ret, img = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                label_id, confidence = model.predict(face_roi)
                label = names[label_id] if label_id < len(names) else "Unknown"
                color = (0, 255, 0) if label == 'friend2' else (255, 0, 0) 

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label}, {int(confidence)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"Recognition error: {e}")

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
