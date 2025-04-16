import cv2
import dlib
import numpy as np
import winsound
from keras.models import load_model
from imutils import face_utils

# Cấu hình
EYE_CLOSED_LABEL = 0  # 0 = mắt nhắm, 1 = mắt mở
EYE_AR_CONSEC_FRAMES = 15  # Số frame liên tiếp mắt nhắm để cảnh báo

# Load model
model = load_model("weights.149-0.01.hdf5")

# Load Dlib detector + predictor
predictor = dlib.shape_predictor("68_face_landmarks_predictor.dat")
detector = dlib.get_frontal_face_detector()

# Index mắt
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Hàm dự đoán mắt
def predict_eye_state(model, eye_img):
    eye_img = cv2.resize(eye_img, (20, 10))
    eye_img = eye_img.astype(np.float32) / 255.0
    eye_img = np.reshape(eye_img, (1, 10, 20, 1))
    prediction = model.predict(eye_img, verbose=0)
    return np.argmax(prediction[0])  # 0 = nhắm, 1 = mở

# Webcam
cap = cv2.VideoCapture(0)
frame_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        left_eye_pts = shape_np[lStart:lEnd]
        right_eye_pts = shape_np[rStart:rEnd]

        def crop_eye(eye_pts):
            x, y, w, h = cv2.boundingRect(eye_pts)
            eye_img = gray[y:y + h, x:x + w]
            return eye_img

        left_eye_img = crop_eye(left_eye_pts)
        right_eye_img = crop_eye(right_eye_pts)

        left_state = predict_eye_state(model, left_eye_img)
        right_state = predict_eye_state(model, right_eye_img)

        left_status = "Dong" if left_state == EYE_CLOSED_LABEL else "Mo"
        right_status = "Dong" if right_state == EYE_CLOSED_LABEL else "Mo"

        print(f"Mắt trái: {left_status} | Mắt phải: {right_status}")

        cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.putText(frame, f"Phai: {left_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Trai: {right_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if left_state == EYE_CLOSED_LABEL and right_state == EYE_CLOSED_LABEL:
            frame_count += 1
            if frame_count >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "BUON NGU!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(4000, 800)

        else:
            frame_count = 0


    cv2.imshow("Canh bao ngu gat", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
