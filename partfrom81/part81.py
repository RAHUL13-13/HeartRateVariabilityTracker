import cv2
import dlib
import os
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")


for filename in os.listdir("/pythonProject1/HRV/png_"):
    img = cv2.imread(os.path.join("/pythonProject1/HRV/png_", filename))
    (h, w, _) = img.shape
    h2 = 600
    w2 = int(h2 * h / w)
    img = cv2.resize(img , (h2, w2))
    img = cv2.flip(img , 1)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face = faces[0]
    landmarks = predictor(gray, face)

    # pupil_x = int((abs(landmarks.part(37).x + landmarks.part(38).x + landmarks.part(40).x + landmarks.part(41).x)) / 4)
    # pupil_y = int((abs(landmarks.part(37).y + landmarks.part(38).y + landmarks.part(40).y + landmarks.part(41).y)) / 4)
    # pupil_coordination = (pupil_x, pupil_y)
    #
    # cv2.circle(img, pupil_coordination, 6, (0, 0, 255), 3)

    start_point = (int(abs(landmarks.part(1).x + landmarks.part(15).x)/8), int(abs(landmarks.part(1).y + landmarks.part(15).y)/2))
    end_point = (int(abs(landmarks.part(30).x + landmarks.part(2).x)*4/7), int(abs(landmarks.part(2).y)))
    left_cheek = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    start_point = (int(abs(landmarks.part(1).x + landmarks.part(15).x)*7/8), int(abs(landmarks.part(1).y + landmarks.part(15).y) / 2))
    end_point = (int(abs(landmarks.part(30).x + landmarks.part(14).x) * 4 / 8), int(abs(landmarks.part(2).y)))
    right_cheek = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    start_point = (int(abs(landmarks.part(21).x)), int(abs(landmarks.part(70).y)))
    end_point = (int(abs(landmarks.part(22).x)), int(abs(landmarks.part(74).y)))
    fore_head = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    cv2.imshow('Show', fore_head )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break

