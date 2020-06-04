import cv2
import numpy as np 
from lib_detection import load_model, detect_lp, im2single

video = cv2.VideoCapture(0)

wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

Dmax = 608
Dmin = 288

digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

while True:
    check, frame = video.read()
    for i in frame:
        ratio = float(max(frame[i].shape[:2])) / min(frame[i].shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)

        _[i] , LpImg[i], lp_type[i] = detect_lp(wpod_net, im2single(frame[i]), bound_dim, lp_threshold=0.5)

        if (len(LpImg[i])):

            # Chuyen doi anh bien so
            LpImg[i][0] = cv2.convertScaleAbs(LpImg[i][0], alpha=(255.0))

            roi = LpImg[i][0]

            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor( LpImg[i][0], cv2.COLOR_BGR2GRAY)


            # Ap dung threshold de phan tach so va nen
            binary = cv2.threshold(gray, 127, 255,
                                cv2.THRESH_BINARY_INV)[1]

            cv2.imshow("Anh bien so sau threshold", binary)

            # Segment kí tự
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            cont, _[i]  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


            plate_info = ""

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                    if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        curr_num = np.array(curr_num,dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)

                        # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])

                        if result<=9: # Neu la so thi hien thi luon
                            result = str(result)
                        else: #Neu la chu thi chuyen bang ASCII
                            result = chr(result)

                        plate_info +=result

                        # Viet bien so len anh
                        cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

    
cv2.waitKey(0) 
video.release()