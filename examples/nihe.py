import cv2

cap = cv2.VideoCapture('IP2.mp4')
num = 0
#print(cap.get(cv2.CAP_PROP_FPS))
while True:
    try:
        ret,ori_im = cap.read()
        if not ret:
            break
        num += 1
        print(num)
    except:
        break

