import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 24) # Частота кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400) # Ширина кадров в видеопотоке.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200) # Высота кадров в видеопотоке.

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("camera", gray) # Будет показывать в оттенках серого.

while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()
cv2.destroyAllWindows()