import cv2
import numpy as np

def create_plate_image(text, filename):
    img = np.ones((100, 800, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (20, 65), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite(filename, img)

create_plate_image("KA 03 MN 5678", "plate2.jpg")
create_plate_image("DL 8C AB 0001", "plate3.jpg")
