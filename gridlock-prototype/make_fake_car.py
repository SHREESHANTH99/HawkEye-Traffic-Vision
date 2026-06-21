import cv2

car = cv2.imread("static/violations/car_crop.jpg")
plate = cv2.imread("static/violations/clear_plate.jpg")

# resize plate to fit somewhere on the car
plate = cv2.resize(plate, (50, 20))
car[car.shape[0]-25:car.shape[0]-5, car.shape[1]//2-25:car.shape[1]//2+25] = plate

cv2.imwrite("static/violations/fake_car.jpg", car)
print("Created fake_car.jpg")
