import cv2

# Our Image
image_file = "car-image.jpg"

# Our pre-trianed car classifier
classifier_file = "car-detector.xml"

# create opencv image
img = cv2.imread(image_file)

# convert to grayscale (need for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create our classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars

# multiscale means detect car of any size or of any scale
#  This function will return a rectangle with coordinates(x,y,w,h) around the detected cars .
cars = car_tracker.detectMultiScale(black_n_white)
print(cars)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
  cv2.rectangle(
    img,
    (x, y),
    (x + w, y + h),
    (255,0,0),
    2
    )
"""
# Explanation
# cv2.rectangle(image, start_point, end_point, color, thickness)
# car1 =cars[0]
# (x, y, w, h) = car1
# cv2.rectangle(
#     img,
#     (x, y),
#     (x + w, y + h),
#     (255,0,0),
#     2
# )
"""
# display img
cv2.imshow("Shahid Car detector img", img)

# Dont autoclose (wait for key press)
cv2.waitKey()

print("code completed")