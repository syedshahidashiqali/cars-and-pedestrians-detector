import cv2

# Our Image
image_file = "car-image.jpg"

# Our pre-trianed car classifier
classifier_file = "car-detector.xml"

# create our classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
video = cv2.VideoCapture('../Tesla Autopilot Dashcam Compilation 2021 Version-d4L1Pte7zVc.mp4')

# Runs until car stops
while True:
    
    # read the current frame
    (read_successful, frame) = video.read()
    
    if read_successful:
        gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars = car_tracker.detectMultiScale(gray_scaled_frame)
    # display img
    for(x, y, w, h) in cars:
        cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255,0,0),
        2
        )
    cv2.imshow("Shahid Car detector img", frame)

    # Dont autoclose (wait for key press)
    cv2.waitKey(1)

"""
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
"""

print("code completed")