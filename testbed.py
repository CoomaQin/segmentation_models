import cv2

image = cv2.imread("data/merge/synthetic_images/rs00042.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)