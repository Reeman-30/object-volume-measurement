import numpy as np
import argparse
import imutils
import cv2

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours as cnts

def midPoint(x, y):
  return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

# Arguments
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", required=True, help="path to the input image")
arg.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(arg.parse_args())

# Load the input image
img = cv2.imread(args["image"])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)

# Find the edges in the image
edged = cv2.Canny(img_gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find contours in the edged image
contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Initialize the screen contour and the list of objects
(contours, _) = cnts.sort_contours(contours)
pixelsPerMetric = None

# Initial loop for contours
for contour in contours:

  if cv2.contourArea(contour) < 100:
    continue

  img_ori = img.copy()
  box = cv2.minAreaRect(contour)
  box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
  box = np.array(box, dtype="int")

  box = perspective.order_points(box)
  cv2.drawContours(img_ori, [box.astype("int")], -1, (0, 255, 0), 2)

  for(x, y) in box:
    cv2.circle(img_ori, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midPoint(tl, tr)
    (blbrX, blbrY) = midPoint(bl, br)

    (tlblX, tlblY) = midPoint(tl, bl)
    (trbrX, trbrY) = midPoint(tr, br)

    # Draw the midpoints
    cv2.circle(img_ori, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(img_ori, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(img_ori, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(img_ori, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # Draw lines from the midpoints
    cv2.line(img_ori, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(img_ori, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # Calculate the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
      pixelsPerMetric = dB / args["width"]
    
    # Convert the distance from inches to centimeters
    cm_distA = dA / pixelsPerMetric
    cm_distB = dB / pixelsPerMetric
    
    # Display the distance
    cv2.putText(img_ori, "{:.2f} cm".format(cm_distA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(img_ori, "{:.2f} cm".format(cm_distB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Image", img_ori)
    cv2.waitKey(0)