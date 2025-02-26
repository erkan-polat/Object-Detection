# USAGE
# python yolo.py --image images/baggage_claim.jpg
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = 'yolo-coco\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = 'yolo-coco\yolov3.weights'
configPath = 'yolo-coco\yolov3.cfg'


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

if isinstance(unconnected_layers, list) and len(unconnected_layers) > 0:
    ln = [layer_names[i[0] - 1] for i in unconnected_layers]
else:
    ln = [layer_names[i - 1] for i in unconnected_layers.flatten()]



blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
# [,frame,no of detections,[classid,class score,conf,x,y,h,w]
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:

		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]


		if confidence > args["confidence"]:

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")


			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))


			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

if len(idxs) > 0:

	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)