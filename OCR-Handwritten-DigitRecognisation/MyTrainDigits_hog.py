from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import pickle as cPickle
from sklearn.linear_model import LogisticRegression
from PIL import Image as im
from numpy import asarray
import numpy as np


print("[INFO] extracting features...")
data = []
labels = []


for imagePath in paths.list_images("Data\\Training"):
	digit = imagePath.split("\\")[-2]
	#print("image path : " + str(imagePath))
	gray = cv2.imread(imagePath, 0)
	H = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
					cells_per_block=(2, 2), transform_sqrt=True)
	#print(H)
	data.append(H)
	labels.append(digit)

print("[INFO] training classifier...")


model = RandomForestClassifier(n_estimators=30, random_state=42)

model.fit(data, labels)
print("[INFO] evaluating...")

f = open("classifier1.cPickle", "wb")
f.write(cPickle.dumps(model))
f.close()


act_res = []
pred_res = []


for imagePath in paths.list_images("Data\\Testing"):
	digit_val = imagePath.split("\\")[-2]
	gray = cv2.imread(imagePath)
	gray = cv2.resize(gray, (20, 20))
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	Hf = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
								cells_per_block=(2, 2), transform_sqrt=True)


	#cv2.imshow("Hog Image", hogImage)
	#cv2.waitKey(0)

	pred = model.predict(Hf.reshape(1, -1))[0]
	print("Prediction : " + str(pred))

	act_res.append(digit_val)
	pred_res.append(pred)
print(classification_report(act_res, pred_res))
