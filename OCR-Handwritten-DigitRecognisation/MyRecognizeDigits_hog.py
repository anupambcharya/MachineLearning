import numpy as np
import sys
import cv2
import imutils
import pickle as cPickle
from skimage import exposure
from skimage import feature
from numpy import asarray

gray = cv2.imread("Data\\Testing\\" + str(sys.argv[1]),0)

loaded_model = cPickle.load(open("classifier1.cPickle", 'rb'))

_, threshInv = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY_INV)
#cv2.imshow("threshInv",threshInv)
#cv2.waitKey(0)

for i in range(0, 6):
	dilated = cv2.dilate(threshInv.copy(), None, iterations=i + 1)
	dilated = 255 - dilated

#cv2.imshow("Cropped", dilated)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = gray.copy()

cv2.imshow("output", clone)
cv2.waitKey(0)

for cnt in cnts:
	#print(cnt)
	x,y,w,h = cv2.boundingRect(cnt)

	if (w<150 and w>10 and h<150 and h>40):
		crp = threshInv[y:y+h,x:x+w]
		crp = cv2.resize(crp, (20,20))
		crp = 255 - crp
		#cv2.imshow("cropped 2",crp)

		#cv2.waitKey(0)

		(H, hogImage) = feature.hog(crp, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
		pred = loaded_model.predict(H.reshape(1, -1))[0]
		#print("Alphabet : " + str(pred))
		cv2.putText(clone, pred, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(200, 200, 200), 3)
		cv2.rectangle(clone,(x,y),(x+w,y+h),(200,200,200),2)
		cv2.imshow("output", clone)

		cv2.waitKey(0)

cv2.destroyAllWindows()
