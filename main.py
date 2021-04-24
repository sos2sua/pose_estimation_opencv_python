import cv2

protoFile = "models/openpose_pose_coco.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"

frame = cv2.imread("inputImages/5.jpg")
frameHeight, frameWidth, _ = frame.shape

inWidth = 736
inHeight = 736

pointsName = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "REye": 14, "LEye": 15,
            "REar": 16, "LEar": 17,
            "Background": 18}
numPoints = 7 #18
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()

H = output.shape[2]
W = output.shape[3]

points = []
for i in range(numPoints):
    probMap = output[0, i, :, :]

    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > 0.20:
        for key in pointsName:
            if pointsName[key]==i:
                print("Part name: "+key+" prob: "+str(prob))
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

cv2.imshow("Output",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
