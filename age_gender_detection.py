import cv2
import numpy as np
import io
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

app = FastAPI()

faceProto = "model/opencv_face_detector.pbtxt"
faceModel = "model/opencv_face_detector_uint8.pb"

ageProto = "model/age_deploy.prototxt"
ageModel = "model/age_net.caffemodel"

genderProto = "model/gender_deploy.prototxt"
genderModel = "model/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append((x1, y1, x2, y2,i))
    return frame, faceBoxes

def base64_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    return cv2.imdecode(np.fromstring(imgdata, np.uint8), cv2.IMREAD_COLOR)

@app.post("/predict/")
async def predict_age_gender(img: dict):
    base64_string = img.get("img")
    if not base64_string:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="No image provided")

    frame = base64_to_image(base64_string)
    frame, faceBoxes = getFaceBox(faceNet, frame)

    if not faceBoxes:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="No face detected")
    
    results = []
    for (x1,y1,x2,y2,face_id) in faceBoxes:
        face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
               max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        results.append({"face_id": face_id,"gender": gender, "age": age, "coordinates": [x1, y1, x2, y2]})

    return JSONResponse(content=results)

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)