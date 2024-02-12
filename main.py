import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
# import easyocr
from detector import detector
from pydantic import BaseModel
from fastapi import FastAPI
import urllib.request
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://checkd.vercel.app"
    "https://anpr-1lby.onrender.com"
]

# Add the CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageDetail(BaseModel):
    image_url: str

@app.get("/test")
def test():
    return {
        "status": "Works Fine! 👍"
    }

@app.post("/detect")
def detect(imageDetails: ImageDetail):
    
    img_url = imageDetails.image_url

    # Download the image from the URL
    response = urllib.request.urlopen(img_url)
    img_bytes = response.read()

    # Convert the image bytes to a numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # img = cv2.imread('./images/'+img_url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
            
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    try:
        image_url = './number_plate_extracted/cropped_image.jpg'
        cv2.imwrite(image_url, cropped_image)
        carDetail = ImageDetail(image_url = image_url)
        res = detector(carDetail)
        return {
            "status": "Success",
            "value": res[0].replace(' ', ''),
            "model": "1"
        }

    except:
        try:
            carDetail = ImageDetail(image_url = img_url)
            res = detector(carDetail)
            return {
                "status": "Success",
                "value": res[0].replace(' ', ''),
                "model": "2"
            }
            
            # reader = easyocr.Reader(['en'])
            # result = reader.readtext(cropped_image)
            # return {
            #     "status": "Success",
            #     "value": result[0][2].replace(' ', ''),
            #     "model": "2"
            # }
        except:
            return {
                "status": "Error"
            }

# imageDetails = ImageDetail(image_url = 'image4.jpg')
# detect(imageDetails)
