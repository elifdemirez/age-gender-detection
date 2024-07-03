import streamlit as st
import requests
import base64
import numpy as np
import cv2
from PIL import Image
import io

def send_request(image_data):
    base64_image = base64.b64encode(image_data).decode()
    data={'img': base64_image}
    response=requests.post('http://127.0.0.1:8000/predict/', json=data)
    if response.status_code==200:
        results=response.json()
        return results
    else:
        st.error(f"Error: {response.status_code}")
    
def draw_rectangles(image_array, results):
    for result in results:
        face_id = result["face_id"]
        coordinates = result["coordinates"]
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_array, str(face_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    return image_array

def main():
    st.title("Age and Gender Prediction Streamlit Application")
    uploaded_file=st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_array = np.array(Image.open(io.BytesIO(image_data)))
        st.image(image_data, caption="Uploaded Image", use_column_width=True)
        

        if st.button("Predict"):
            response_data=send_request(image_data)
            image_array_with_rectangles = draw_rectangles(image_array.copy(), response_data)
            st.image(image_array_with_rectangles, caption="Predicted Image with Rectangles", use_column_width=True)
            for i, result in enumerate(response_data):
                face_id = result["face_id"]
                gender = result["gender"]
                age = result["age"]
                coordinates = result["coordinates"]
                st.write(f"result {i+1}: Face ID: {face_id}, Gender: {gender}, Age: {age}, Coordinates: {coordinates}")
            
if __name__=="__main__":
    main()