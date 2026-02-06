# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 22:03:13 2026

@author: FeudalLordS
"""

import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

threshold = 0.5

# model_path = os.path.join(os.path.dirname(__file__), "yolo26s.pt")
model = YOLO("yolo26s.pt")

uploaded_file = st.file_uploader("Bir görüntü yükleyiniz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    results = model(img)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2, score, class_id = int(x1), int(y1), int(x2), int(y2), score * 100, int(class_id)
        
        if score >= threshold:
            class_name = results.names[class_id]
            text = f"{class_name} %{score:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        

