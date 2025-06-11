#Tommy Tang
#June 2025
#Clothing Detection API

#Libraries
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json
import io
import tempfile
import os
import uvicorn

#Initialize FastAPI app
app = FastAPI(title="Clothing Detection API", description="API for detecting clothing items in images using YOLOv11 model.")

#Load model
model = YOLO("/Docker_Image/models/yolov11m-fashionpedia.pt", task="detect")

#Home route for basic API information
@app.get("/")
async def home():
    """
    Home page with API information
    """
    return {
        "message": "Clothing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/predict-image": "Upload image for clothing detection",
            "/predict-video": "Upload video for clothing detection",
            "/webcam": "Webcam streaming page",
            "ws/webcam": "WebSocket for real-time webcam detection"
        },
        "model": "YOLOv11m trained on Fashionpedia dataset"
    }

#Endpoint for image uploads
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get clothing detection results
    """
    #Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    #Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    #Run prediction 
    results = model.predict(image, conf=0.25)
    
    #Extract detection results
    detections = []
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist() # [x1, y1, x2, y2]
                }
                detections.append(detection)
                
    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }

#Endpoint for video uploads
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Upload a video and get clothing detection results for each frame
    """
    #Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        #Open video
        cap = cv2.VideoCapture(temp_file_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        frame_predictions = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            #Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #Run prediction
            results = model.predict(rgb_frame, conf=0.25, verbose=False)
            
            #Extract detections for this frame
            frame_detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "class_id": int(box.cls[0]),
                            "class_name": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()
                        }
                        frame_detections.append(detection)
            
            frame_predictions.append({
                "frame_number": frame_count,
                "detections_count": len(frame_detections),
                "detections": frame_detections
            })
            
            frame_count += 1
            
            #Limit processing to first 100 frames for demo
            if frame_count >= 100:
                break
        
        cap.release()
        
        return {
            "filename": file.filename,
            "total_frames_processed": frame_count,
            "frame_predictions": frame_predictions
        }
    
    finally:
        #Clean up temporary file
        os.unlink(temp_file_path)
        
#Endpoint for HTML page
@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page():
    """
    Serve HTML page for webcam streaming
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clothing Detection Webcam</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #video { width: 640px; height: 480px; border: 2px solid #ccc; }
            #detections { margin-top: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
            .detection-item { margin: 5px 0; padding: 5px; background: white; border-radius: 3px; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
        </style>
    </head>
    <body>
        <h1>Real-time Clothing Detection</h1>
        <video id="video" autoplay muted></video>
        <br>
        <button onclick="startWebcam()">Start Webcam</button>
        <button onclick="stopWebcam()">Stop Webcam</button>
        
        <div id="detections">
            <h3>Detected Clothing Items:</h3>
            <div id="detection-list">No detections yet...</div>
        </div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.createElement('canvas');
            let ctx = canvas.getContext('2d');
            let ws = null;
            let stream = null;

            async function startWebcam() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    
                    // Connect to WebSocket
                    ws = new WebSocket('ws://localhost:8000/ws/webcam');
                    
                    ws.onopen = function(event) {
                        console.log('WebSocket connected');
                        sendFrames();
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        displayDetections(data.detections);
                    };
                    
                    ws.onclose = function(event) {
                        console.log('WebSocket disconnected');
                    };
                    
                } catch (err) {
                    console.error('Error accessing webcam:', err);
                }
            }

            function sendFrames() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.drawImage(video, 0, 0);
                    
                    canvas.toBlob(function(blob) {
                        if (blob && ws.readyState === WebSocket.OPEN) {
                            ws.send(blob);
                        }
                    }, 'image/jpeg', 0.8);
                }
                
                setTimeout(sendFrames, 100); // Send frame every 100ms
            }

            function stopWebcam() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                if (ws) {
                    ws.close();
                }
                document.getElementById('detection-list').innerHTML = 'Webcam stopped';
            }

            function displayDetections(detections) {
                const detectionList = document.getElementById('detection-list');
                
                if (detections.length === 0) {
                    detectionList.innerHTML = 'No clothing items detected';
                    return;
                }
                
                let html = '';
                detections.forEach(detection => {
                    html += `<div class="detection-item">
                        <strong>${detection.class_name}</strong> 
                        (${(detection.confidence * 100).toFixed(1)}% confidence)
                    </div>`;
                });
                
                detectionList.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """
    WebSocket endpoint for real-time webcam detection
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive image data from client
            data = await websocket.receive_bytes()
            
            # Convert bytes to image
            image = Image.open(io.BytesIO(data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run prediction
            results = model.predict(image, conf=0.25, verbose=False)
            
            # Extract detections
            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "class_id": int(box.cls[0]),
                            "class_name": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()
                        }
                        detections.append(detection)
            
            # Send results back to client
            response = {
                "detections_count": len(detections),
                "detections": detections
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


""" 
#Old code for webcam detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam/video.")
    exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video Properties: {width}x{height}, {fps} FPS")

#Save video
fourcc = cv2.VideoWriter_fourcc(*'mp4v")
out = cv2.VideoWriter("output_detection.mp4", fourcc, fps, (width, height))

#Video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    #Predict on frame
    results = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25)
    #Plot predictions on frame
    results = results[0]
    annotated_frame = results.plot()
    #Convert back to BGR for OpenCV display
    bgr_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    #out.write(bgr_annotated_frame) # Uncomment to save video with detections
    #Show annotated frame
    cv2.imshow("Clothing Detection", bgr_annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
"""
