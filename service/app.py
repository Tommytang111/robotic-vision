#Tommy Tang
#June 2025
#Clothing Detection API Prototype

#Libraries
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import io
import tempfile
import os
import uvicorn

from segmentation import DEFAULT_PROMPTS, OptimizedSAM3Segmenter, load_predictor

#Initialize FastAPI app
app = FastAPI(title="Clothing Detection API", description="API for detecting clothing items in images using YOLOv11 model.")

#Weights are located by environment variable so the same code runs unchanged on a
#workstation and inside a container. The default is repo-relative; the image sets it
#to the absolute path the weights were downloaded to at build time.
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH", "models/yolov11m-fashionpedia.pt"
)

#Loaded at import rather than on demand: an instance with no detector is of no use to
#anyone, so it is better to fail at startup than to accept traffic that cannot be served.
model = YOLO(YOLO_MODEL_PATH, task="detect")

#SAM3 is loaded on first use rather than at import, so the API still boots and serves
#its detection routes when the segmentation weights are missing.
_sam3_predictor = None

#SAM3SemanticPredictor carries per-image state through set_image(), so concurrent
#connections sharing one predictor would interleave and corrupt each other's results.
sam3_lock = asyncio.Lock()


def get_predictor():
    """
    Return the process-wide SAM3 predictor, loading it on first call
    """
    global _sam3_predictor
    if _sam3_predictor is None:
        _sam3_predictor = load_predictor()
    return _sam3_predictor

#Liveness probe. Deliberately does no inference: a probe that ran the model would turn
#every health check into GPU/CPU work and would report unhealthy under nothing worse
#than load.
@app.get("/health")
async def health():
    """
    Liveness check for container orchestrators
    """
    return {"status": "ok", "detector_loaded": model is not None}

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
            "/health": "Liveness check",
            "/predict-image": "Upload image for clothing detection",
            "/predict-video": "Upload video for clothing detection",
            "/webcam": "Webcam streaming page",
            "ws/webcam": "WebSocket for real-time webcam detection",
            "ws/segment": "WebSocket for real-time prompt-driven clothing segmentation"
        },
        "models": {
            "detection": "YOLOv11m trained on Fashionpedia dataset",
            "segmentation": "SAM3 semantic predictor, loaded on first use"
        }
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
                    
                    // Connect to WebSocket on whatever host served this page
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/webcam`);
                    
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

def detect_frame(image):
    """
    Run the detector on one image and return serialisable detections.

    Kept synchronous so it can be handed to a worker thread: ultralytics blocks,
    and calling it directly inside a coroutine would stall every other connection.
    """
    results = model.predict(image, conf=0.25, verbose=False)

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

    return detections

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

            # Run prediction off the event loop
            detections = await asyncio.to_thread(detect_frame, image)

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

@app.websocket("/ws/segment")
async def websocket_segment(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prompt-driven clothing segmentation.

    Returns mask geometry rather than masks. Each detection carries a bounding box,
    a mask centroid, a pixel area and a simplified outline, which is what a caller
    needs to locate a garment; the mask itself is far too large to send per frame.
    """
    await websocket.accept()

    #Tracking state lives on the segmenter and must persist between frames but never
    #leak between clients, so each connection gets its own. The loaded model is shared.
    #Loading happens in a worker thread under the lock: the first connection would
    #otherwise stall every other client while the weights load, and two simultaneous
    #first connections would each load their own copy.
    try:
        async with sam3_lock:
            predictor = await asyncio.to_thread(get_predictor)

        segmenter = OptimizedSAM3Segmenter(
            predictor=predictor,
            clothing_prompts=DEFAULT_PROMPTS,
            device="cpu",
        )
    except Exception as e:
        print(f"SAM3 unavailable: {e}")
        await websocket.send_text(json.dumps({"error": f"SAM3 model unavailable: {e}"}))
        await websocket.close()
        return

    try:
        while True:
            # Receive image data from client
            data = await websocket.receive_bytes()

            # Decode to a BGR frame, which is what the segmenter expects
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Could not decode frame"}))
                continue

            # Serialise SAM3 access across connections, and keep the blocking call
            # off the event loop
            async with sam3_lock:
                result = await asyncio.to_thread(
                    segmenter.process_frame, frame, False
                )

            response = {
                "mode": result["mode"],
                "detections_count": len(result["detections"]),
                "detections": result["detections"]
            }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Segmentation WebSocket disconnected")
    except Exception as e:
        print(f"Segmentation WebSocket error: {e}")

if __name__ == "__main__":
    #Container platforms inject the port to listen on. Binding 0.0.0.0 is what makes the
    #server reachable from outside the container at all.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))