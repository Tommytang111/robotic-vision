#Object segmentation with sam3
#Tommy Tang
#Feb 10 2026

# Libraries
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from typing import List, Dict
import matplotlib.pyplot as plt
from IPython.display import Video, display
from ultralytics.models.sam import SAM3SemanticPredictor

# Check GPU availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

class OptimizedSAM3Segmenter:
    """
    Real-time optimized segmentation with mask tracking.
    - Segments once with SAM3
    - Tracks masks across frames using optical flow
    - Re-segments only when needed (scene change, tracking loss, periodic refresh)
    """
    
    def __init__(
        self,
        predictor,
        clothing_prompts: List[str],
        device='cuda',
        capture_size=(640, 480),      # Lower resolution for capture
        inference_size=320,            # Even lower for SAM3 inference
        resegment_interval=150,        # Re-run SAM3 every N frames (5 sec @ 30fps)
        scene_change_threshold=0.3,    # Re-segment if scene changes > 30%
        use_optical_flow=True
    ):
        """
        Args:
            predictor: SAM3SemanticPredictor instance
            clothing_prompts: List of text prompts
            device: 'cuda', 'mps', or 'cpu'
            capture_size: (width, height) for webcam capture
            inference_size: Size for SAM3 inference (smaller = faster)
            resegment_interval: Re-run SAM3 every N frames
            scene_change_threshold: Trigger re-segmentation on large scene changes
            use_optical_flow: Use optical flow for mask tracking
        """
        self.predictor = predictor
        self.clothing_prompts = clothing_prompts
        self.device = device
        self.capture_size = capture_size
        self.inference_size = inference_size
        self.resegment_interval = resegment_interval
        self.scene_change_threshold = scene_change_threshold
        self.use_optical_flow = use_optical_flow
        
        # Tracking state
        self.masks = None
        self.boxes = None
        self.class_ids = None
        self.prev_gray = None
        self.frame_count = 0
        self.last_segment_frame = -999
        
        # Performance metrics
        self.fps_history = []
        self.segment_count = 0
        self.track_count = 0
        
        # Color map
        self.color_map = {}
        for i, prompt in enumerate(clothing_prompts):
            hue = (i * 137.5) % 360
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
            self.color_map[i] = [int(r*255), int(g*255), int(b*255)]
    
    def _run_segmentation(self, frame_small: np.ndarray):
        """Run SAM3 segmentation on a frame"""
        self.predictor.set_image(frame_small)
        results = self.predictor(text=self.clothing_prompts)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                self.masks = result.masks.data.cpu().numpy()
                
                if result.boxes is not None:
                    self.boxes = result.boxes.xyxy.cpu().numpy()
                    if hasattr(result.boxes, 'cls'):
                        self.class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    else:
                        self.class_ids = np.zeros(len(self.masks), dtype=int)
                else:
                    self.boxes = None
                    self.class_ids = np.zeros(len(self.masks), dtype=int)
                
                self.segment_count += 1
                return True
        
        return False
    
    def _track_masks(self, curr_gray: np.ndarray, prev_gray: np.ndarray):
        """Track masks using optical flow"""
        if self.masks is None or len(self.masks) == 0:
            return False
        
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Warp each mask using the flow
            h, w = curr_gray.shape
            tracked_masks = []
            
            for mask in self.masks:
                # Resize mask to flow dimensions
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                
                # Create mesh grid
                flow_map = np.copy(flow)
                h_indices, w_indices = np.meshgrid(
                    np.arange(h), np.arange(w), indexing='ij'
                )
                
                # Warp coordinates
                new_h = np.clip(h_indices + flow[..., 1], 0, h - 1).astype(np.float32)
                new_w = np.clip(w_indices + flow[..., 0], 0, w - 1).astype(np.float32)
                
                # Remap mask
                tracked_mask = cv2.remap(
                    mask_resized.astype(np.float32),
                    new_w, new_h,
                    cv2.INTER_LINEAR
                )
                
                tracked_masks.append(tracked_mask)
            
            self.masks = np.array(tracked_masks)
            self.track_count += 1
            return True
            
        except Exception as e:
            print(f"Tracking failed: {e}")
            return False
    
    def _detect_scene_change(self, curr_gray: np.ndarray, prev_gray: np.ndarray):
        """Detect if scene has changed significantly"""
        if prev_gray is None:
            return True
        
        # Calculate frame difference
        diff = cv2.absdiff(curr_gray, prev_gray)
        change_ratio = np.mean(diff) / 255.0
        
        return change_ratio > self.scene_change_threshold
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame with smart segmentation/tracking"""
        start_time = time.time()
        self.frame_count += 1
        
        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Determine if we need to run segmentation
        frames_since_segment = self.frame_count - self.last_segment_frame
        need_segment = (
            self.masks is None or  # First frame
            frames_since_segment >= self.resegment_interval or  # Periodic refresh
            self._detect_scene_change(gray, self.prev_gray)  # Scene changed
        )
        
        if need_segment:
            # Run full segmentation
            # Resize frame for faster inference
            scale = self.inference_size / max(frame.shape[:2])
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
            
            success = self._run_segmentation(frame_small)
            self.last_segment_frame = self.frame_count
            mode = "SEGMENT"
        else:
            # Track existing masks
            if self.use_optical_flow and self.prev_gray is not None:
                success = self._track_masks(gray, self.prev_gray)
                mode = "TRACK"
            else:
                success = False
                mode = "SKIP"
        
        self.prev_gray = gray.copy()
        
        # Render masks on frame
        overlay = frame.copy()
        detections = []
        
        if self.masks is not None and len(self.masks) > 0:
            h, w = frame.shape[:2]
            
            for i, mask in enumerate(self.masks):
                # Get color and prompt
                class_id = self.class_ids[i] if i < len(self.class_ids) else 0
                color = self.color_map.get(class_id, [255, 255, 255])
                prompt = self.clothing_prompts[class_id] if class_id < len(self.clothing_prompts) else "object"
                
                # Resize mask to frame size
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Apply colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_binary > 0] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
                
                # Draw bounding box
                if self.boxes is not None and i < len(self.boxes):
                    # Scale box to current frame size
                    box = self.boxes[i]
                    scale_x = w / self.inference_size
                    scale_y = h / self.inference_size
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{prompt}"
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(overlay, (x1, y1 - h_text - 8), (x1 + w_text, y1), color, -1)
                    cv2.putText(overlay, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                detections.append({'prompt': prompt, 'color': color})
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        
        # Add status overlay
        mode_color = (0, 255, 0) if mode == "TRACK" else (0, 165, 255)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Mode: {mode}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        cv2.putText(overlay, f"Detections: {len(detections)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Segments: {self.segment_count} | Tracks: {self.track_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'frame': overlay,
            'fps': fps,
            'mode': mode,
            'detections': detections,
            'elapsed': elapsed
        }

def process_webcam_optimized(segmenter, camera_id=0, save_output=False):
    """
    Optimized real-time webcam processing with mask tracking.
    Much faster than running SAM3 on every frame!
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    # Set camera properties to match segmenter
    w, h = segmenter.capture_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Camera resolution: {actual_w}x{actual_h}")
    
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('optimized_webcam_output.mp4', fourcc, 30, (actual_w, actual_h))
    
    print(".  Starting optimized webcam...")
    print("   Press 'q' to quit")
    print("   Press 'r' to force re-segmentation")
    print("   Press 's' to toggle save output")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame (will auto-segment or track)
            result = segmenter.process_frame(frame)
            result_frame = result['frame']
            
            # Save if requested
            if writer:
                writer.write(result_frame)
            
            # Display
            cv2.imshow('Optimized SAM3 Segmentation (Track + Segment)', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Force re-segmentation
                segmenter.masks = None
                print("🔄 Forced re-segmentation")
            elif key == ord('s'):
                # Toggle save
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter('optimized_webcam_output.mp4', fourcc, 30, (actual_w, actual_h))
                    print("💾 Recording started")
                else:
                    writer.release()
                    writer = None
                    print("⏹️  Recording stopped")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print stats
        avg_fps = np.mean(segmenter.fps_history) if segmenter.fps_history else 0
        print(f"\n Session Stats:")
        print(f"   Total frames: {segmenter.frame_count}")
        print(f"   Segmentations: {segmenter.segment_count}")
        print(f"   Tracks: {segmenter.track_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Speedup: {segmenter.track_count / max(segmenter.segment_count, 1):.1f}x fewer SAM3 calls")

if __name__ == "__main__":
    
    # Initialize SAM3 semantic predictor
    overrides = dict(
        conf=0.25,              # Confidence threshold
        task="segment",
        mode="predict",
        model="/Users/tommy/Projects/drone/models/sam3.pt",  # Full path to model
        half=True,              # Use FP16 for faster inference on GPU
        save=False,
        verbose=False
    )

    # Use SAM3SemanticPredictor for frame-by-frame (webcam) processing
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Define which clothing items to detect
    selected_prompts = [
    "clothes on ground",
    "pile of clothes",
    "dropped clothing"
    ]

    # Initialize Optimized Segmenter
    optimized_segmenter = OptimizedSAM3Segmenter(
        predictor=predictor,
        clothing_prompts=selected_prompts,
        device=device,
        capture_size=(640, 480),      # Webcam capture resolution
        inference_size=320,            # SAM3 inference resolution (lower = faster)
        resegment_interval=15000,        # Re-segment every 150 frames (~5 sec)
        scene_change_threshold=0.3,    # Re-segment on 30% scene change
        use_optical_flow=True          # Use optical flow tracking
    )

    print(f"Optimized segmenter initialized with:")
    print(f"Capture: {optimized_segmenter.capture_size}")
    print(f"Inference: {optimized_segmenter.inference_size}px") 
    print(f"Re-segment every: {optimized_segmenter.resegment_interval} frames")
    print(f"Tracking: {'ON' if optimized_segmenter.use_optical_flow else 'OFF'}")

    # Run the webcam segmentation
    process_webcam_optimized(optimized_segmenter, camera_id=0, save_output=False)