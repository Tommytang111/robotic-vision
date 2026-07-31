#Prompt-driven clothing segmentation with mask tracking
#Tommy Tang
#July 2026

#Libraries
import argparse
import colorsys
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

#Weights are located by environment variable so the same code runs unchanged on a
#workstation and inside a container. The default is repo-relative; the image sets it
#to the absolute path the weights were downloaded to at build time.
SAM3_MODEL_PATH = os.environ.get("SAM3_MODEL_PATH", "models/sam3.pt")

#Floor-level clothing: what a cleaning robot needs to find, as opposed to garments
#being worn, which is what the Fashionpedia detector is trained for.
DEFAULT_PROMPTS = [
    "clothes on ground",
    "pile of clothes",
    "dropped clothing",
]


def load_predictor(model_path: Optional[str] = None, conf: float = 0.25,
                   half: bool = True, device: Optional[str] = None):
    """
    Build a SAM3SemanticPredictor.

    ultralytics is imported here rather than at module scope so this module stays
    importable without the weights or the package present. That keeps the API able
    to boot and serve its detection routes when SAM3 is unavailable.
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path or SAM3_MODEL_PATH,
        half=half,
        save=False,
        verbose=False,
    )
    if device is not None:
        overrides["device"] = device

    return SAM3SemanticPredictor(overrides=overrides)


class OptimizedSAM3Segmenter:
    """
    Real-time optimized segmentation with mask tracking.
    - Segments with SAM3 on the first frame, on a periodic refresh, on a large
      scene change, and whenever mask tracking is judged to have failed
    - Tracks masks between segmentations using dense optical flow

    Masks and boxes are stored in FULL-FRAME pixel coordinates, so tracking and
    rendering never have to reason about the inference scale.

    One instance holds tracking state across frames, so a single instance belongs to
    a single video stream and must not be shared between concurrent streams.
    """

    def __init__(
        self,
        predictor,
        clothing_prompts: List[str],
        device='cuda',
        capture_size=(640, 480),      # Lower resolution for capture
        inference_size=320,            # Even lower for SAM3 inference
        resegment_interval=150,        # Re-run SAM3 every N frames (5 sec @ 30fps)
        scene_change_threshold=0.3,    # Re-segment if >30% of pixels change
        pixel_change_threshold=25,     # Grey levels a pixel must move to count as changed
        mask_area_tolerance=2.0,       # Max area drift of a tracked mask before giving up
        use_optical_flow=True
    ):
        """
        Args:
            predictor: SAM3SemanticPredictor instance
            clothing_prompts: List of text prompts
            device: 'cuda', 'mps', or 'cpu'
            capture_size: (width, height) for webcam capture
            inference_size: Long-side size for SAM3 inference (smaller = faster)
            resegment_interval: Re-run SAM3 every N frames
            scene_change_threshold: Fraction of pixels that must change appreciably
                before forcing a re-segmentation
            pixel_change_threshold: Absolute intensity change that counts as "changed"
            mask_area_tolerance: A tracked mask may grow or shrink by at most this
                factor relative to the last segmentation before tracking is
                declared lost
            use_optical_flow: Use optical flow for mask tracking
        """
        self.predictor = predictor
        self.clothing_prompts = clothing_prompts
        self.device = device
        self.capture_size = capture_size
        self.inference_size = inference_size
        self.resegment_interval = resegment_interval
        self.scene_change_threshold = scene_change_threshold
        self.pixel_change_threshold = pixel_change_threshold
        self.mask_area_tolerance = mask_area_tolerance
        self.use_optical_flow = use_optical_flow

        # Tracking state (all at full frame resolution)
        self.masks = None          # (N, H, W) float32, binary valued
        self.boxes = None          # (N, 4) xyxy in full-frame pixels
        self.class_ids = None
        self.ref_areas = None      # Mask areas at the last segmentation
        self.prev_gray = None
        self.frame_count = 0
        self.last_segment_frame = -999

        # Performance metrics
        self.fps_history = []
        self.segment_times = []
        self.track_times = []
        self.segment_count = 0
        self.track_count = 0
        self.track_failures = 0

        # Color map
        self.color_map = {}
        for i, prompt in enumerate(clothing_prompts):
            hue = (i * 137.5) % 360
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
            self.color_map[i] = [int(r * 255), int(g * 255), int(b * 255)]

    def _run_segmentation(self, frame_small: np.ndarray, frame_shape) -> bool:
        """Run SAM3 and store masks/boxes in full-frame coordinates"""
        h, w = frame_shape[:2]
        sh, sw = frame_small.shape[:2]

        self.predictor.set_image(frame_small)
        results = self.predictor(text=self.clothing_prompts)

        if not results:
            return False

        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return False

        masks_small = result.masks.data.cpu().numpy()
        if len(masks_small) == 0:
            return False

        # Upscale masks to full frame size once, here, so tracking and rendering
        # all operate in the same coordinate space.
        self.masks = np.stack([
            (cv2.resize(m.astype(np.float32), (w, h)) > 0.5).astype(np.float32)
            for m in masks_small
        ])

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            # Scale from the ACTUAL small-frame size. inference_size is only the
            # long side, so using it for both axes squashes boxes on the short one.
            boxes[:, [0, 2]] *= w / sw
            boxes[:, [1, 3]] *= h / sh
            self.boxes = boxes
            if hasattr(result.boxes, 'cls'):
                self.class_ids = result.boxes.cls.cpu().numpy().astype(int)
            else:
                self.class_ids = np.zeros(len(self.masks), dtype=int)
        else:
            self.boxes = None
            self.class_ids = np.zeros(len(self.masks), dtype=int)

        self.ref_areas = self.masks.reshape(len(self.masks), -1).sum(axis=1)
        self.segment_count += 1
        return True

    @staticmethod
    def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
        """Tight xyxy boxes around each binary mask"""
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask > 0.5)
            if len(xs) == 0:
                boxes.append([0, 0, 0, 0])
            else:
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        return np.array(boxes, dtype=np.float32)

    def _track_masks(self, curr_gray: np.ndarray, prev_gray: np.ndarray) -> bool:
        """
        Warp masks from the previous frame into the current one.

        cv2.remap is a BACKWARD warp: for every destination pixel it needs the
        location to sample in the source. Flow computed curr -> prev is exactly
        that map, which is why the frames are passed in that order.
        """
        if self.masks is None or len(self.masks) == 0:
            return False

        h, w = curr_gray.shape

        try:
            flow = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        except cv2.error as e:
            print(f"Optical flow failed: {e}")
            return False

        h_idx, w_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        map_x = np.clip(w_idx + flow[..., 0], 0, w - 1).astype(np.float32)
        map_y = np.clip(h_idx + flow[..., 1], 0, h - 1).astype(np.float32)

        # Re-binarize after warping: repeated bilinear sampling would otherwise
        # blur the masks away over a long tracking run.
        tracked = np.stack([
            (cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR) > 0.5).astype(np.float32)
            for mask in self.masks
        ])

        # Tracking-failure check. A mask that has vanished or ballooned means the
        # flow has lost the object, and it is worth paying for a fresh segment.
        areas = tracked.reshape(len(tracked), -1).sum(axis=1)
        if np.any(areas < 1):
            return False
        if self.ref_areas is not None:
            ratio = areas / np.maximum(self.ref_areas, 1.0)
            if np.any(ratio > self.mask_area_tolerance) or \
               np.any(ratio < 1.0 / self.mask_area_tolerance):
                return False

        self.masks = tracked
        # Boxes follow the masks instead of staying frozen at the last segmentation.
        self.boxes = self._boxes_from_masks(tracked)
        self.track_count += 1
        return True

    def _detect_scene_change(self, curr_gray: np.ndarray, prev_gray: np.ndarray) -> bool:
        """
        Fraction of pixels whose intensity changed appreciably.

        Comparing mean(|diff|)/255 against 0.3 (the previous approach) requires
        the AVERAGE pixel to move ~76 grey levels, so it effectively never fired.
        """
        if prev_gray is None:
            return True

        diff = cv2.absdiff(curr_gray, prev_gray)
        change_ratio = float(np.mean(diff > self.pixel_change_threshold))

        return change_ratio > self.scene_change_threshold

    def _prompt_for(self, class_id: int) -> str:
        """Prompt text a class index refers to, tolerating an out-of-range index"""
        if 0 <= class_id < len(self.clothing_prompts):
            return self.clothing_prompts[class_id]
        return "object"

    def _extract_detections(self, frame_shape, include_polygons: bool = True,
                            polygon_epsilon: float = 2.0) -> List[Dict]:
        """
        Reduce the current masks to serialisable geometry.

        The masks themselves are deliberately not included: one 640x480 mask is
        307200 values, which is not something to put on the wire once per frame.
        What a caller actually needs to act on a garment is where it is, so each
        mask is reduced to a centroid, an extent and an optional outline.

        The centroid comes from image moments over the mask rather than the centre
        of the bounding box. On a crumpled garment the box centre can easily fall on
        bare floor between two sleeves, which would poison any depth sampled there.
        """
        if self.masks is None or len(self.masks) == 0:
            return []

        h, w = frame_shape[:2]
        frame_area = float(h * w)
        detections = []

        for i, mask in enumerate(self.masks):
            mask_binary = (mask > 0.5).astype(np.uint8)
            area_px = int(mask_binary.sum())
            if area_px == 0:
                continue

            class_id = int(self.class_ids[i]) if self.class_ids is not None and i < len(self.class_ids) else 0

            moments = cv2.moments(mask_binary, binaryImage=True)
            if moments["m00"] > 0:
                centroid = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            else:
                centroid = [0.0, 0.0]

            if self.boxes is not None and i < len(self.boxes):
                bbox = [float(v) for v in self.boxes[i]]
            else:
                bbox = [float(v) for v in self._boxes_from_masks(mask_binary[None])[0]]

            detection = {
                "class_id": class_id,
                "prompt": self._prompt_for(class_id),
                "bbox": bbox,
                "centroid": [round(c, 2) for c in centroid],
                "area_px": area_px,
                "area_fraction": round(area_px / frame_area, 5),
                "color": self.color_map.get(class_id, [255, 255, 255]),
            }

            if include_polygons:
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Largest contour only, simplified. A raw contour can run to
                    # thousands of points; approxPolyDP keeps the shape recognisable
                    # at a fraction of the payload.
                    largest = max(contours, key=cv2.contourArea)
                    simplified = cv2.approxPolyDP(largest, polygon_epsilon, True)
                    detection["polygon"] = [[int(p[0][0]), int(p[0][1])] for p in simplified]
                else:
                    detection["polygon"] = []

            detections.append(detection)

        return detections

    def _render(self, frame: np.ndarray, detections: List[Dict], mode: str,
                fps: float) -> np.ndarray:
        """Draw masks, boxes, labels and a status readout onto a copy of the frame"""
        overlay = frame.copy()

        if self.masks is not None:
            for i, mask in enumerate(self.masks):
                class_id = int(self.class_ids[i]) if self.class_ids is not None and i < len(self.class_ids) else 0
                color = self.color_map.get(class_id, [255, 255, 255])

                # Masks are already stored at full frame resolution
                mask_binary = (mask > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(frame)
                colored_mask[mask_binary > 0] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

                # Draw bounding box (already in full-frame coordinates)
                if self.boxes is not None and i < len(self.boxes):
                    x1, y1, x2, y2 = self.boxes[i].astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                    label = self._prompt_for(class_id)
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(overlay, (x1, y1 - h_text - 8), (x1 + w_text, y1), color, -1)
                    cv2.putText(overlay, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for detection in detections:
            cx, cy = detection["centroid"]
            cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        mode_color = (0, 255, 0) if mode == "TRACK" else (0, 165, 255)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Mode: {mode}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        cv2.putText(overlay, f"Detections: {len(detections)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay,
                    f"Segments: {self.segment_count} | Tracks: {self.track_count} | Lost: {self.track_failures}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return overlay

    def process_frame(self, frame: np.ndarray, render: bool = True,
                      include_polygons: bool = True) -> Dict:
        """
        Process a single frame with smart segmentation/tracking.

        Args:
            frame: BGR frame at full capture resolution
            render: Build an annotated overlay image. A server has no use for it and
                skipping it avoids an expensive per-mask blend.
            include_polygons: Include simplified mask outlines in the detections

        Returns a dict with 'detections', 'mode', 'fps', 'elapsed', and 'frame'
        (the overlay, or None when render is False).
        """
        start_time = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Determine if segmentation needs to run
        frames_since_segment = self.frame_count - self.last_segment_frame
        need_segment = (
            self.masks is None or                                # First frame
            frames_since_segment >= self.resegment_interval or   # Periodic refresh
            self._detect_scene_change(gray, self.prev_gray)      # Scene changed
        )

        mode = "SEGMENT"
        if not need_segment:
            if self.use_optical_flow and self.prev_gray is not None:
                t0 = time.time()
                if self._track_masks(gray, self.prev_gray):
                    mode = "TRACK"
                    self.track_times.append(time.time() - t0)
                else:
                    # Tracking lost the object: fall through and re-segment now.
                    self.track_failures += 1
                    mode = "SEGMENT"
            else:
                mode = "SKIP"

        if mode == "SEGMENT":
            t0 = time.time()
            # Resize frame for faster inference (long side -> inference_size)
            scale = self.inference_size / max(h, w)
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

            self._run_segmentation(frame_small, frame.shape)
            self.last_segment_frame = self.frame_count
            self.segment_times.append(time.time() - t0)

        self.prev_gray = gray

        detections = self._extract_detections(frame.shape, include_polygons=include_polygons)

        # Calculate FPS (processing only - see the capture loop for end-to-end)
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)

        return {
            'frame': self._render(frame, detections, mode, fps) if render else None,
            'fps': fps,
            'mode': mode,
            'detections': detections,
            'elapsed': elapsed,
        }


def build_segmenter(prompts: Optional[List[str]] = None, model_path: Optional[str] = None,
                    device: str = 'cpu', predictor=None, **kwargs) -> OptimizedSAM3Segmenter:
    """
    Convenience constructor: load a predictor if one is not supplied, then wrap it.

    Pass an existing predictor to share one loaded model across several segmenters,
    which is what a server does when each connection needs its own tracking state.
    """
    prompts = prompts or DEFAULT_PROMPTS
    if predictor is None:
        predictor = load_predictor(model_path=model_path, device=device)
    return OptimizedSAM3Segmenter(predictor=predictor, clothing_prompts=prompts,
                                  device=device, **kwargs)


def run_webcam(segmenter: OptimizedSAM3Segmenter, camera_id: int = 0,
               save_output: bool = False):
    """
    Local webcam loop for testing the segmenter without starting the API.

    Press 'q' to quit, 'r' to force a re-segmentation.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    w, h = segmenter.capture_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('segmentation_output.mp4', fourcc, 30, (actual_w, actual_h))

    print("Starting webcam. Press 'q' to quit, 'r' to force re-segmentation.")

    # End-to-end wall clock, including capture, display and encoding
    loop_start = None
    loop_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            if loop_start is None:
                loop_start = time.time()

            result = segmenter.process_frame(frame)
            loop_frames += 1

            if writer:
                writer.write(result['frame'])

            cv2.imshow('SAM3 Segmentation (Track + Segment)', result['frame'])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                segmenter.masks = None
                print("Forced re-segmentation")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Segment and track costs are reported separately, since a single blended
        # average just hides how expensive SAM3 actually is.
        elapsed = (time.time() - loop_start) if loop_start else 0
        end_to_end_fps = loop_frames / elapsed if elapsed > 0 else 0
        mean_segment_ms = np.mean(segmenter.segment_times) * 1000 if segmenter.segment_times else 0
        mean_track_ms = np.mean(segmenter.track_times) * 1000 if segmenter.track_times else 0

        print("\nSession Stats:")
        print(f"   Total frames: {segmenter.frame_count}  ({elapsed:.1f}s wall clock)")
        print(f"   Segmentations: {segmenter.segment_count}  (mean {mean_segment_ms:.0f} ms)")
        print(f"   Tracks: {segmenter.track_count}  (mean {mean_track_ms:.0f} ms)")
        print(f"   Tracking failures: {segmenter.track_failures}")
        print(f"   End-to-end FPS: {end_to_end_fps:.1f}")
        print(f"   Speedup: {segmenter.track_count / max(segmenter.segment_count, 1):.1f}x fewer SAM3 calls")
        if segmenter.frame_count < 100:
            print("   Short run - collect 300+ frames before quoting these numbers")


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt-driven clothing segmentation against a local camera."
    )
    parser.add_argument("--model", default=None,
                        help=f"Path to sam3.pt (default: $SAM3_MODEL_PATH or {SAM3_MODEL_PATH})")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Text prompts to segment (default: %(default)s)")
    parser.add_argument("--inference-size", type=int, default=320,
                        help="Long-side size for SAM3 inference (default: %(default)s)")
    parser.add_argument("--resegment-interval", type=int, default=150,
                        help="Re-run SAM3 every N frames (default: %(default)s)")
    parser.add_argument("--save", action="store_true", help="Write segmentation_output.mp4")
    args = parser.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    segmenter = build_segmenter(
        prompts=args.prompts,
        model_path=args.model,
        device=device,
        inference_size=args.inference_size,
        resegment_interval=args.resegment_interval,
    )
    run_webcam(segmenter, camera_id=args.camera, save_output=args.save)


if __name__ == "__main__":
    main()
