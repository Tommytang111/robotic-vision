#Robot-side streaming client for the Clothing Detection API
#Tommy Tang
#June 2025

#Libraries
import argparse
import asyncio
import json
import os
import time
import cv2
import numpy as np
import websockets

#The inference host lives somewhere else on the network, so it has to be configured
#rather than discovered. Env var first, --server flag overrides.
DEFAULT_SERVER = os.environ.get("DETECTION_SERVER", "ws://localhost:8000")

#Which endpoint each mode streams to. Both speak the same protocol - JPEG bytes up,
#JSON down - and differ only in what the detections describe.
MODE_PATHS = {
    "detect": "/ws/webcam",
    "segment": "/ws/segment",
}


def format_detections(detections, latency_ms, mode, server_mode=None):
    """
    One-line summary of what the server saw in the frame just sent
    """
    prefix = f"[{latency_ms:6.1f} ms]"
    if server_mode:
        prefix += f" {server_mode:7}"

    if not detections:
        return f"{prefix} no detections"

    if mode == "segment":
        summary = ", ".join(
            f"{d['prompt']} @({d['centroid'][0]:.0f},{d['centroid'][1]:.0f}) "
            f"{d['area_fraction']:.1%}"
            for d in detections
        )
    else:
        summary = ", ".join(f"{d['class_name']} {d['confidence']:.0%}" for d in detections)

    return f"{prefix} {summary}"


def draw_detections(frame, detections, mode):
    """
    Draw the server's geometry onto the frame. Coordinates come back in the pixel
    space of the image that was sent, so they map straight onto the raw frame.
    """
    for d in detections:
        if mode == "segment":
            color = tuple(int(c) for c in reversed(d.get("color", [0, 255, 0])))
            label = d["prompt"]

            #Outline first, so the box and label stay legible on top of it
            polygon = d.get("polygon")
            if polygon:
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], True, color, 2)

            cx, cy = d["centroid"]
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, 2)
        else:
            color = (0, 255, 0)
            label = f"{d['class_name']} {d['confidence']:.0%}"

        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


async def stream(server, camera_index, quality, show, mode):
    """
    Capture -> JPEG encode -> send -> await detections -> repeat
    """
    uri = f"{server.rstrip('/')}{MODE_PATHS[mode]}"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    #Keep the driver's buffer shallow so read() hands back a current frame
    #instead of one that has been sitting in a queue
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    frames = 0

    print(f"Connecting to {uri}")
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            print("Connected. Ctrl-C to stop.")

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed, stopping.")
                    break

                ok, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                sent_at = time.perf_counter()
                await ws.send(buffer.tobytes())

                #Wait for this frame's result before grabbing the next one. Exactly one
                #frame is ever in flight, so detections can never fall further behind the
                #camera than a single round trip, however slow the server is. Sending on a
                #fixed timer instead would let frames pile up and the lag grow without bound.
                reply = await ws.recv()
                latency_ms = (time.perf_counter() - sent_at) * 1000

                payload = json.loads(reply)
                if "error" in payload:
                    print(f"Server error: {payload['error']}")
                    break

                detections = payload["detections"]
                frames += 1
                print(format_detections(detections, latency_ms, mode, payload.get("mode")))

                #This is where the robot-side logic goes: project each bbox through a
                #depth map and the current camera pose to get a 3D target, then plan a move.
                #For now, just report what was seen.

                if show:
                    draw_detections(frame, detections, mode)
                    cv2.imshow("Clothing Detection (robot client)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except websockets.exceptions.ConnectionClosed:
        print("Server closed the connection.")
    except OSError as e:
        print(f"Could not reach {uri}: {e}")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        print(f"Processed {frames} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="Stream camera frames to the Clothing Detection API and print detections."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help="Base WebSocket URL of the inference host (default: %(default)s)")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: %(default)s)")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG encode quality, 1-100 (default: %(default)s)")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated frames in a window")
    parser.add_argument("--mode", choices=sorted(MODE_PATHS), default="detect",
                        help="detect = YOLO boxes, segment = SAM3 mask geometry "
                             "(default: %(default)s)")
    args = parser.parse_args()

    try:
        asyncio.run(stream(args.server, args.camera, args.quality, args.show, args.mode))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
