#3D Spatial Mapping for Robot Prototype
#Tommy Tang
#Feb 10 2026

#Libraries
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d
import time
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

print(f"PyTorch version: {torch.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

#CLASSES AND FUNCTIONS
class VisualOdometry:
    """Simple visual odometry using feature tracking"""
    
    def __init__(self):
        # ORB feature detector
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera intrinsics (approximate for webcam - will need calibration for the target camera)
        self.focal_length = 500  # pixels
        self.cx = 320  # principal point x
        self.cy = 240  # principal point y
        
        # Pose tracking
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.rotation = np.eye(3)  # rotation matrix
        
        # Previous frame data
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        
        # Movement scale (will be refined with depth)
        self.scale = 0.01  # meters per pixel movement
        
    def update(self, frame_gray, depth_map=None):
        """Update pose based on new frame"""
        
        # Detect features
        kp, desc = self.detector.detectAndCompute(frame_gray, None)
        
        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = frame_gray.copy()
            self.prev_kp = kp
            self.prev_desc = desc
            return self.position.copy(), self.rotation.copy()
        
        # Match features between frames
        if desc is not None and self.prev_desc is not None and len(kp) > 10:
            matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
            
            # Apply ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 10:
                # Get matched points
                pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                # Compute essential matrix
                K = np.array([[self.focal_length, 0, self.cx],
                              [0, self.focal_length, self.cy],
                              [0, 0, 1]])
                
                E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    # Recover pose
                    _, R_delta, t_delta, mask = cv2.recoverPose(E, pts_prev, pts_curr, K)
                    
                    # Update scale with depth if available
                    if depth_map is not None:
                        # Use median depth of matched points for scale estimation
                        depths = []
                        for pt in pts_prev:
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                                depths.append(depth_map[y, x])
                        if depths:
                            median_depth = np.median(depths)
                            self.scale = median_depth / 10.0  # adaptive scale
                    
                    # Update rotation
                    self.rotation = self.rotation @ R_delta
                    
                    # Update position
                    translation = self.scale * self.rotation @ t_delta.flatten()
                    self.position += translation
        
        # Update previous frame
        self.prev_frame = frame_gray.copy()
        self.prev_kp = kp
        self.prev_desc = desc
        
        return self.position.copy(), self.rotation.copy()
    
    def get_transform_matrix(self):
        """Get 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T
    
class OccupancyGrid3D:
    """3D voxel-based occupancy grid with probabilistic updates"""
    
    def __init__(self, voxel_size=0.05, grid_size=(200, 200, 100)):
        """
        Args:
            voxel_size: Size of each voxel in meters
            grid_size: Number of voxels in (x, y, z) dimensions
        """
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        
        # Initialize grid with log-odds (0 = unknown, >0 = occupied, <0 = free)
        self.grid = np.zeros(grid_size, dtype=np.float32)
        
        # Grid origin (center of grid)
        self.origin = np.array([grid_size[0] // 2, grid_size[1] // 2, 0])
        
        # Occupancy probability parameters
        self.prob_hit = 0.7  # Probability of hit if occupied
        self.prob_miss = 0.4  # Probability of miss if free
        self.log_odds_hit = np.log(self.prob_hit / (1 - self.prob_hit))
        self.log_odds_miss = np.log(self.prob_miss / (1 - self.prob_miss))
        
        # Thresholds
        self.occupied_thresh = 0.6
        self.free_thresh = 0.4
        
    def world_to_grid(self, points):
        """Convert world coordinates (meters) to grid indices"""
        grid_coords = (points / self.voxel_size + self.origin).astype(np.int32)
        return grid_coords
    
    def grid_to_world(self, indices):
        """Convert grid indices to world coordinates (meters)"""
        world_coords = (indices - self.origin) * self.voxel_size
        return world_coords
    
    def is_valid_index(self, indices):
        """Check if indices are within grid bounds"""
        return np.all((indices >= 0) & (indices < self.grid_size), axis=-1)
    
    def update_from_depth(self, depth_map, camera_position, camera_rotation, 
                          focal_length=500, cx=320, cy=240):
        """
        Update occupancy grid from depth map and camera pose
        
        Args:
            depth_map: HxW depth map (in meters)
            camera_position: 3D position of camera (x, y, z)
            camera_rotation: 3x3 rotation matrix
            focal_length: Camera focal length in pixels
            cx, cy: Camera principal point
        """
        h, w = depth_map.shape
        
        # Downsample for efficiency (every 4th pixel)
        step = 4
        
        for v in range(0, h, step):
            for u in range(0, w, step):
                depth = depth_map[v, u]
                
                if depth <= 0 or depth > 10:  # Ignore invalid or too far
                    continue
                
                # Back-project to 3D camera coordinates
                x_cam = (u - cx) * depth / focal_length
                y_cam = (v - cy) * depth / focal_length
                z_cam = depth
                
                point_cam = np.array([x_cam, y_cam, z_cam])
                
                # Transform to world coordinates
                point_world = camera_rotation @ point_cam + camera_position
                
                # Convert to grid coordinates
                grid_coord = self.world_to_grid(point_world)
                
                if not self.is_valid_index(grid_coord):
                    continue
                
                # Ray tracing from camera to point
                cam_grid = self.world_to_grid(camera_position)
                
                # Bresenham's line algorithm in 3D
                line_points = self.bresenham_3d(cam_grid, grid_coord)
                
                for i, pt in enumerate(line_points):
                    if not self.is_valid_index(pt):
                        continue
                    
                    x, y, z = pt
                    
                    if i < len(line_points) - 1:
                        # Free space along the ray
                        self.grid[x, y, z] += self.log_odds_miss
                        self.grid[x, y, z] = np.clip(self.grid[x, y, z], -10, 10)
                    else:
                        # Occupied at the endpoint
                        self.grid[x, y, z] += self.log_odds_hit
                        self.grid[x, y, z] = np.clip(self.grid[x, y, z], -10, 10)
    
    def bresenham_3d(self, start, end):
        """3D Bresenham's line algorithm"""
        points = []
        
        x0, y0, z0 = start
        x1, y1, z1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1
        
        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x0 != x1:
                points.append(np.array([x0, y0, z0]))
                x0 += xs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y0 != y1:
                points.append(np.array([x0, y0, z0]))
                y0 += ys
                if p1 >= 0:
                    x0 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z0 != z1:
                points.append(np.array([x0, y0, z0]))
                z0 += zs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x0 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
        
        points.append(np.array([x0, y0, z0]))
        return points
    
    def get_occupied_voxels(self):
        """Get list of occupied voxel coordinates"""
        # Convert log-odds to probability
        prob = 1 / (1 + np.exp(-self.grid))
        occupied_mask = prob > self.occupied_thresh
        occupied_indices = np.argwhere(occupied_mask)
        return occupied_indices
    
    def get_free_voxels(self):
        """Get list of free voxel coordinates"""
        prob = 1 / (1 + np.exp(-self.grid))
        free_mask = prob < self.free_thresh
        free_indices = np.argwhere(free_mask)
        return free_indices
    
    def export_point_cloud(self):
        """Export occupied voxels as point cloud for visualization"""
        occupied = self.get_occupied_voxels()
        world_points = self.grid_to_world(occupied)
        return world_points
    
def estimate_depth(frame_rgb, model, processor, device, max_depth=10.0):
    """
    Estimate depth from RGB image (works with grayscale converted to RGB)
    
    Args:
        frame_rgb: Input frame (H, W, 3)
        model: Depth model
        processor: Image processor
        device: torch device
        max_depth: Maximum depth in meters
    
    Returns:
        depth_map: Depth map in meters (H, W)
    """
    # Prepare image
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    # Convert to numpy and normalize to meters
    depth = prediction.cpu().numpy()
    
    # Normalize depth to 0-max_depth meters
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * max_depth
    
    return depth_normalized

def visualize_occupancy_grid_3d(occupancy_grid, camera_trajectory=None):
    """
    Visualize 3D occupancy grid using Open3D
    
    Args:
        occupancy_grid: OccupancyGrid3D instance
        camera_trajectory: List of camera positions (optional)
    """
    # Get occupied voxels
    occupied_points = occupancy_grid.export_point_cloud()
    
    if len(occupied_points) == 0:
        print("No occupied voxels to visualize")
        return
    
    print(f"Visualizing {len(occupied_points)} occupied voxels...")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_points)
    
    # Color by height (Z coordinate)
    colors = np.zeros_like(occupied_points)
    z_vals = occupied_points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    if z_max > z_min:
        z_normalized = (z_vals - z_min) / (z_max - z_min)
    else:
        z_normalized = np.zeros_like(z_vals)
    
    # Turbo colormap
    colors[:, 0] = z_normalized  # R
    colors[:, 1] = 1 - z_normalized  # G
    colors[:, 2] = 0.5  # B
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame at origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    
    geometries = [pcd, coordinate_frame]
    
    # Add camera trajectory if provided
    if camera_trajectory is not None and len(camera_trajectory) > 1:
        points = np.array(camera_trajectory)
        lines = [[i, i+1] for i in range(len(points)-1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red trajectory
        
        geometries.append(line_set)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Occupancy Grid",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )
    
# Load Depth Anything V2 Model for Depth Estimation
print("Loading Depth Anything V2 model...")
# Use the small model for 30 FPS performance
model_name = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_name)
depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
depth_model = depth_model.to(device)
depth_model.eval()
print("✓ Depth model loaded successfully")

if __name__ == "__main__":
    
    # Initialize components
    vo = VisualOdometry()
    occupancy_grid = OccupancyGrid3D(voxel_size=0.05, grid_size=(200, 200, 100))

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # FPS tracking
    fps_history = deque(maxlen=30)
    frame_count = 0
    update_grid_every = 3  # Update occupancy grid every N frames for performance

    print("Starting spatial mapping...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save occupancy grid")
    print("  'r' - Reset mapping")
    print("\\nMove the camera slowly around the room...")

    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Convert to grayscale (simulating B&W camera)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert grayscale to RGB for depth model
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            
            # Estimate depth
            depth_map = estimate_depth(frame_rgb, depth_model, image_processor, device)
            
            # Update visual odometry
            camera_pos, camera_rot = vo.update(frame_gray, depth_map)
            
            # Update occupancy grid (every N frames)
            if frame_count % update_grid_every == 0:
                occupancy_grid.update_from_depth(
                    depth_map, 
                    camera_pos, 
                    camera_rot,
                    focal_length=vo.focal_length,
                    cx=vo.cx,
                    cy=vo.cy
                )
            
            # Visualization
            depth_colored = visualize_depth(depth_map)
            
            # Create display frame
            display_frame = frame.copy()
            display_frame = draw_trajectory(display_frame, vo)
            
            # Add info text
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            occupied_voxels = len(occupancy_grid.get_occupied_voxels())
            
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Occupied Voxels: {occupied_voxels}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Stack displays
            top_row = np.hstack([display_frame, cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)])
            
            # Show
            cv2.imshow('Spatial Mapping - Camera + Depth', top_row)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\\nSaving occupancy grid...")
                point_cloud = occupancy_grid.export_point_cloud()
                np.save('occupancy_grid.npy', occupancy_grid.grid)
                np.save('occupied_points.npy', point_cloud)
                print(f"✓ Saved {len(point_cloud)} occupied voxels")
            elif key == ord('r'):
                print("\\nResetting mapping...")
                vo = VisualOdometry()
                occupancy_grid = OccupancyGrid3D(voxel_size=0.05, grid_size=(200, 200, 100))
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\\nCleaned up resources")
        
    # Run visualization of occupancy grid
    if 'occupancy_grid' in locals():
        visualize_occupancy_grid_3d(occupancy_grid)
    else:
        print("Run the mapping loop first to generate occupancy grid")