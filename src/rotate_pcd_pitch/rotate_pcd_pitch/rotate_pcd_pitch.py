import math
import copy
import numpy as np
import open3d as o3d


INPUT_PCD = "/home/robot/slam_ws/PCD/test.pcd"
OUTPUT_PCD = "/home/robot/slam_ws/PCD/test_level.pcd"

# 反向补偿 10 度
PITCH_DEG = -10.0


def rotation_matrix_y(pitch_rad: float) -> np.ndarray:
    c = math.cos(pitch_rad)
    s = math.sin(pitch_rad)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=float)


def main():
    print(f"[INFO] Loading PCD: {INPUT_PCD}")
    pcd = o3d.io.read_point_cloud(INPUT_PCD)
    if pcd.is_empty():
        raise RuntimeError("Failed to load input PCD or PCD is empty.")

    pitch_rad = math.radians(PITCH_DEG)
    R = rotation_matrix_y(pitch_rad)

    center = pcd.get_center()
    print(f"[INFO] Point cloud center: {center}")
    print(f"[INFO] Applying pitch compensation: {PITCH_DEG} deg")

    pcd_rot = copy.deepcopy(pcd)
    pcd_rot.rotate(R, center=center)

    print(f"[INFO] Saving rotated PCD to: {OUTPUT_PCD}")
    ok = o3d.io.write_point_cloud(OUTPUT_PCD, pcd_rot)
    if not ok:
        raise RuntimeError("Failed to save output PCD.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()