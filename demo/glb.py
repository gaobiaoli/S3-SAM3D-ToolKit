"""Load, pose, inspect, and optionally display a SAM3D GLB asset."""

from pathlib import Path

from s3dis_sam3d import GLBMesh
from s3dis_sam3d.pointcloud import visualize_point_clouds

DEMO_DIR = Path(r"C:\Users\bgao491\pythonProject\outputs\sam3d_inference\a1_office_1_pano_fuse")
GLB_PATH = DEMO_DIR / "camera_14c620a723e54e8cb6847b4a0c532dca_office_1_frame_4_domain_rgb_chair_2.glb"
POSE_PATH = DEMO_DIR / "camera_14c620a723e54e8cb6847b4a0c532dca_office_1_frame_4_domain_rgb_chair_2.json"
# OUTPUT_PATH = DEMO_DIR / "outputs" / "prediction_posed.glb"
SHOW = True


def main():
    asset = GLBMesh.load_posed(GLB_PATH, POSE_PATH, pose_space="auto")
    bounds = asset.bounding_box()
    print(f"transform:\n{asset.transform_matrix}")
    print(f"bounds: {bounds.min_bound} -> {bounds.max_bound}")
    # print(f"exported: {asset.export(OUTPUT_PATH)}")
    if SHOW:
        visualize_point_clouds(
            [asset],
            camera_view=True,
        )
        # asset.visualize(show_axis=False, **{"camera_view": True})


if __name__ == "__main__":
    main()
