"""Open a configurable Open3D window for an S3DIS room."""

from s3dis_sam3d import S3DISDataset
from s3dis_sam3d.config import MINIMAL_S3DIS_ROOT

DATASET_ROOT = MINIMAL_S3DIS_ROOT

DEFAULT_AREA = "Area_4"
DEFAULT_ROOM = "hallway_5"
DEFAULT_COLOR_MODE = "rgb"  # rgb, semantic, instance
DEFAULT_HIDDEN_CLASSES = []
DEFAULT_HIDDEN_INSTANCES = []
DEFAULT_BBOX_INSTANCES = []
DEFAULT_MAX_POINTS = None

# Paste parameters printed by get_parameters=True here to reuse a fixed view.
DEFAULT_VIEW_PARAMETERS = None


# def prepare_visualization(
#     area=DEFAULT_AREA,
#     room=DEFAULT_ROOM,
#     color_mode=DEFAULT_COLOR_MODE,
#     hidden_classes=DEFAULT_HIDDEN_CLASSES,
#     hidden_instances=DEFAULT_HIDDEN_INSTANCES,
#     max_points=DEFAULT_MAX_POINTS,
# ):
#     return S3DISDataset(DATASET_ROOT).room(f"{area}/{room}").point_cloud(
#         color_mode=color_mode,
#         exclude_classes=hidden_classes,
#         exclude_instances=hidden_instances,
#         ignore_missing_instances=True,
#         max_points=max_points,
#     )


if __name__ == "__main__":
    room = f"{DEFAULT_AREA}/{DEFAULT_ROOM}"
    print(f"Opening Open3D: {room}")
    S3DISDataset(DATASET_ROOT).room(room).visualize(
        color_mode=DEFAULT_COLOR_MODE,
        hidden_classes=DEFAULT_HIDDEN_CLASSES,
        hidden_instances=DEFAULT_HIDDEN_INSTANCES,
        bbox_instances=DEFAULT_BBOX_INSTANCES,
        max_points=DEFAULT_MAX_POINTS,
        point_size=2.0,
        get_parameters=True,
        set_parameters=DEFAULT_VIEW_PARAMETERS,
    )
