# S3DIS + SAM3D Toolkit

一个自用的 Python 工具库，用于处理 S3DIS、Stanford 2D-3D-S 数据，以及 SAM3D 输出的 GLB 和位姿。

仓库内的 `dataset/` 提供最小真实数据，可直接运行示例；完整数据集、SAM3D 权重和推理服务不包含在项目中。

## 安装

Python 3.9 及以上：

```bash
conda activate 3d
python -m pip install -e .
```

读取 2D-3D-S 的 `global_xyz/*.exr` 时额外安装：

```bash
python -m pip install -e ".[exr]"
```

处理 BIMSync IFC 时安装 IfcOpenShell：

```bash
python -m pip install -e ".[ifc]"
```

## 目录

```text
s3dis-sam3d-toolkit/
├── src/s3dis_sam3d/
│   ├── config.py         # S3DIS、2D-3D-S、BIMSync 默认路径
│   ├── s3dis.py          # S3DISDataset、S3DISRoom、S3DISInstance
│   ├── s23dis.py         # 2D-3D-S
│   ├── annotations.py    # S3DIS instance 到图像 bbox 的批量标注
│   ├── bimsync.py        # BIMSyncDataset、BIMSyncScene 与 IFC/S3DIS 配准
│   ├── utils_ifc.py      # IFC 类型筛选、语义映射与 mesh 加载
│   ├── models.py         # PointCloud、BoundingBox3D、GLBMesh
│   ├── pointcloud.py     # 下采样、变换、Open3D 可视化
│   ├── rendering.py      # FrameRender 与共享渲染结果约定
│   ├── utils.py          # 相机、投影/反投影、配准等通用数学函数
│   └── sam3d/            # SAM3D 请求与位姿转换
├── dataset/
│   ├── s3dis/
│   └── 2d3ds/
├── demo/
├── examples/
├── script/
└── tests/
```

## 数据集路径配置

完整数据集路径统一定义在
[`src/s3dis_sam3d/config.py`](src/s3dis_sam3d/config.py)，更换电脑或数据目录时只需修改：

```python
S23DIS_ROOT = Path("path/to/2d3ds")
S3DIS_ROOT = Path("path/to/Stanford3dDataset_v1.2")
BIMSYNC_ROOT = Path("path/to/bimsync/ifc")
BIMNET_ROOT = Path("path/to/BIMNet_release")
BIMSYNC_CALIBRATION_ROOT = BUNDLED_DATASET_ROOT / "ifc_to_s3dis"
```

省略数据集路径时会读取上述默认值：

```python
from s3dis_sam3d import BIMSyncDataset, S3DISDataset, S23Dataset

s3dis = S3DISDataset()
s23dis = S23Dataset(area="Area_1")
bimsync = BIMSyncDataset(area="Area_1")
```

`BIMSyncDataset()` 使用默认 BIMSync 路径时，会从
`dataset/ifc_to_s3dis/<Area>_upright_v2` 自动加载校准矩阵；该目录不存在或不包含校准矩阵时会立即报错。
仓库内最小示例使用同一配置文件中的
`MINIMAL_S3DIS_ROOT` 和 `MINIMAL_S23DIS_ROOT`，不会与完整数据路径混淆。

## S3DIS

```python
from s3dis_sam3d import S3DISDataset

dataset = S3DISDataset("dataset/s3dis")
room = dataset.room("Area_4/hallway_5")
cloud = room.point_cloud()
instance = room.instance("clutter_2")
print(instance.point_cloud.xyz.shape, instance.bbox.extent)
print(cloud.xyz.shape)

room.visualize(
    color_mode="semantic",
    hidden_classes=["door", "clutter", "board", "beam", "ceiling", "column"],
    hidden_instances=["wall_3", "wall_4"],
)
```

## 2D-3D-S

```python
from s3dis_sam3d import S23Dataset

dataset = S23Dataset("dataset/2d3ds/area_1")
room = dataset.room(0)
frame = room.frames[0]
cloud = frame.point_cloud(stride=4)
print(cloud.xyz.shape)

# UUID 可省略，默认按排序结果选择第一个
frame = room.get_frame(frame.frame_id)
print(room.list_uuids(frame.frame_id))

# 常用单帧数据
rgb = frame.rgb
pose = frame.pose
depth = frame.depth if frame.has_depth else None
xyz = frame.xyz if frame.has_xyz else None
point_map = frame.point_map()
intrinsics = frame.intrinsics
camera_to_world = frame.camera_to_world
world_to_camera = frame.world_to_camera

# 房间级可视化与保存
cloud = room.reconstruct(stride=8)
room.visualize(stride=8)
room.save_ply("room.ply", stride=8, progress=True)
```

`Frame` 自己持有 `projection_type`，负责读取 RGB、pose、depth、global XYZ，以及单帧投影、
反投影、point map 和点云生成；`S23Room` 持有 frames 并负责房间重建、可视化和导出，
`S23Dataset` 只负责索引和选择 room。`mask` 可以是数组、图片路径、
`{frame.stem: mask}` 映射或接收 `Frame` 的回调。regular 与 pano 会自动使用各自的反投影模型。

相机变换、针孔/全景投影与反投影、mask 处理、单位归一化和 ICP 等不依赖数据集目录的
函数位于 `s3dis_sam3d.utils`：

```python
from s3dis_sam3d.utils import (
    backproject_regular,
    camera_to_world_from_pose,
    euler_xyz_to_matrix,
    project_pinhole_points,
)
```

这些数学函数不再作为 `S23Dataset` 的别名重复暴露。

## BIMNet

`BIMNetDataset` indexes all 25 train/test scenes and keeps BIMNet's parallel
asset trees together: IFC, component-level OBJ, wall-filled OBJ, point-cloud to
OBJ matrices, labeled point clouds, room metadata, and optional RVT files.

```python
from s3dis_sam3d import BIMNetDataset, Matterport3DDataset

bimnet = BIMNetDataset(r"C:\Users\bgao491\DepthEstimation\BIMNet_release")
matterport_dataset = Matterport3DDataset(
    r"C:\Users\bgao491\DepthEstimation\Matterport3D"
)
scene = bimnet["hxp"]

print(scene.key, scene.matterport_scan_id, scene.availability)
print(len(scene.instances), len(scene.rooms))

# Component metadata and selective OBJ loading.
walls = scene.elements(["IfcWall", "IfcWallStandardCase"])
wall = scene.element(walls[0].guid)
wall_mesh = wall.mesh()
structural_mesh = scene.mesh(
    source="obj",
    include_types=["IfcWall", "IfcWallStandardCase", "IfcSlab"],
)

# Preserve the selected source's own coordinates (IFC and OBJ differ).
native_ifc_mesh = scene.mesh(source="ifc", coordinates="original")
native_obj_mesh = scene.mesh(source="obj", coordinates="original")

# Register either source into the original point-cloud coordinates.
# OBJ: inverse(mat_pc2obj)
# IFC: inverse(mat_pc2obj) @ ifc_to_obj
mesh_in_point_cloud_coordinates = scene.mesh(
    source="ifc",
    coordinates="point_cloud",
)

# BIMNet point clouds contain x y z r g b label.
cloud = scene.point_cloud(include_labels=[0, 1, 2, 3])
# visualize(aligned=True) uses the original PC coordinates as the common frame.
scene.visualize(
    point_cloud_options={"voxel_size": 0.03},
    mesh_options={"source": "obj", "wall_filled": True},
)

# Render the registered BIM mesh from an original Matterport RGB-D frame.
matterport_scene = scene.matterport_scene(matterport_dataset)
frame = matterport_scene.frames[0]
render = scene.render_frame(frame, source="ifc", render_depth=True)
render.save("outputs/hxp_bim.png")
```

Missing optional downloads do not prevent scene discovery. Check
`scene.has_point_cloud`, `scene.has_rvt`, `scene.has_wall_filled_mesh`, or the
combined `scene.availability` mapping before loading those assets. A full
Matterport house can be resolved with
`scene.matterport_scene(Matterport3DDataset(...))`; floor suffixes such as
`_1` and `_2` remain BIMNet scene metadata.

## BIMSync IFC

`BIMSyncDataset` 支持 `root/Area_1/*.ifc` 和 IFC 直接位于 root 下的扁平目录。
dataset 负责发现和批量处理场景；单个 IFC 的 mesh、配准、导出、可视化和相机渲染
均由 `BIMSyncScene` 负责。

```python
from s3dis_sam3d import BIMSyncDataset, S3DISDataset

bimsync = BIMSyncDataset("path/to/bimsync/ifc", area="Area_1")
print([scene.key for scene in bimsync.scenes])
scene = bimsync.scene("office_11")

# IFC 世界坐标提取为 Open3D mesh，并统一为米
raw_mesh = scene.mesh(calibrated=False)

# 求 IFC -> S3DIS 坐标转换
s3dis = S3DISDataset("path/to/Stanford3dDataset_v1.2")
s3dis_room = s3dis.room("Area_1/office_11")
registration = scene.register(s3dis_room)
print(registration.ifc_to_s3dis)
scene.save_registration(registration, "outputs/ifc_to_s3dis/Area_1/office_11")

# 保存后，该 scene 读取 mesh 时会自动应用 IFC -> S3DIS 矩阵
mesh = scene.mesh()
scene.export("outputs/office_11_calibrated.ply")

# 将配准后的 IFC mesh 与 S3DIS 点云叠加，并保存截图
scene.visualize_registration(
    registration,
    s3dis_room,
    "outputs/ifc_to_s3dis/Area_1/office_11/registration.png",
)
```

批量校准并保存每个 scene 的矩阵：

```python
summary = bimsync.calibrate_scenes(
    s3dis,
    "outputs/ifc_to_s3dis/Area_1",
    scenes=None,           # 自动处理所有同名 IFC/S3DIS scene
    visualize=True,
)
```

在新的进程中，传入校准目录即可自动恢复所有矩阵。之后 `scene.mesh()`、`scene.export()`
和 `dataset.export_meshes()` 默认输出 S3DIS 坐标；需要 IFC 原始坐标时传入
`calibrated=False`：

```python
bimsync = BIMSyncDataset(
    "path/to/bimsync/ifc",
    area="Area_1",
    calibration_dir="dataset/ifc_to_s3dis/Area_1_upright_v2",
)
scene = bimsync.scene("office_11")
mesh = scene.mesh()
print(scene.is_calibrated, scene.calibration)
```

`visualize_registration` 默认不打开窗口，适合批处理保存截图；需要交互查看时传入
`show=True`。S3DIS 的房间数据与实例数据分别由 `S3DISRoom` 和 `S3DISInstance` 负责。

### 使用 2D-3D-S regular 相机渲染 IFC

校准后的 IFC 可以直接使用 regular 图像的内参和位姿渲染，输出尺寸与原 RGB 图像一致：

```python
from s3dis_sam3d import BIMSyncDataset, S23Dataset

s23dis = S23Dataset(area="Area_1", projection_type="regular")
bimsync = BIMSyncDataset(area="Area_1")
frame = s23dis.room("office_11").get_frame(frame_id=0)
result = bimsync.scene("office_11").render_frame(frame)
result.save("outputs/office_11_frame_0.png")

print(result.source_image_path, result.source_image.shape)
print(result.source_depth_path, result.source_depth.shape)
print(result.rendered_image_path, result.rendered_image.shape)
print(result.rendered_depth_path, result.rendered_depth.shape)

# Visualize one pair at a time. Depth heatmaps share one jointly computed scale.
result.visualize(mode="rgb")
result.visualize(
    mode="depth",
    show=False,
    save_path="outputs/office_11_depth_comparison.png",
)
```

同一个 room/frame 对应多个相机 UUID 时，默认使用按字典序排列后的第一个；可通过
`s23dis.room(room).list_uuids(frame_id)` 查看并显式传入其他 `uuid`。该方法要求 scene 已有
IFC→S3DIS 校准矩阵；`render_frame()` 在内存中返回 RGB-D，调用 `result.save(...)` 时才写入
RGB 和深度文件，传入 `show=True` 可同时打开 Open3D
窗口。`source_image` 和 `rendered_image` 是 `[0, 1]` RGB 数组，`source_depth` 和
`rendered_depth` 的单位均为米。
渲染深度与 2D-3D-S 原图严格使用相同编码：16 位 PNG、`depth_scale=512`、无效值
`65535`，读取逻辑与 `Frame.depth` 完全一致；不需要渲染深度时传入
`render_depth=False`。

也可以编辑并运行现成脚本：

```bash
python script/render_calibrated_ifc.py
```

配准只使用墙、地面、天花板、梁柱、门窗等结构几何。算法固定尺度为 1，并固定 Z 轴
竖直，只优化 yaw 和 XYZ 平移；36 个水平旋转起点经过双向最近邻、trim 和 Huber
鲁棒拟合，再用门窗、梁柱类别消除近 180° 的对称解。结果包含两个方向的矩阵、
fitness/RMSE、质量门限和候选审计；未通过门限的房间进入 summary 的 `errors`。

如果没有传统的 `Stanford3dDataset_v1.2` 点云，也可以直接使用 2D-3D-S 自带的
`semantic.obj`。这是 PriorBIMDA 标定 Area_1 使用的输入：

```bash
python script/register_ifc_to_semantic_obj.py \
  --semantic-obj /path/to/area_1/3d/semantic.obj \
  --bimsync-root /path/to/BIMSyn/BIM_model/ifc \
  --output-dir dataset/ifc_to_s3dis/Area_1_upright
```

批量转换 Area_1 IFC mesh：

```bash
python script/export_bimsync_ifc_meshes.py
```

批量计算并保存 IFC→S3DIS 校准矩阵：

```bash
python script/register_ifc_to_s3dis.py
```

该脚本的 `SCENES` 默认列出要处理的房间；设为 `None` 会自动匹配并处理 Area 下所有
同名的 IFC/S3DIS 区域。每个房间单独保存矩阵，同时持续更新 Area 级 summary，某个
房间失败不会中断其余房间。

## 批量生成房间图像标注

编辑 [script/batch_annotate_room_images.py](script/batch_annotate_room_images.py) 顶部的数据路径、房间和类别，然后运行：

```bash
python script/batch_annotate_room_images.py
```

脚本会把 S3DIS 的 3D instance 投影到对应的 2D-3D-S frame，每帧生成一个
`*_ann.json`。regular 使用针孔投影，pano 使用等距柱状投影；depth 仅用于计算遮挡率，
bbox 保持 3D 实例的 amodal 投影范围。

也可以直接在 Python 中调用：

```python
from s3dis_sam3d.annotations import RoomImageBatchAnnotator

annotator = RoomImageBatchAnnotator(s3dis_root, area_path, projection_type="regular")
summary = annotator.annotate_room_images(
    "Area_1/office_8",
    selected_classes=["chair", "table", "sofa"],
    output_dir="outputs/room_image_annotations/office_8",
    save_empty=False,
)
```

解析、查询和可视化生成的 annotation：

```python
from s3dis_sam3d.annotations import RoomImageAnnotationsParser

parser = RoomImageAnnotationsParser(
    "outputs/room_image_annotations/office_8",
    image_root="path/to/image/root",  # image_path 已可直接访问时可省略
)
print(parser.list_records())
record = parser.get_record_by_uuid_frame(uuid, frame_id)
parser.visualize_bboxes_on_image(0, save_path="outputs/annotation_preview.png")
```

`room + frame_id` 在 2D-3D-S 中可能对应多个 UUID；解析器遇到这种查询会明确报歧义，
应改用 `uuid + frame_id`。

## 批量 SAM3D 预测

编辑 [script/batch_predict_sam3d.py](script/batch_predict_sam3d.py) 顶部的 annotation
目录、2D-3D-S Area、服务 URL、筛选阈值和推理选项，然后运行：

```bash
python script/batch_predict_sam3d.py
```

脚本默认跳过 clutter、低 bbox 完整度、高遮挡率和跨 pano 接缝的 annotation。开启
`USE_DEPTH` 时发送 depth 与 K；开启 `OPTIMIZE_POSE` 时同时发送 camera-to-world。
结果可断点复用，并持续写入 `batch_predict_summary.json`。

## GLBMesh

```python
from s3dis_sam3d import GLBMesh

asset = GLBMesh(
    "prediction.glb",
    pose_path="prediction.json",
    pose_space="auto",
)
mesh = asset.get()  # open3d.geometry.TriangleMesh
cloud = asset.sample_points(20_000, voxel_size=0.02)
asset.export("prediction_posed.glb")
asset.visualize()
```

点云与网格可以在同一个 Open3D 窗口中显示：

```python
from s3dis_sam3d.pointcloud import visualize_point_clouds

visualize_point_clouds([cloud, asset])
```

如果需要把模型放入 2D-3D-S 世界坐标，可传入相机位姿：

```python
mesh = GLBMesh.load_posed(
    "prediction.glb",
    "prediction.json",
    camera_to_world=frame.camera_to_world,
)
```

## SAM3D

```python
from s3dis_sam3d.sam3d import SAM3DClient

result = SAM3DClient("http://127.0.0.1:8000/infer").infer(
    "image.png",
    request_id="chair_001",
    output_dir="outputs",
    mask_path="mask.png",
)

print(result.glb_path)
```
