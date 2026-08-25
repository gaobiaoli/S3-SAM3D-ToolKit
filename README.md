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
│   ├── s3dis.py          # S3DIS
│   ├── s23dis.py         # 2D-3D-S
│   ├── annotations.py    # S3DIS instance 到图像 bbox 的批量标注
│   ├── bimsync.py        # BIMSync IFC mesh 与 IFC/S3DIS 坐标配准
│   ├── models.py         # PointCloud、BoundingBox3D、GLBMesh
│   ├── pointcloud.py     # 下采样、变换、Open3D 可视化
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
BIMSYNC_CALIBRATION_ROOT = OUTPUT_ROOT / "ifc_to_s3dis"
```

省略数据集路径时会读取上述默认值：

```python
from s3dis_sam3d import BIMSyncDataset, S3DISDataset, S23Dataset

s3dis = S3DISDataset()
s23dis = S23Dataset(area="Area_1")
bimsync = BIMSyncDataset(area="Area_1")
```

`BIMSyncDataset()` 使用默认 BIMSync 路径时，如果对应 Area 的默认校准目录已经存在，
还会自动加载其中保存的校准矩阵。仓库内最小示例使用同一配置文件中的
`MINIMAL_S3DIS_ROOT` 和 `MINIMAL_S23DIS_ROOT`，不会与完整数据路径混淆。

## S3DIS

```python
from s3dis_sam3d import S3DISDataset

dataset = S3DISDataset("dataset/s3dis")
cloud = dataset.load_room("Area_4/hallway_5")
print(cloud.xyz.shape)

dataset.visualize_room(
    "Area_4/hallway_5",
    color_mode="semantic",
    hidden_classes=["door", "clutter", "board", "beam", "ceiling", "column"],
    hidden_instances=["wall_3", "wall_4"],
)
```

## 2D-3D-S

```python
from s3dis_sam3d import S23Dataset

dataset = S23Dataset("dataset/2d3ds/area_1")
room = dataset.list_rooms()[0][0]
frame = dataset.room_frames(room)[0]
cloud = dataset.frame_cloud(frame, stride=4)
print(cloud.xyz.shape)

# UUID 可省略，默认按排序结果选择第一个
frame = dataset.get_frame(room, frame.frame_id)
print(dataset.list_uuids(room, frame.frame_id))

# 常用单帧数据
rgb = dataset.get_image(room, frame.frame_id, frame.uuid)
depth = dataset.get_depth(room, frame.frame_id, frame.uuid)
point_map = dataset.get_depth(room, frame.frame_id, frame.uuid, backproject=True)
K = dataset.get_k(room, frame.frame_id, frame.uuid)
camera_to_world = dataset.camera_to_world(room, frame.frame_id, frame.uuid)

# 房间级可视化与保存
dataset.visualize_room(room, stride=8)
dataset.save_room_ply(room, "room.ply", stride=8, progress=True)
```

`mask` 可以是数组、图片路径、`{frame.stem: mask}` 映射或接收 `FrameInfo` 的回调。regular 与 pano 会自动使用各自的反投影模型。

## BIMSync IFC

`BIMSyncDataset` 支持 `root/Area_1/*.ifc` 和 IFC 直接位于 root 下的扁平目录：

```python
from s3dis_sam3d import BIMSyncDataset, S3DISDataset

bimsync = BIMSyncDataset("path/to/bimsync/ifc", area="Area_1")
print(bimsync.list_regions())

# IFC 世界坐标提取为 Open3D mesh，并统一为米
raw_mesh = bimsync.load_mesh("office_11", apply_calibration=False)

# 求 IFC -> S3DIS 坐标转换
s3dis = S3DISDataset("path/to/Stanford3dDataset_v1.2")
registration = bimsync.estimate_ifc_to_s3dis("office_11", s3dis)
print(registration.ifc_to_s3dis)
bimsync.save_registration(registration, "outputs/ifc_to_s3dis/Area_1/office_11")

# 保存后，该 dataset 读取 office_11 时会自动应用 IFC -> S3DIS 矩阵
mesh = bimsync.load_mesh("office_11")
bimsync.export_mesh("office_11", "outputs/office_11_calibrated.ply")

# 将配准后的 IFC mesh 与 S3DIS 点云叠加，并保存截图
bimsync.visualize_registration(
    registration,
    s3dis.get_region_point_cloud("office_11"),
    "outputs/ifc_to_s3dis/Area_1/office_11/registration.png",
)
```

批量校准并保存每个 region 的矩阵：

```python
summary = bimsync.calibrate_regions(
    s3dis,
    "outputs/ifc_to_s3dis/Area_1",
    regions=None,          # 自动处理所有同名 IFC/S3DIS region
    visualize=True,
    with_scaling=False,    # True 时额外估计统一尺度
)
```

在新的进程中，传入校准目录即可自动恢复所有矩阵。之后 `load_mesh()`、`export_mesh()`
和 `export_meshes()` 默认输出 S3DIS 坐标；需要 IFC 原始坐标时传入
`apply_calibration=False`：

```python
bimsync = BIMSyncDataset(
    "path/to/bimsync/ifc",
    area="Area_1",
    calibration_dir="outputs/ifc_to_s3dis/Area_1",
)
mesh = bimsync.load_mesh("office_11")
print(bimsync.calibrated_regions())
```

`visualize_registration` 默认不打开窗口，适合批处理保存截图；需要交互查看时传入
`show=True`。`get_region_point_cloud("office_11")` 默认读取 `Area_1/office_11`，也可传入完整房间名。

### 使用 2D-3D-S regular 相机渲染 IFC

校准后的 IFC 可以直接使用 regular 图像的内参和位姿渲染，输出尺寸与原 RGB 图像一致：

```python
from s3dis_sam3d import BIMSyncDataset, S23Dataset

s23dis = S23Dataset(area="Area_1", image_type="regular")
bimsync = BIMSyncDataset(area="Area_1")
result = bimsync.render_regular_frame(
    "office_11",
    s23dis,
    frame_id=0,
    output_path="outputs/office_11_frame_0.png",
)

print(result.source_image_path, result.source_image.shape)
print(result.source_depth_path, result.source_depth.shape)
print(result.rendered_image_path)
print(result.rendered_depth_path, result.rendered_depth.shape)
```

同一个 room/frame 对应多个相机 UUID 时，默认使用按字典序排列后的第一个；可通过
`s23dis.list_uuids(room, frame_id)` 查看并显式传入其他 `uuid`。该方法要求 region 已有
IFC→S3DIS 校准矩阵；默认后台保存 RGB 和深度，传入 `show=True` 可同时打开 Open3D
窗口。`source_image` 是 `[0, 1]` RGB 数组，`source_depth` 和 `rendered_depth` 的单位均为米。
渲染深度与 2D-3D-S 原图严格使用相同编码：16 位 PNG、`depth_scale=512`、无效值
`65535`，因此可以直接交给 `S23Dataset.load_depth()`；不需要渲染深度时传入
`render_depth=False`。

也可以编辑并运行现成脚本：

```bash
python script/render_calibrated_ifc.py
```

配准默认使用 S3DIS 与 IFC 的墙、地面、天花板、梁柱、门窗等结构几何，通过多组
水平旋转初始化和分阶段 point-to-plane ICP 求出 `S3DIS -> IFC`，再取严格逆矩阵得到
`IFC -> S3DIS`。开启 `with_scaling=True` 后改用 similarity ICP，统一尺度会直接写入
4×4 校准矩阵。保存结果同时包含两个方向、scale 及 fitness/RMSE，坐标单位均为米。

批量转换 Area_1 IFC mesh：

```bash
python script/export_bimsync_ifc_meshes.py
```

批量计算并保存 IFC→S3DIS 校准矩阵：

```bash
python script/register_ifc_to_s3dis.py
```

该脚本的 `REGIONS` 默认列出要处理的房间；设为 `None` 会自动匹配并处理 Area 下所有
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

annotator = RoomImageBatchAnnotator(s3dis_root, area_path, image_type="regular")
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
    camera_to_world=dataset.camera_to_world_from_pose(dataset.load_pose(frame)),
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
