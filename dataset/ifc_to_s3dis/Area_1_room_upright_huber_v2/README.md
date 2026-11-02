# Area1 BIMSync: room-level Huber refinement

保存 `rgb-bim-refine` 的 D33/E40（2026-10-23）精修结果，覆盖全部 44 个可用 IFC 房间、
186 个扫描位置、10,327 张方向照片。88 份双向矩阵从原结果按字节复制，没有重新配准。
原工具包的初始配准仍保存在相邻的 `Area_1_upright_v2`。

## 文件和坐标

```text
Area_1_room_upright_huber_v2/
├── README.md
├── calibration_summary.json
└── <room>/
    ├── <room>_ifc_to_s3dis_transform.npy
    └── <room>_s3dis_to_ifc_transform.npy
```

矩阵为 float64、4×4，采用列向量，平移单位为米。正向将原始 IFC 的米制几何变换到原始扫描坐标，
逆向为扫描坐标到 IFC。相机继续使用数据集原始扫描位姿。

`calibration_summary.json` 集中保存参数、前后几何统计、房间数量、质量标记、源 IFC 和结果的
SHA-256，以及每个房间相对于初始配准的修正 W。矩阵关系：

```text
T_ifc_to_s3dis_refined = inverse(W_pointcloud_to_original_bim) @ T_ifc_to_s3dis_original
```

## 读取

```python
from s3dis_sam3d import BIMSyncDataset
from s3dis_sam3d.config import bimsync_calibration_dir

dataset = BIMSyncDataset(
    "path/to/BIMSyn/BIM_model/ifc",
    calibration_dir=bimsync_calibration_dir("Area_1", refined=True),
)
scene = dataset.scene("office_15")
T = scene.calibration
mesh = scene.mesh()  # 已应用精修矩阵，位于扫描坐标；此调用需要 IfcOpenShell
```

省略 `refined=True` 的旧调用仍读取原工具包结果。只读取 NPY 矩阵不需要 IFC 或 IfcOpenShell。
`scene.mesh()`、`scene.export()` 等现有接口会应用已加载矩阵，导出的网格直接配合原始相机使用。

## 方法与前后对比

以原工具包矩阵为初值，合并每个房间全部扫描位置中属于该房间的墙、地、顶结构点，
与同类 IFC 完整三角面做鲁棒点到面配准。保持竖直，只优化 yaw 和三维平移，比例固定为 1；
Huber 尺度 3 cm，距离门限 30/15/8 cm，扫描位置等权。

先计算每个扫描位置的几何指标，再对 186 个位置等权平均：

| 指标 | 原工具包矩阵 | 精修后 |
|---|---:|---:|
| 中位距离的平均 | 4.09 cm | 2.67 cm |
| P90 距离的平均 | 7.26 cm | 5.94 cm |
| 2 cm 内比例 | 33.9% | 68.4% |
| 5 cm 内比例 | 92.2% | 92.3% |

这些是参与拟合数据的结构点到同类 IFC 表面距离，不是独立相机定位精度。
离线校准使用全部房间几何，包含原验证房间；`hallway_1` 只有一个扫描位置。
44 个房间分别拟合，共享墙面和门洞尚未做联合约束。原始 IFC 和相机位姿不包含在本目录中。
