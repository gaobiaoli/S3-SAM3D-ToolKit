# SLABIM reference poses: v1 and v2

两版均覆盖 12 个区域：164,843 帧 LiDAR、3,929 帧相机。
原始 ROS 包、点云、图像与 BIM 独立存放，本目录只保存派生位姿。
这些是计算得到的参考位姿，不是独立测量的精确相机真值。

```text
slabim/
├── README.md
├── v1/                       # 原有恢复结果，数值保持不变
│   ├── summary.json
│   └── <region>/
│       ├── lidar_pose_to_bim.txt
│       ├── camera_pose_to_bim.txt
│       └── camera_frames.txt
└── v2/                       # 在 v1 上增加每区域一个整体 map→BIM 矫正
    ├── summary.json
    └── <region>/             # 同样三个文件
```

## 读取与坐标

```python
from s3dis_sam3d.config import slabim_registration_dir
from s3dis_sam3d.slabim import load_camera_poses, load_pose_table

names, times, camera_to_bim = load_camera_poses("5F_Region2", version="v2")
# camera_to_bim.shape == (N, 4, 4)
times, lidar_to_bim = load_pose_table(
    slabim_registration_dir("v2") / "5F_Region2/lidar_pose_to_bim.txt"
)
```

读取函数省略版本时仍使用 v1，避免已有调用悄悄改变基准。
`root=` 可显式指定某一版本的结果目录。
两种位姿表每行都是 `timestamp_seconds tx ty tz qx qy qz qw`，平移为米，四元数为 xyzw，
变换方向为传感器局部坐标到 BIM。相机坐标为 x 向右、y 向下、z 向前。
LiDAR 第 i 行对应 `sensor_data/<region>/points/data/<i>.pcd`（文件名可能补零）；
相机行与 `camera_frames.txt` 同行文件名对应，图像路径为
`sensor_data/<region>/images/data/<filename>`。不能用相机表行号推断图像编号。

两版时间戳、图像名单和过滤规则相同。4F_Region3 沿用历史 v1 排除名单：
40、42、44、46、48、50、54；其余区域没有额外排除图像。
每版 `summary.json` 集中记录参数、数量、源文件和结果的 SHA-256。

## v1：从 ROS 恢复

从 `/livox/lidar` 读取局部点云，以 `timebase / 1e9` 匹配官方 LiDAR 时间戳（最多差 0.01 秒）。
局部点云与同帧官方 SLAM-global PCD 作 0.2 米下采样后执行 point-to-point ICP。
首帧的距离阈值/最大迭代数为 1.0/60、0.5/60、0.2/80、0.1/80；后续帧以前帧初始化，使用 0.35/25。
轨迹用窗口 5、3 次多项式的 Savitzky–Golay 平滑；短序列缩短窗口，四元数先统一符号，平滑后归一化。
随后左乘官方固定 map→BIM。原始 `points/pose_frame_to_bim.txt` 重复记录的是固定矩阵，不是运动轨迹。
在图像时刻对平移线性插值、旋转 SLERP，右乘官方 camera→LiDAR 外参；拒绝时间范围外的外推。

## v2：整体 map→BIM 矫正（Claude G1）

每个区域重新从官方 `map/data/uncolorized.ply` 出发，在 SLAM 坐标下作 0.05 米体素下采样，
再乘官方 `map/pose_map_to_bim.txt`。匹配该楼层完整 BIM 的墙、楼板和柱，排除门扇。
Rhino PLY 的所有三角形/四边形都显式读取并三角化，避免丢面。
计算中统一移除零面积退化三角形，因为它们没有有效法向量；各类移除数量写入汇总，原始 BIM 不改写。

每个区域求一个 6 自由度鲁棒点到面刚体变换，Huber 尺度 0.03 米，距离门限依次为
0.30、0.15、0.08 米，每级 6 次迭代，初值为单位阵。点数超过 3,000,000 时以种子 0 固定抽样。
该方法和参数用于全部 12 个区域，始终从 v1 生成，不重复叠加已有矫正：

```text
T_lidar_to_bim_v2(t)  = C_region @ T_lidar_to_bim_v1(t)
T_camera_to_bim_v2(t) = C_region @ T_camera_to_bim_v1(t)
T_map_to_bim_v2       = C_region @ T_map_to_bim_official
```

全部区域统一应用拟合矩阵，包括较小的修正；原审计的 2 cm/0.2° 阈值用于判断是否存在明显系统偏差，
不用于本版的区域筛选。

不重新拟合逐帧位姿或相机外参，不按方法成绩调参数或筛选照片。
`v2/summary.json` 中每个区域的 `correction_to_bim` 为 C，`map_to_bim` 为矫正后的地图变换。
同时保存相机中心移动量、点到面残差、地面坡度及 10 米地块重采样 200 次的区间。
重采样在最终对应上作一次线性化更新；不约束完整 6 自由度的重采样会跳过并记录有效次数。

统计的含义：残差取距 BIM 30 cm 内的点；前后点集可能不同。地面坡度使用修正后距楼板 8 cm 内
同一批点和同一楼板面，属于同数据的一致性检查。相机中心移动量不是独立测得的绝对位姿误差，
重采样区间也不包含全部 SLAM/BIM 系统误差。

v2 包含 5F 的几何校准（用户授权全部区域）。原项目冻结的 v1 基准与预测不改写；
按 v2 做实验时应明确披露版本，不能直接沿用 v1 成绩或称为同一次盲测。

## 运行

读取附带的位姿不需要 ROS、rosbags 或 plyfile。恢复 ROS 数据需要 Python 3.10 及以上。

```bash
python -m pip install -e ".[slabim]"
python -c 'from s3dis_sam3d import configure; configure(slabim_root="/path/to/SLABIM")'
python script/register_slabim.py    # 恢复/核验 v1
python script/correct_slabim.py     # 从 v1 生成/核验 v2
```

原始根目录包含 `sensor_data/<region>`、`calibration_files` 和 `BIM/<floor>/mesh`。
脚本中的 `REGIONS` 可指定区域，输出目录按版本分开。已完成结果先核验再复用；
改变方法参数请另设输出目录，保留固定基准。
v1 恢复的 `.cache/` 在成功后删除；v2 不保存地图缓存、中间轨迹或日志。
不同 Open3D 版本、浮点平台的重算可能有微小差异，比较实验以附带的固定数值表为准。
