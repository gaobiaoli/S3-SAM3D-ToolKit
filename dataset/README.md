# Minimal real-data samples

这个目录从本机已有的 Stanford 数据集中原样复制了一个 S3DIS 房间和一个 2D-3D-S regular 帧，用于展示真实目录结构并运行仓库示例。文件未经过降采样、裁剪或重编码。

## S3DIS

```text
s3dis/
└── Area_4/
    └── hallway_5/
        ├── hallway_5.txt
        └── Annotations/
            ├── ceiling_1.txt
            ├── clutter_1.txt
            ├── ...
            └── wall_2.txt
```

每行格式为：

```text
x y z red green blue
```

实际路径为 `s3dis/Area_4/hallway_5`。`hallway_5.txt` 是完整房间点，`Annotations/*.txt` 是 8 个真实实例，共有 85,855 个标注点。

## 2D-3D-S

```text
2d3ds/
└── area_1/
    └── data/
        ├── rgb/
        │   └── camera_<uuid>_hallway_2_frame_1_domain_rgb.png
        ├── depth/
        │   └── camera_<uuid>_hallway_2_frame_1_domain_depth.png
        ├── pose/
        │   └── camera_<uuid>_hallway_2_frame_1_domain_pose.json
        └── global_xyz/       # 可选，本示例为空
```

实际样例是 `area_1/hallway_2/frame_1`，RGB 和 16-bit depth 图均为 1080×1080。深度值采用 2D-3D-S 的 `raw / 512.0 = metres` 约定，其中 `65535` 表示无效深度。原数据随附的许可文件保存在 `2d3ds/LICENSE.pdf`。

## 再分发提醒

这些是数据集的真实原始文件，不是本项目自行生成的测试数据。将仓库公开上传到 GitHub 前，请确认 Stanford S3DIS 与 2D-3D-S 的当前许可、下载条款和再分发条件；本工具包的代码许可证不会自动覆盖数据文件。

从项目根目录运行：

```bash
python examples/run_minimal_dataset.py
```

结果会写入被 Git 忽略的 `outputs/minimal_dataset/`。
