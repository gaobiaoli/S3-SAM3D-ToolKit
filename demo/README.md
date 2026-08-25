# Demos

示例直接使用仓库内 `dataset/` 的真实最小数据，不依赖 CLI。

## S3DIS 可视化

```bash
python -m pip install -e .
python demo/s3dis_visualization.py
```

脚本默认打开 `Area_4/hallway_5` 的 Open3D 窗口。可在文件顶部修改房间、着色方式、隐藏类别、隐藏实例、包围盒实例与最大点数，例如：

```python
DEFAULT_AREA = "Area_1"
DEFAULT_ROOM = "office_8"
DEFAULT_HIDDEN_CLASSES = ["door", "clutter", "board", "beam", "ceiling", "column"]
DEFAULT_HIDDEN_INSTANCES = ["wall_3", "wall_4"]
```

`get_parameters=True` 会在关闭窗口后打印相机参数。将结果填入 `DEFAULT_VIEW_PARAMETERS`，即可复用固定视角。

## 2D-3D-S 单帧重建

```bash
python demo/s23dis.py
```

示例读取仓库内的 `area_1/hallway_2/frame_1`，打印内参与 camera-to-world，并在 Open3D 中显示重建点云。
