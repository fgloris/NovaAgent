# visualize_grid:网格粗定位

## 用途
在图像上叠加 grid_size×grid_size 网格并标注 行-列 编号,快速粗定位目标所在区域,
再换算成像素坐标,作为后续三角化/画圈核对的起点。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `grid_size`:网格划分大小,默认取节点参数(默认 8,即 8x8)。

## 返回
- `cell_px`:每个格子的像素尺寸,用于把行列换算成像素:
  `u ≈ (列 - 0.5) * cell_px[0]`,`v ≈ (行 - 0.5) * cell_px[1]`。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg", "grid_size": 8}
```

## 常见错误
- 直接把行列当像素:要乘 `cell_px` 再换算。
- 网格太粗定位不准:减小 `grid_size` 或换更清晰的视图。
- 拿到粗略像素后忘记用 `visualize_pixels` 画圈核对。
