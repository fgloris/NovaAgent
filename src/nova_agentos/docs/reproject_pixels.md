# reproject_pixels:多视图像素三角化

## 用途
把多个相机上目标的 2D 像素坐标,用相机投影矩阵 DLT 三角化出机器人 **base 系**下的 3D 坐标
(x, y, z, 单位 m),并返回各视图的重投影像素与误差,用于核对定位是否收敛。

## 参数
- `points`(必填):`{相机名: [u, v]}`,至少 **2 个**相机;像素用该图**实际分辨率**。
- `image_size`(必填):`{相机名: [w, h]}`,各相机像素坐标所属分辨率(取自图像描述里的 `size`)。

## 返回
- `position`:base 系 3D 坐标。
- `reprojected`:各视图重投影像素。
- `errors_px` / `mean_error_px`:各视图/平均重投影误差(像素)。
- `converged`:误差是否收敛,**只有 true 才认为定位可靠**。

## 示例
```json
{"points": {"robot0_agentview_left": [210, 160], "robot0_agentview_right": [150, 175]},
 "image_size": {"robot0_agentview_left": [256, 256], "robot0_agentview_right": [256, 256]}}
```

## 常见错误
- 只给 1 个相机:无法三角化,至少 2 个。
- 像素与 `image_size` 不匹配:用该图描述里的 `size`。
- `mean_error_px` 偏大:某视图像素偏了,用 `visualize_pixels` 逐个核对修正。
- 相机名写错:以图像描述里的 `camera` 为准(如 `robot0_agentview_left`)。
