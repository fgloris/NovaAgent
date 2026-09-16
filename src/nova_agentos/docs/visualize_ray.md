# visualize_ray:叠加 3D 射线/方向

## 用途
在图像上叠加一条 3D 射线/方向(origin + orientation),画成半透明箭头,
用于核对朝向、视线或抓取接近方向。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `origin`(必填):射线起点,**base 系** [x, y, z],单位 m。
- `orientation`(必填):方向,**base 系 xyzw** 四元数,**局部 +z 指向该方向**。
- `length`:箭头长度(m),默认 0.2。
- `radius` / `color` / `label`:可选。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg",
 "origin": [0.35, 0.02, 0.82], "orientation": [0.0, 0.707, 0.0, 0.707], "length": 0.2}
```

## 常见错误
- 以为 `orientation` 是方向向量:它是 xyzw 四元数,局部 +z 才是射线方向。
- origin 传像素坐标:这里要 base 系米制 3D 坐标。
- 方向反了:把四元数取共轭或旋转 180° 后重画核对。
