# visualize_point:叠加 3D 点

## 用途
在图像上叠加一个 3D 点(半透明小球,可带标签),用于核对物体/夹爪在 base 系下的**位置**。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `point`(必填):3D 点,**base 系** [x, y, z],单位 m。
- `radius`:小球半径(m),默认取节点参数。
- `color`:颜色名(red/green/blue/...)或 [r, g, b]。
- `label`:可选文本标签。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg",
 "point": [0.35, 0.02, 0.82], "label": "grasp"}
```

## 常见错误
- point 传成像素坐标:这里要 base 系米制 3D 坐标;像素核对请用 `visualize_pixels`。
- 坐标来源不明:先用 `reproject_pixels` 三角化,或从机器人状态读 EEF 位置。
