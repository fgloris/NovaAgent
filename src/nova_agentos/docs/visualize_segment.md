# visualize_segment:叠加 3D 线段

## 用途
在图像上叠加一条 3D 线段(两端点,半透明圆柱),可标注 3D 距离,
用于核对两点间的位置关系/间距。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `points`(必填):线段两端点,**base 系** [[x, y, z], [x, y, z]],单位 m。
- `radius`:杆半径(m),默认取节点参数。
- `show_distance`:是否在中点标注 3D 距离,默认 true。
- `color` / `label`:可选。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg",
 "points": [[0.35, 0.02, 0.82], [0.42, 0.02, 0.82]]}
```

## 常见错误
- points 传像素坐标:这里要 base 系米制 3D 坐标。
- 端点顺序无所谓,但两点都要在 base 系下。
