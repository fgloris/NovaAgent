# visualize_frame:叠加 3D 坐标系

## 用途
在图像上叠加一个 3D 坐标系(origin + orientation),沿局部 x/y/z 画红/绿/蓝半透明箭头,
用于核对夹爪、物体等的**朝向**。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `origin`(必填):坐标系原点,**base 系** [x, y, z],单位 m。
- `orientation`(必填):坐标系姿态,**base 系 xyzw** 四元数。
- `axis_length`:坐标轴长度(m),默认取节点参数。
- `radius`:箭头/杆半径(m),默认取节点参数。
- `labels`:是否在轴端标注 x/y/z,默认 true。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg",
 "origin": [0.35, 0.02, 0.82], "orientation": [0.0, 0.0, 0.0, 1.0]}
```

## 常见错误
- 四元数顺序写成 wxyz:必须是 **xyzw**。
- origin 传成像素坐标:这里要 base 系米制 3D 坐标。
- 坐标来源不明:EEF 位姿可从机器人状态读;物体坐标先用 `reproject_pixels` 得到。
