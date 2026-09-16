# visualize_pixels:像素画圈核对

## 用途
在图像上按像素坐标画空心圆圈(可带文本标签),用于核对目标物体在某张图上的像素位置;
常与 `reproject_pixels` 返回的 `reprojected` 像素对照,判断定位是否对准。

## 参数
- `image`(必填):源图 `file://<kind>/<file>` 引用。
- `points`(必填):`[[u, v], ...]` 或 `{label: [u, v]}`;像素用该图**实际分辨率**。
- `radius_px`:圆圈半径(像素),默认取节点参数。
- `color`:颜色名(red/green/blue/...)或 [r, g, b]。
- `label`:整体文本标签(点级标签用 `points` 的对象键)。
- `image_size`:可选,覆盖像素所属分辨率 [w, h];不传则自动推断。

## 示例
```json
{"image": "file://current/robot0_agentview_left.jpg", "points": [[210, 160]]}
```

## 常见错误
- 像素超出图像范围:先用 `visualize_grid` 或图像描述确认分辨率。
- 圈没对准:重估像素后重画,或与 `reprojected` 对比。
- 传了 base 系 3D 坐标:本工具要的是 2D 像素;3D 点请用 `visualize_point`。
