# 定位物体 3D 坐标的流程经验

## 目标
利用多路相机的 2D 像素观测,三角化出物体在机器人 **base 系**下的 3D 坐标 (x, y, z)。

## 关键约定
- 图像统一用 `file://<kind>/<file>` 引用(见上下文里的 current/processed/history 图像描述)。
  像素坐标就用**该图的实际像素**;`reproject_pixels` 需要同时给出该图的 `size` 作为 `image_size`。
- 至少需要 **2 个**同时能看到目标的相机。
- 相机名以图像描述里的 `camera` 为准(如 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`)。

## 流程
1. **选相机**:挑 2 个以上能同时看到目标物体的相机,记录各自 current 图像的 `url` 与 `size`。
2. **粗定位(可选)**:对每个视图调用 `visualize_grid`(传该图的 `url`)叠加网格,读出目标中心所在格子,再换算成像素:
   `u ≈ (列 - 0.5) * cell_px[0]`,`v ≈ (行 - 0.5) * cell_px[1]`(`cell_px` 由工具返回,单位是该图像素)。
3. **三角化**:调用 `reproject_pixels`,传入 `points = {相机名: [u, v], ...}` 和 `image_size = {相机名: [w, h], ...}`
   (取各图描述里的 `size`)。返回 `position`(base 系 3D)、`reprojected`(各视图重投影像素)、`errors_px`、`mean_error_px`。
4. **核对**:调用 `visualize_pixels`,把上一步的 `reprojected` 像素画到对应相机图上(传该图的 `url`),观察是否落在目标中心;
   也可同时画出你自己的观测像素,对比二者是否重合。
5. **迭代**:若 `mean_error_px` 偏大或圈没对准,修正像素后重做第 3 步;
   只有 **`converged=true`** 时才认为定位可靠。
6. **收尾**:给出 `position` 作为物体的 base 系 3D 坐标。

## 常见失败与对策
- 某个视图看不到目标:换一个相机,只用能看到的相机(仍需 ≥2 个)。
- 误差一直降不下来:通常是某个视图的像素偏了;用 `visualize_pixels` 逐个视图核对并修正。也有可能是因为场景不是静态，物体一直在移动，这种情况下应当放弃静态定位，随机应变。
- 提示"在相机后方/超出画面":该点像素不合理,重新估计。
