# 结构标定

先选择 `alignment_basis`：`front_back_images` 为正背面图片参考，`model_uv` 为当前模型真实接缝（兼容默认模式）。

## 模型 UV 模式

1. `prepare` 后在新的 `normalized` 图片上标源坐标。抠图、旋转或缩放变化都会使旧源坐标失效。
2. 按钮目标使用指定 SKU 的对应按钮边界；天线目标必须使用该 SKU 自己的背板。读取当前 Blender 场景，备份未保存修改，再提取同一空间位置的完整 UV 接缝曲线。
3. 用 `texture_alignment.seam_targets.transfer_row` 将背板上下天线的两条边界分别换算到侧面。左右材质对应背板哪一侧由真实 UV 决定，不能按名字猜测。
4. 在 Blender 中调用 `texture_alignment.blender_scene_guard.capture(obj)`，保存返回的 JSON 为场景记录。`scene_record` 指向此记录；把记录和每张背板的绝对路径、SHA256 写入 `reference_files`。
5. 将源标记放入 `points[SKU][side]`；同顺序的目标放入 `targets_by_sku[SKU][side]`。上下天线各两条边界都要包含。`antenna_slopes` 可给天线逐列斜率，按钮通常为 0。统一入口拒绝缺少逐 SKU 天线基准或场景记录的配置。

字段示意（坐标仅说明结构，不能直接执行）：

```text
reference: 按钮基准 SKU
height: 本批 normalized 的统一高度
points: 各 SKU 的 left/right 源图标记列表
targets_by_sku: 各 SKU 的 left/right 目标标记列表
antenna_slopes: 各 SKU 的 left/right 源标记逐列斜率（可选）
scene_record: 当前 Blender 场景指纹 JSON 的绝对路径
reference_files: [{path: 背板或场景记录的绝对路径, sha256: 文件哈希}, ...]
```

文件哈希只保护离线标定数据。写回 Blender 前还要用 `require_match(expected, capture(obj))` 核对实时场景，或用 `bind_images` 安装贴图。不要为了让旧配置通过而重写场景记录；UV 或几何变化需重新标定。

局部返修从最初标准化输入重新映射。在已验收区前后加入固定映射锚点，并与已验收图逐像素比对；SKU_10 按钮颜色不得回退。每 SKU 保留六面图片。检查上下天线的两条边界、宽度与左右底部近景，上端接上不代表底部正确。

旧算法细节见 [原对齐说明](alignment.md)。其中旧脚本路径仅用于理解历史接口；新项目统一通过 `phone-texture prepare/apply` 调用。

## iPhone 17 系列：正背面图片参考

已有六面照片时可用图片参考生成左右贴图，无需先提供模型。先摆正正背面并分别量机身上下端，排除透明画布空白、贴纸纸尾与镜头凸起；每侧分别登记同一台机器正面、背面天线四条边界。不能仅以画布等高或把两图天线位置取平均代替标定。

`texture_alignment.image_reference_targets.build_targets(guide, height, width)` 接收：

```text
front/back: {path, sha256, body_y: [机身顶,机身底], antenna_rows: [上天线上界,上天线下界,下天线上界,下天线下界]}
side_body_y: [侧面机身顶,侧面机身底]
front_at: left 或 right，明确图片哪一列接正面，不按材质名猜测
button_rows: 独立量取/指定基准的全部按钮上下边界
button_basis: 按钮基准图及量取依据
```

前后图通过各自机身坐标换算到侧面，侧面两端分别对齐，再逐列插值。按钮目标保持单独约束；目标交叉时拒绝，重核姿态及标记。把返回的数组写入 `targets_by_sku[SKU][side]`，对应 guide 写入 `image_reference_guides[SKU][side]`，并设置 `alignment_basis: front_back_images`。`points` 按同一顺序标本次侧面源图位置；front/back 绝对路径和哈希登记在 `reference_files`。随后正常 `phone-texture apply`。发布前程序重新从前后图标记计算目标并核对，不能只手写一组任意 targets 绕过。

结果报告为图片空间对齐。导入已有模型后仍核对当前 UV；如果模型背板 UV 被单独缩放/平移，切到 `model_uv` 模式重新换算，不能宣称图片等高就已验证 3D 接缝。
