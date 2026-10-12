# 结构标定

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
