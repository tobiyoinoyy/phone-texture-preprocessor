---
name: phone-texture-preprocessor
description: 手机六面贴图统一预处理，包含原图抠图精裁、四面边框摆正、按钮与天线结构对齐；适用于批量 SKU 贴图及 Blender 接缝修复，保留其他面和独立过程记录。
---

# 手机贴图预处理

从本技能实际路径向上两级找到仓库根目录（先解析符号链接），使用仓库环境的 `python -m phone_texture`。安装、目录约定和命令见 [README](../../README.md)。按输入实际状态选择步骤，不能对已经竖向摆好的左右贴图再次执行原图标准化。

- 原始六面照片：`standardize`，只用 BiRefNet_dynamic 分割，四个边框面默认摆正。
- 已裁切透明贴图：`deskew --faces ...`，只刚性旋转所选面；其余图片字节不变。已验收侧面不要再摆正。
- 结构对齐：`prepare` → 查看标注页 → 按正背面图片参考或真实 UV 标定 → `apply`。标准化已经摆正时跳过 deskew。

生成候选后必须执行[几何复核门禁](../../docs/geometry-review.md)：逐面看原图叠加、四角、完整下缘及接触区的原尺寸/放大深浅背景图。边界歧义、启发式去支架、人工标定或外推边缘需要独立机身/支架像素检查，完成 `approve-geometry` 才能发布和冻结光照基线。矩形只包围完整轮廓，去支架使用局部蒙版；保留纸尾，旋转前后都复核，不能用高置信度或 reviewed 跳过。

几何处理使用原图像素，不用生图模型，不调色。光照处理走独立的铝金属或不锈钢 Skill。

按钮可以按指定 SKU 对齐。iPhone 17 系列可选 `alignment_basis: front_back_images`，按每 SKU 正面与背面的机身上下端、天线两条上下边界分别定位侧面的前后端，使用 `texture_alignment.image_reference_targets.build_targets` 生成逐列目标；按钮独立锚定。`model_uv` 模式按当前模型 UV 换算接缝。两种模式均禁止回退到基准 SKU 天线坐标。标定步骤和配置见 [calibration.md](../../docs/calibration.md)。导入模型前通过 Blender MCP 检查当前场景及未保存改动，备份后用 `texture_alignment.blender_scene_guard.capture` 保存指纹，安装贴图用 `bind_images`。文件名相同的 GLB 不能代替实际工程。

上部正确、底部断开曾由背板 UV 缩放平移造成。先比较实际加载图片哈希和 UV，再定位问题；只修底部时保留上方与按钮已验收像素，并逐像素检查，保留 SKU_10 已修按钮颜色。检查左右上下天线的两条边界及宽度，保存底部近景，不能只看总览判定通过。新裁切或 UV 变化后重新标定，Git 提交不等于场景验收。

输入始终只读，输出不存在就创建，已有输出另选新目录。正式目录只放每 SKU 六面图片，脚本、报告、备份和检查图放过程目录。总览只交付一张，left 在一起、right 在一起；顶底对比时 top 在一起、bottom 在一起。
