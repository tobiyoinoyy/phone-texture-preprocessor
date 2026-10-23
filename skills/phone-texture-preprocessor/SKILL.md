---
name: phone-texture-preprocessor
description: 手机六面贴图统一预处理，包含原图抠图精裁、四面边框摆正、按钮与天线结构对齐；适用于批量 SKU 贴图及 Blender 接缝修复，保留其他面和独立过程记录。
---

# 手机贴图预处理

最新自动处理口径（2026-10-23）：用户自行处理 UV，`auto` 采用精裁后的画布高度比例对齐。本机背板为基准；排除极淡半透明拖尾后保留 1 像素抗锯齿及真实纸尾，调光前冻结结构标记，重映射固定圆角和按钮，成品再次检查尺寸与上下天线。全部由程序执行，不依赖 Agent 判断、补点或裁切。不能以历史固定 UV 规则覆盖此用户新要求；显式模型任务仍适用后文独立模型检查。

从本技能实际路径向上两级找到仓库根目录（先解析符号链接），使用仓库环境的 `python -m phone_texture`。完整六面任务先用 [统一运行流程](../../docs/workflow.md) 的 `phone-texture start/run/status`，下列命令是同程序的几何子步骤。安装、目录约定和命令见 [README](../../README.md)。按输入实际状态选择步骤，不能对已经竖向摆好的左右贴图再次执行原图标准化。

- 原始六面照片：`standardize`，只用 BiRefNet_dynamic 分割，四个边框面默认摆正。
- 已裁切透明贴图：`deskew --faces ...`，只刚性旋转所选面；其余图片字节不变。已验收侧面不要再摆正。
- 结构对齐：`prepare` → 查看标注页 → 按正背面图片参考或真实 UV 标定 → `apply`。标准化已经摆正时跳过 deskew。

生成候选后必须执行[几何复核门禁](../../docs/geometry-review.md)：逐面看原图叠加、四角、完整下缘及接触区的原尺寸/放大深浅背景图。边界歧义、启发式去支架、人工标定或外推边缘需要独立机身/支架像素检查，完成 `approve-geometry` 才能发布和冻结光照基线。矩形只包围完整轮廓，去支架使用局部蒙版；保留纸尾，旋转前后都复核，不能用高置信度或 reviewed 跳过。

无 Agent 运行以 `phone-texture auto` 为入口：本采集规范的六面分析与处理均由程序执行，不能依赖 Agent 看图补点、画蒙版或填签收。失败明确返回，不降低标准。用户最终验收独立于程序运行。顶面必须排除整个摄像头台阶及镜头，仅保留顶边框本体；原图、旋转后、精裁后均检查。算法、适用范围和旧标定拒绝规则见 [顶面本体标定](../../docs/top-body-calibration.md)。

几何处理使用原图像素，不用生图模型，不调色。光照处理走独立的铝金属或不锈钢 Skill。

默认 `alignment_basis: back_image`，以待处理 SKU 自己的背面约束左右侧上下天线四边界，天线横向平直；不要求先提供模型。按钮可以按指定 SKU 对齐。iPhone 17 系列可选 `alignment_basis: front_back_images`，按每 SKU 正面与背面的机身上下端、天线两条上下边界分别定位侧面的前后端，使用 `texture_alignment.image_reference_targets.build_targets` 生成逐列目标；按钮独立锚定。`model_uv` 模式按当前模型 UV 换算接缝。所有模式均禁止回退到其他 SKU 天线坐标。标定步骤和配置见 [calibration.md](../../docs/calibration.md)。导入模型前通过 Blender MCP 检查当前场景及未保存改动，备份后用 `texture_alignment.blender_scene_guard.capture` 保存指纹，安装贴图用 `bind_images`。文件名相同的 GLB 不能代替实际工程。

上部正确、底部断开曾由背板 UV 缩放平移造成。先比较实际加载图片哈希和 UV，再定位问题；只修底部时保留上方与按钮已验收像素，并逐像素检查，保留 SKU_10 已修按钮颜色。检查左右上下天线的两条边界及宽度，保存底部近景，不能只看总览判定通过。新裁切或 UV 变化后重新标定，Git 提交不等于场景验收。

输入始终只读，输出不存在就创建，已有输出另选新目录。正式目录只放每 SKU 六面图片，脚本、报告、备份和检查图放过程目录。总览只交付一张，left 在一起、right 在一起；顶底对比时 top 在一起、bottom 在一起。

模型贴图最终交付必须接入统一六面 Skill 的[结构发布门禁](../phone-six-view-process/references/structure-delivery.md)。`standardize`、光照处理或 `apply` 退出成功都不能代替该门禁。图片空间对齐与真实模型 UV 验收分开记录。
