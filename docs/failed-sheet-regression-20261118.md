# 2026-11-18：Pro / Pro Max 失败样本复跑

读取同一选品表的两个工作表，各 120 条数据，完整检查至物理第 200 行。失败栏共 13 条：iPhone 17 Pro 7 条、Pro Max 6 条；下载并绑定 78 张原图的 SHA-256。报告中的行号是工作表实际行号，不是 SKU 序号。原始照片、质检码、资源 token 和本机运行记录不提交到仓库。

## 原失败分类

| 失败原因 | 数量 |
|---|---:|
| 边框边界不一致 | 5 |
| 顶面上下导轨不一致 | 3 |
| 端头不符合细长边框 | 2 |
| 顶面圆角拟合不支持 | 2 |
| 不同透明度下边界不稳定 | 1 |

旧边框报错来自固定端头采样入口；当前 `auto` 的源图 RGB / 多采样带边界流程已能通过这 8 条非顶面样本的几何步骤，但后续阶段仍需独立执行，不能把几何恢复算成完整生成。

## 本轮修改

1. 顶面短下圆角的指数外推不稳定时，直接测量两端可见直边。上下导轨、残差、留出采样、原图 RGB、相机排除与机身保留检查不放宽。Pro 第 88、94 行和 Pro Max 第 107 行恢复；详见 [顶面标定规则](top-body-calibration.md)。原规则成功的输入不进入新分支。
2. Pro Max 第 6 行的分段背面天线测量没有输出 `agreeing_columns`，导致侧面续接抛出 `KeyError` 并被当作完整性错误。现在按实际返回的上下条带分别重数支持列，输出较小值；不把附近种子点的支持数借给返回位置，仍要求至少三个独立采样窗。
3. Pro Max 第 75 行的四个边框照片放错方向：顶/底、左/右分别对调。`scan_folder` 的原图识别确认完整六面后生成命名副本并校验哈希；没有按人工坐标补点，也未修改线上表格或原图。其余 12 台的方向识别与原文件名一致。
4. 上述第 6 行继续运行后暴露 JSON 往返类型问题：局部量化提示的 `coordinates_yx` 是元组列表，写入 JSON 后成为列表的列表。正常调光证据现在在保存前规范为同一个 JSON 值；证据内容、路径和哈希的完整性检查照旧，篡改内容的反例仍被拒绝。

## 复现与验收边界

修复前基线为 `cec5a8d`。两轮均从原始 JPEG 重新执行 BiRefNet_dynamic、几何、材质调光、天线结构和接缝流程，无几何 checkpoint 或人工标注输入。参数与本机服务一致：

```sh
.venv/bin/phone-texture auto <已完成原图方向识别的型号目录> \
  --model iphone17pro --jobs 2 --inference-slots 1 --cpu-threads 1 \
  --work <全新过程目录> --output <全新候选目录> \
  --source-body-reference-review --side-candidate-review
# Pro Max 使用 --model iphone17promax
.venv/bin/python scripts/test_all.py
```

第 6 行修复 JSON 问题后额外重跑一次后续阶段，显式使用第二轮源哈希绑定的完整几何 checkpoint（`--resume-geometry-from ... --require-geometry-resume --skus row6`）；该次没有重复分割，报告分别保留两轮原图记录和续接记录。

仅 `processing_state(result)['all_stages_complete']` 计为完整生成；六面文件存在或复核候选已保存均不等于全部阶段完成。所有原图哈希核验不变。新顶面相机保留、机身见证像素变化、旋转/精裁后机身丢失均为零，并查看原尺寸及实际 4 倍深浅背景四角。未检查用户当前模型 UV。

Pro 第 102 行原导轨残余倾角为 0.174°，下直边兜底的训练/留出误差分别为 8.87 / 9.69 px；Pro Max 第 74 行原残余倾角为 0.189°，兜底后仍为 0.174°。两者继续拒绝，没有把 0.15° 限制放宽。需要更稳定的源图边界或另外开发并独立验收透视校正，不能靠改成功状态解决。

背面镜头环边界、同材质参考不足或恢复幅度超限仍按原门禁保留失败和部分候选；本轮没有把未执行的调光改为已完成。

## 最终结果

| 型号 | 工作表行 | 四阶段完整 | 六面候选 | 未完成原因 |
|---|---:|---|---|---|
| iphone17pro | 15 | 否 | 有 | lighting/back: Rear shadow: outer ring edge is not independently supported |
| iphone17pro | 35 | 是 | 有 | 无阻断，仅效果提示 |
| iphone17pro | 84 | 是 | 有 | 无阻断，仅效果提示 |
| iphone17pro | 88 | 是 | 有 | 无阻断，仅效果提示 |
| iphone17pro | 94 | 是 | 有 | 无阻断，仅效果提示 |
| iphone17pro | 102 | 否 | 无 | TOP_BODY_RAILS_DISAGREE |
| iphone17pro | 120 | 否 | 有 | lighting/back: Rear shadow: narrow joint lightness recovery exceeds limit |
| iphone17promax | 6 | 否 | 有 | structure_proposal/left: Unstable central antenna contour；side_review_observation/left: Current raw paired observations unavailable |
| iphone17promax | 16 | 否 | 有 | lighting/back: Rear shadow: surrounding metal cannot support a continuous lighting reference |
| iphone17promax | 74 | 否 | 无 | TOP_BODY_RAILS_DISAGREE |
| iphone17promax | 75 | 否 | 有 | lighting/back: Rear shadow: lens geometry has no threshold consensus；material/right: Unresolved right hardware materials: 1, expected 2；structure/None: Insufficient final paired antenna evidence: left_end；side_review_observation/right: Current raw paired observations unavailable |
| iphone17promax | 82 | 是 | 有 | 无阻断，仅效果提示 |
| iphone17promax | 107 | 是 | 有 | 无阻断，仅效果提示 |

修复前：3 / 13 台全阶段完成，7 / 13 台有六面候选。最终：6 / 13 台全阶段完成，另 5 台有部分处理候选、2 台在几何检查失败。没有把部分候选记为通过。

测试：1182 项通过，3 项因原有外部素材条件跳过。原本已产出且未改方向的 6 台，全部 36 张最终 PNG 与基线哈希一致；输入原图 78 张哈希全部不变。
