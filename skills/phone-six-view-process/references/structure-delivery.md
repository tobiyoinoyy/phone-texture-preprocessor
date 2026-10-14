# 结构对齐与最终发布

## 为什么不能跳过

Pro Max 曾完成裁切与光照却漏掉结构对齐，上下天线相对背板偏移约 30 像素。光照校验的 alpha/像素范围合格不能证明位置正确。六面入口 `intake` 默认 `delivery_goal=model_textures`；`publish` 必须接收本次光照输出对应的结构过程与逐侧视觉复核。旧 intake 没有 delivery_goal 时也按模型贴图处理。

只有用户明确只要照片、裁切或光照样图时，在 intake 指定 `--delivery-goal photos --goal-reason '用户限定的照片任务范围'`。不得为了绕过失败而改成 photos；照片交付不得称作已对齐模型贴图。

## 执行顺序

1. 几何复核与 checkpoint。
2. 冻结光照范围、计算光照候选、完成逐面光照检查。保留完整六面候选目录，逐面 delivery.json 中 output 指向这里的**光照输入**，不是结构结果。
3. 从本次六面候选量取结构标记，运行下方 `align-structure`。
4. 查看结构报告中的同坐标原尺寸/4倍深浅背景对比、总览、正背面基准，以及整颗按钮和卡槽。将 `review-template.json` 另存为 `review.json`，按实际检查逐项填写。
5. `publish` 先验证几何和光照，再重新验证结构输入、像素重映射、两侧上下边界和宽度、两端/中央检测、按钮像素、其他四面、视觉记录，最后才复制结构结果到正式目录。

```bash
python skills/phone-six-view-process/scripts/workspace.py align-structure \
  --input 本轮六面光照候选 --spec 本次结构标定.json --process 独立结构过程
python skills/phone-six-view-process/scripts/workspace.py publish \
  --work 本轮过程 --plan 本轮过程/delivery.json \
  --structure-process 独立结构过程 --structure-review 独立结构过程/review.json \
  --output 六面成品
```

输入目录、结构过程、最终目录相互独立。结构报告及检查图不放进正式贴图目录。

## 标定字段

JSON 顶层为 `SKU -> left/right -> 配置`，必须覆盖本批所有 SKU 和两侧。源文件改变就重新标定；原始标定可以在**内容完全相同并重新核对**后复用。以下字段均按零起点像素中心坐标：

- `source_sha256`：本侧光照候选哈希。
- `basis`：`back_image`（默认，按当前 SKU 背板）或 `front_back_images`（前后两端分别约束，逐列插值）。
- `body_y`：本侧实际机身顶底，排除纸尾、透明空白。正背面也各自测量，不能直接按画布高度代替。
- `source_lines`：有序源行列表，或 `[锚点数, 图片宽度]` 数组，用后者记录倾斜。包含天线四条边界及其周围的固定行，首末画布行自动固定。
- `antenna_indices`：source_lines 中天线四边界的四个递增索引。其目标由参考图计算；其他锚点保持源位置。上下天线独立计算。
- `back`：`face: back`、`sha256`、`body_y`、`antenna_rows`（上天线上下边、下天线上下边），以及实际背板对应侧的 `x_range`、四个 `search_windows`、四个 `signs`。路径由当前 SKU 的 back.png 获取，不能引用别的 SKU。
- `front_back_images` 时还需要同结构的 `front`（`face: front`）及明确的 `front_at: left/right`。不要从文件名猜前后端。
- `button_ranges`：所有按钮/卡槽所在的完整行区间列表，半开区间 `[起始,结束)`；`button_basis` 说明独立量取依据。固定锚点必须使这些区间保持逐像素相同。本步骤只移动天线；若跨 SKU 按钮还需另行校准，应先完成对应几何标定与验收再建立当前光照基线，不能借天线映射顺带拉伸按钮。
- `measurements`：必须有 `left_end/center/right_end`，各有 `x_range`、`search_windows`、`signs`。端点检查区位于左右各四分之一内，center 覆盖中线，避开完全透明像素；每个 x_range 至少 3 列。

`search_windows` 是四个互不重叠的整数行范围 `[开始,结束)`，至少 3 行，涵盖当前边界及附近像素。`signs` 为相应边缘的灰度梯度方向（+1 变亮，-1 变暗），取决于实际材料。边缘太弱、落在搜索边界或不透明度不足即拒绝，必须查看原图重新标注，不能放宽阈值。固定检测容差为 2.5 个当前侧面像素，并同时检查条带宽度。检测是对已审阅区域的复核，不会自动识别所有手机的天线。

源锚点和参考 ROI 必须先从图像独立量取，不能为使输出通过而修改它们。技能任务中同样保留按钮/轮廓的原尺寸视觉检查；哈希只绑定材料，不证明标记语义正确。

## 状态与边界

`align-structure` 成功只生成 `needs_visual_review` 候选。复核记录绑定 report.json 哈希，包含每侧 `upper_edges_and_width/lower_edges_and_width/both_endpoints/buttons_and_card_tray/body_outline/native_and_4x` 六项及具体 notes、reviewer。执行 agent 检查后填写，无需额外要求用户逐项点击。

最终报告分别记录 `geometry_status`、`lighting_status`、`structure.status=image_space_passed`。每面同时保留 `lighting_output_sha256` 和最终 `output_sha256`。改变光照候选、基准图、结构配置、结果或检查证据后，旧复核失效。

这道门禁验收图片空间。已有 UV 缩放/平移时，继续按预处理 Skill 的 model_uv 流程检查当前 Blender 场景，另存真实接缝证据；不得用 image_space_passed 宣称 3D 已验收。没有可可靠标注的天线时保持待复核，不改交付目标绕过。模型 UV 校验不由此图片门禁自动完成。
