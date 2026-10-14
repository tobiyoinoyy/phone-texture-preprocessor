# 正背面整圈金属：处理目标与效果门禁

Pro Max 曾出现 front 原样复制、back 仅轻度提亮，却以六面铝材优化完成交付。alpha 与保护区检查均通过，重黑边仍在。以后把范围、像素约束、实际效果分开验证。

## 先确定范围

- 全六面铝材优化包含 front/back **可见金属**。屏幕、镜头、玻璃和真实黑色结构缝单独保护，不等于整张正面跳过。
- 用户要求按去黑边参考效果处理时，front/back 登记 `perimeter`；其余处理面 `regional`，未授权或确认无须处理面 `unchanged`。每面写依据。只有用户明确缩小范围时才排除已要求的面。
- 默认只调明暗。只提亮后发灰时，先检查材质分类、原图饱和度与曝光；用户已要求同已接受效果保持连续金属颜色时，可以登记局部同材质改色依据，用本机原始稳定金属取样。不能自动套其他型号、颜色或宣传图底色。
- 整圈目标不允许外轮廓羽化回零，也不允许旧硬件/贴纸矩形冻结邻近金属。保留尺寸、alpha、实拍纹理、瑕疵和真实材料边界。颜色参考非空、支持度足够且所有校正值有限才可输出。

## 冻结 scope

几何 checkpoint 后、光照处理前，从源图独立描出需要校正的完整金属轮廓 `required_metal_mask`。再独立选择四边、四角的明确金属样点蒙版：`top/bottom/left/right/top_left/top_right/bottom_left/bottom_right`。不取镜头、贴纸、玻璃、天线和真实深色损伤；不能从成品差异生成样点，或直接复制工作蒙版作为独立效果检查。

根据本图同材质稳定参考与用户效果要求，记录样点的 CIE L* 中位数接受区间及 `reference_basis`。区间只属于当前图：深蓝不能套橙色门槛。上下限在处理前审核，不能因候选失败而放宽。门禁只检查被登记的样点，连续性、颜色和真实材料判断仍需要看图。

scope JSON 与 geometry.json 同样按 SKU、六面组织，例如单面条目：

```json
{
  "source_sha256": "几何基线SHA256",
  "action": "aluminum",
  "protection_policy": "sticker_only",
  "sticker_mask": "/过程/真实纸面.png",
  "mode": "perimeter",
  "reason": "按用户参考消除正背面金属重黑边",
  "reference_basis": "本机同材质稳定区域、目标及接受区间的依据",
  "required_metal_mask": "/过程/完整金属轮廓.png",
  "zones": {
    "top": {"mask": "/过程/top-witness.png", "min_L": 52, "max_L": 78}
  }
}
```

示例必须补齐八个区域；52..78 仅是本次橙色正面案例的数值，不能当默认参数。`regional`/`unchanged` 通常只需前四项。批次包含铝材处理时，front/back 默认需要 perimeter；若确实排除整圈处理，必须额外登记 `perimeter_exemption`（`reason` 为 `user_scope_excludes_perimeter`、`no_visible_metal` 或 `already_accepted`，以及具体 `basis` 和本图 `source_sha256`）。发布器不接受仅写“保护屏幕”而直接略过正面金属。`unchanged` 对应 action `copy`。先冻结：

```bash
python workspace.py lighting-scope --work 本轮过程 --spec 光照目标.json
```

发布时重新核对 scope 与蒙版哈希。不得覆盖已冻结目标；确需用户改变目标时开新复核工作区并记录原因，不能修改 proof 绕过。

## 生成并查看证据

```bash
python workspace.py perimeter-evidence --work 本轮过程 --sku SKU_01 --face front --candidate 候选/front.png --output 本轮过程/front-perimeter-review
```

back 同样执行。每个区域生成深浅背景各 1×/4×对照，左原图、右成品；输出 `review.json` 默认所有项为 false。看完整金属轮廓，再逐一看八区、实际纸边（沿用贴纸专项近景）和材质分界，确认无灰边、新增色块、光晕及纹理丢失后填写对应项。不能由程序自动全部置 true，不能只看缩略总览。某面确实无贴纸时确认源图无贴纸后记录该检查已完成。

将 review.json 路径作为该面 delivery 条目的 `perimeter_review`。发布同时检查：

1. action 必须履行冻结目标，不能将必处理面改成 copy。
2. 必需金属全部可编辑，alpha、区外及保护区仍锁定。
3. 八区实际输出的 L* 落在源图审核区间，直接复制或轻微提亮不能凭变化像素数混过。
4. 复核记录绑定本次源图、成品、scope、证据哈希，缺项和旧成品记录均拒绝。

`validation.json` 分别报告几何、像素约束和整圈效果。失败不创建正式成品目录。

## 回归

单元回归：`python -m pytest tests/test_lighting_review.py tests/test_six_view_workflow.py`。

实际 Pro Max 回归：

```bash
python -m scripts.regression_aluminum_perimeter --process 本次铝材过程目录 --output 新建回归目录
```

仓库 `tests/fixtures/aluminum-perimeter` 固定本次源图哈希、参考依据与样点；原图/候选保留在独立过程目录，不打包私人照片。回归必须同时拒绝原先 front 复制和 back 轻度提亮的错误结果、接受修正版效果、验证其余四面未变。生成的视觉记录默认待复核，不把数值通过等同于视觉签收。新型号/新色款要重新标定，禁止改本用例来迎合错误结果。

## 只保护贴纸与局部暗斑（2026-10-13 补充）

用户已明确“除了贴纸不需要其他保护”时，在 scope 中登记 `protection_policy: sticker_only` 和完整真实纸面 `sticker_mask`（空白蒙版表示无贴纸）。protected_mask 的可见像素必须与真实纸面一致。主铝材工具在执行前拒绝多余保护；六面发布器再校验一次。镜头/屏幕等按材质判断所需光照修正，不强制换色，也不能沿用包住金属的旧保护框。

四边、四角中位数可能遗漏细小黑斑；用户标出的每一处需新增 `local_checks`，例如 `{"antenna-left-upper": {"mask":"源坐标局部检查区.png","percentile":10,"min_L":50,"max_L":85}}`。这里数值仅为字段示例，按当前材质源图审核，深色天线和镜头不能套金属阈值。检查区必须可编辑、非保护且来自不透明源像素。其蒙版与 scope 一同冻结，证据生成器纳入每个局部的深浅背景 1×/4×对照；局部失败拒绝发布，即使整边统计合格。

本次曾通过的 v3 背板仍有天线保护细带、旧硬件边界造成的斑块，因此旧的 16 个整圈样点通过只证明这些样点合格，不应继续作为完整视觉通过的案例。回归新增 `scripts/regression_sticker_only.py`，单独检验这些局部及旧保护策略。

铝材 scope 必须显式填写 protection_policy；缺失即拒绝冻结，避免继承旧蒙版。用户确有额外保护要求才使用 reviewed_regions 并写明 protection_basis；本项目当前授权为 sticker_only。

## 侧面与局部任务也必须检查已登记暗斑

`local_checks` 对 `regional` 同样生效：冻结其蒙版哈希，发布时检查局部效果并要求 `local_review`，用 `perimeter-evidence` 生成同坐标 1×/4×、深浅背景对照。不能因侧面没有登记整圈 perimeter 而跳过天线端点。对已指出的问题，逐面登记具体交界；上次修复背板，不代表原样沿用的左右侧也通过了新规则。`copy` 面如包含已登记检查也必须通过实际效果与视觉复核。

2026-10-14 侧面案例：旧蒙版冻结了完整天线带，周围金属提亮后两端出现深块。局部修复从原始几何快照估计各材料光照，保留中段天线颜色、宽度及局部源细纹；不能将邻近金属光照直接外推后过度压暗天线，也不能用金属目标色涂平天线。保留适量边缘明暗并核对交界连续性。旧图需被局部效果回归拒绝，已验收按钮、上部及其余面逐像素锁定。
