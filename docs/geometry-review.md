# 几何复核与发布（1.1）

几何标准化分为生成候选和复核发布。`standardize`、`deskew` 成功表示候选生成完毕：`candidates_complete=true`，`geometry_status=needs_review`，`success=false`。正式输出目录此时不会创建。退出码 0、尺寸、哈希、支持检测置信度和光照像素校验均不代替几何验收。

```bash
phone-texture standardize 原图批次 --geometry-guides 几何标定.json --output 几何成品 --process 几何过程
# 查看几何过程/geometry-review 下的原图叠加、四角和完整下缘。
# 复制 review-template.json，逐面填写实际检查结果和说明。
phone-texture approve-geometry 几何过程 --review 已复核.json --output 几何成品
python skills/phone-six-view-process/scripts/workspace.py checkpoint --work 本轮过程 --geometry 几何成品 --geometry-report 几何过程/report.json
```

这是执行 agent 的检查步骤，不额外要求用户逐张点击批准。检查失败时修正蒙版并生成新候选；无法判断实际边界时保留待复核状态。已有六面 PNG 也必须有与当前输入和图片绑定的几何复核报告，不能通过手写 `geometry.json` 绕过流程。旧报告不会自动升级为已通过。

## 源坐标标定

`--geometry-guides`（兼容参数名 `--frame-guides`）支持六面。角度、保留轮廓、支架删除和贴纸保护分别登记。蒙版为经过 EXIF 正向解释的原图等尺寸 L 模式 PNG；路径用绝对路径，哈希为文件 SHA256。几何只使用原图 RGB。

```json
{
  "SKU_01": {
    "top": {
      "source_sha256": "原图哈希",
      "rotation_ccw_degrees": -0.55,
      "retain_mask": {"path": "/过程/top-outline.png", "sha256": "蒙版哈希"},
      "support_remove_mask": {"path": "/过程/top-stand.png", "sha256": "蒙版哈希"},
      "protect_mask": {"path": "/过程/top-paper.png", "sha256": "蒙版哈希"},
      "crop_xyxy": [0, 0, 1594, 1594]
    }
  }
}
```

- `retain_mask`：可选，255 保留，0 排除，灰度表示原图边缘覆盖率；与源 alpha 取较小值，不补造机身。
- `support_remove_mask`：可选，仅确认属于支架的局部区域非零。覆盖完整支架及其软边，包括远离接触线的斜柱，不能只处理柱顶。
- `protect_mask`：可选，贴纸纸尾等明确应保留的原像素。删除区与保护区或非零保留区相交会拒绝执行。
- `crop_xyxy`：可选，只是完整保留内容的外接工作范围。切到任何保留像素即失败，不再用矩形截断轮廓去支架。未提供时为完整原图。
- 顺序固定：原坐标轮廓/局部去支架 → 整幅源画布旋转一次 → 仅裁全透明外画布 → 左右面的精确 90° 转向。禁止先转一次再套旧裁切坐标。

人工轮廓也会画错。需要对照原照片完整下缘和两端圆角；纸尾按自身轮廓保护。不能把删除范围写成统一裁掉最后几行，也不能用全图腐蚀、縮小主体或放宽阈值通过。

## 固定证据与独立像素检查

每面生成源图边界叠加、去支架后的源画布、旋转后的画布、最终裁切重新置回对应画布的检查图，以及四角和完整下缘的 1× / 4× 深浅背景图。总览用于定位，必须再看接触位置与角部细节。出现 `ambiguous_body_edges`、`heuristic_support_cut`、`manual_geometry` 或 `extrapolated_frame_edges` 时，还必须提供三阶段独立像素检查。

复核文件绑定候选报告哈希，六面逐面填写：

```json
{
  "report_sha256": "候选 report.json 的哈希",
  "reviewer": "执行者",
  "items": {
    "SKU_01/top": {
      "source_overlay": true,
      "four_corners": true,
      "complete_lower_edge": true,
      "contact_before_rotation": true,
      "contact_after_rotation": true,
      "dark_light_backgrounds": true,
      "notes": "具体检查位置、原图交界依据及纸尾处理结论",
      "boundary_assertions": {
        "after-support-source-canvas.png": {
          "foreground": {"path": "/过程/独立机身检查区.png", "sha256": "哈希"},
          "background": {"path": "/过程/独立支架检查区.png", "sha256": "哈希"}
        }
      }
    }
  }
}
```

上述 `boundary_assertions` 还需包含 `after-rotation-canvas.png` 和 `after-crop-canvas.png`，蒙版尺寸/坐标分别对应这两个检查画布。机身及纸尾内部确定应保留的像素要求 alpha≥128；确定是支架/背景的像素要求 alpha=0，任一像素不合格就拦截。阈值不随样本降低。

检查区必须从原图独立标注，覆盖完整下缘、左右圆角、接触区和支架远端。不能直接复制处理蒙版，不能根据成品差异反推“正确区域”；程序也拒绝与处理/保护蒙版哈希相同的验收蒙版。旋转后的内部检查区可按同一个刚性变换投影，排除混合边界像素。混合边缘仍需要在 1× / 4× 图上检查白边和缺口。检测置信度只说明算法判断稳定。

`approve-geometry` 验证源图、成品、证据文件、独立检查区和完整六面记录；任一变化或未复核都不发布。`checkpoint` 与光照 `publish` 再验证这份证据。光照报告分别记录 `geometry_status` 和 `lighting_status`。

## iPhone 17 Pro Max 回归案例

`tests/fixtures/iphone17promax-boundaries.json` 保存本次三张原图哈希、经原照片核对的局部边界、独立机身/支架检查区及纸尾标记。原图和 BiRefNet_dynamic 蒙版留在本地。标注由本次执行者核对，不宣称已获得用户逐像素签核。

```bash
python scripts/regression_iphone17promax.py --source 六面原图目录 --mattes 原尺寸BiRefNet蒙版目录 --previous 旧六面几何目录 --process 新回归目录
```

该命令核对缓存与源 RGB 一致，生成六面候选，检查三个阶段，重现旧版失败，并要求另外三面逐像素不变。候选仍需完整几何复核后发布。自动测试另包含圆角被切、单个软边支架残留、纸尾误删、过期复核、证据变更、缺少几何批准时冻结/发布失败等用例。

2026-10-13 本地实图验证：旧 top 在标注机身区域丢失 400 像素；旧 front / back 在标注支架区分别残留 600 / 390 像素（仅原先接触区统计）。新候选三阶段检查均为零。left / right / bottom 与旧几何 RGBA 一致。另回归 SKU_01、02、07、10：16 张自动处理面逐像素一致；8 份顶底旧矩形标定被新规则拦截，需补充轮廓后重新验收，不沿用历史“已通过”。
