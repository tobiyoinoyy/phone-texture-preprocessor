# 安装与运行

## 依赖定位

当前工作流不内嵌或复制上游算法。先解析本 Skill 符号链接，向上两级的仓库即 `phone-texture-preprocessor`；若单独复制了 Skill，则定位另外安装的该仓库。查找当前 agent 的技能目录，读取铝金属/不锈钢对应 SKILL.md。禁止在共享版本中写死个人主目录。

仓库来源：
- https://github.com/tobiyoinoyy/phone-texture-preprocessor
- https://github.com/tobiyoinoyy/phone-aluminum-frame-restore
- https://github.com/tobiyoinoyy/phone-six-view-restore

这些仓库需要访问权限；其他人可获得仓库读取权限或本地完整工具包，只有一份 SKILL.md 不包含模型和算法。拿不到依赖时列出缺项，不假装已安装。用当前 agent 的 Skill 安装方式注册本目录；Codex 可将本目录复制或链接到 `~/.codex/skills/phone-six-view-process`，几何仓库及两个光照 Skill 保持完整。

在几何仓库按其 README 安装：`python3 -m venv .venv`，然后 `.venv/bin/python -m pip install -e '.[cutout]'`。首次抠图下载 BiRefNet_dynamic 权重；先检查网络、模型缓存和可用设备。`phone-texture --help` 和三个 Skill 文件都可读取才算依赖就绪。不锈钢实际生成另需 agent 内置图像编辑工具。铝默认流程只用确定性算法。

## 运行约定

下列变量只是示意，换为当前环境实际路径。用几何仓库 `.venv/bin/python` 运行本 Skill 的 `scripts/workspace.py`（需要 Pillow、NumPy）。

```bash
python workspace.py intake --input 原图 --work 本轮过程 --skus SKU_01 SKU_02 SKU_07 SKU_10
phone-texture standardize 本轮过程/input --output 本轮过程/geometry --process 本轮过程/geometry-records
# 查看 geometry-records/geometry-review，填写 review-template.json 的逐面检查和独立像素检查区
phone-texture approve-geometry 本轮过程/geometry-records --review 已复核.json --output 本轮过程/geometry
python workspace.py checkpoint --work 本轮过程 --geometry 本轮过程/geometry --geometry-report 本轮过程/geometry-records/report.json
```

`intake` 同时支持单个六面文件夹；单台默认命名 SKU_01，可用 `--single-name` 更改。批次忽略根目录预览图，只选择所列 SKU。重复面、缺面、损坏图、输入/过程互相包含均拒绝。原始边框方向不同于预处理约定时，先保存精确 90° 旋转副本及变换记录，再以新副本哈希建立标定，不能改原图。

仅处理部分 SKU 时，裁切标定也只保留所选 SKU；预处理器会拒绝含其他 SKU 的标定。先核对每个条目的源图哈希，不修改标定哈希去迎合新图。

有经核对的源坐标轮廓/局部支架蒙版/角度时给 standardize 追加 `--geometry-guides 文件.json`。矩形只作为完整主体外接范围；不能靠裁掉下缘去支架。详见[几何门禁](../../../docs/geometry-review.md)。没有自动通过也没有有效标定则停止该图的后续步骤。

按材质 Skill 将光照候选存为 `本轮过程/light/SKU_XX/面.png`，蒙版和参数放 `本轮过程/light-records/`。从刚才的 geometry 图取样，不从原始 JPG 或旧成品估计当前像素坐标。已有几何输出可以复用对应的完整几何复核报告后 checkpoint，避免重新旋转；缺少证据的旧成品先复核轮廓，不能直接冻结。

## 光照范围先冻结

几何 checkpoint 后、光照处理前，按[整圈金属验收](perimeter-review.md)建立六面 scope。所有发布（含 copy 面）均须登记；旧工作区也要先核对目标，不能从已完成的 delivery 反推目标。`lighting-scope.json` 冻结后发布清单必须履行其 action。整圈目标另设效果和视觉门禁。

## 发布清单

写 `本轮过程/delivery.json`，逐 SKU、逐面登记，六面均必填。例中仅展示两个面，其余同样登记：

```json
{
  "SKU_01": {
    "front": {"action": "copy", "reason": "本任务只处理侧边；正面金属另行登记，不将屏幕保护视为整面免处理"},
    "left": {
      "action": "aluminum",
      "output": "/本轮过程/light/SKU_01/left.png",
      "allowed_mask": "/本轮过程/light-records/SKU_01/left-editable.png",
      "protected_mask": "/本轮过程/light-records/SKU_01/left-protected.png",
      "job": "/本轮过程/light-records/SKU_01/job.json",
      "reason": "源色模式，边框与按钮独立拟合"
    }
  }
}
```

`action` 为 `copy/aluminum/stainless`，copy 从登记的 geometry 按字节复制；修复面必须提供落盘结果、源坐标有效允许蒙版、保护蒙版、参数记录和说明。铝工具导出的 editable/protected 可直接使用；不锈钢需由高光、alpha、保护求最终允许蒙版。这份清单记录范围依据，发布器验证像素边界，不代替对材料、job 或视觉效果的判断。仍有未完成的必需光照处理时不要将 action 改为 copy 冒充成功。

```bash
python workspace.py publish --work 本轮过程 --plan 本轮过程/delivery.json --output 六面成品
```

输出目录必须新建且与过程、原图互不包含。所有 SKU 都检查合格才发布；失败保留过程供检查。`validation.json` 存源/基线/输出哈希、变化像素与检查结果；`overview.png` 是分面前后总览。用图像工具查看总览，再逐一检查完整按钮和明显损伤近景。不能仅凭脚本退出码交付。

## 贴纸保护边界：防止纸外出现黑色矩形

曾出现 SKU_07 底部贴纸保护多包住数行金属：保护像素校验全通过，成品却留下深色块。因此“保护区没变”不能证明保护区画对了。

对参与光照处理、含贴纸的面，清单必须登记 `contains_sticker: true` 和 `boundary_checks`。从**当前原图**独立描出含抗锯齿边缘的纸张轮廓 `feature_mask`，另存贴纸左右和底端附近确定属于金属的 `adjacent_metal_mask`，两者互不重叠，并绑定 `source_sha256`。这两张验收蒙版不是直接复制旧保护框，也不能由成品差异反推。

```json
{
  "contains_sticker": true,
  "boundary_checks": [{
    "source_sha256": "本次几何基线的SHA256",
    "feature_mask": "/过程/实际纸张轮廓.png",
    "adjacent_metal_mask": "/过程/纸外金属检查区.png"
  }]
}
```

将这些字段加入已有的逐面清单项目。发布器要求纸张全在保护区、纸外金属全部可编辑；否则拒绝发布。材料分类仍需 agent 看原图判断，程序不会自动识别真实纸边。

- 逐行/轮廓追踪纸张左右边、斜边与底端；不能用包围盒代替轮廓，不盲目腐蚀或缩小全部保护区。
- 准确保护纸面后，纸外金属的校正场应连续。不要在旧粗保护框附近向零羽化，也不要因纸张不参与拟合而冻结邻近金属。需要过渡时放在修复区域外侧，与已接受校正平滑衔接。
- 保存纸面保护叠图、纸外金属检查图，以及左右/底端至少原尺寸与 4 倍对照。逐一检查黑块、光晕、阶梯边；总览不能代替纸边近景。
- 修复从未经校色的几何基线重算；锁定纸面 RGBA、全部 alpha、修复邻域以外及其他图片。旧错误蒙版应在新增边界检查中失败，收紧后的蒙版通过，作为回归样例。
