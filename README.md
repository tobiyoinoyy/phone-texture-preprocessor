# 手机贴图预处理工具

图片标准化、边框角度矫正、贴图结构对齐合并为一个入口：`phone-texture`。

- **原图标准化**：BiRefNet_dynamic 抠图、去支架、精裁；四个边框面默认摆正。
- **边框摆正**：已裁好透明图只做原像素刚性旋转，其他面原样保留。
- **结构对齐**：按钮以指定 SKU 为基准；上下天线可按每个 SKU 正背面图片标定，或按实际模型 UV 接缝换算。沿用逐列天线倾斜修正和透明边缘插值。

每步都能单独使用。原图标准化已经摆正时，直接进入结构标注，不重复旋转。铝金属光照优化和不锈钢高光修复也包含在本仓库，由统一 Skill 按材质调用。

## 安装

Python 3.9 以上，只需一个仓库：

```bash
git clone https://github.com/tobiyoinoyy/phone-texture-preprocessor.git
cd phone-texture-preprocessor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[cutout]'
phone-texture --help
python scripts/install_skills.py
```

只处理已裁图可用 `pip install -e .`，不需要下载抠图模型。首次抠图会获取 `ZhengPeng7/BiRefNet_dynamic` 模型权重。也可用 `python -m phone_texture` 运行全部命令。

## 输入与输出

输入是批次目录，每个 SKU 一个文件夹，内含 `front/back/left/right/top/bottom` 六面图片。支持 PNG、JPEG、WebP、TIFF、BMP，文件名大小写不限，同一面不能有多份文件。

原始边框照片要求手机长边横向，标准化时 left 逆时针 90°、right 顺时针 90°。已竖向摆好的左右贴图使用 `deskew`，不要再走原图入口。

成品目录只放各 SKU 六面图片。标注页、检查总览、报告和中间图片全部放在独立过程目录。输入、输出和过程目录互不包含，输出要求新目录。任意图片失败时不发布整批成品，保留过程记录供检查。

## 使用

从原始照片开始：

```bash
phone-texture standardize 原图 --output 已标准化 --process 过程/标准化
```

候选生成后先按[几何复核流程](docs/geometry-review.md)检查六面四角、完整下缘和支架接触区，再用 `approve-geometry` 发布。`standardize` / `deskew` 不直接创建正式成品。自动检测失败时使用 `--geometry-guides` 提供源坐标轮廓、局部支架蒙版、角度及哈希；旧矩形不能承担去支架作用，`reviewed` 标记不能跳过检查。

如果已有透明贴图，只修选中的面；未选择的面逐字节保留：

```bash
phone-texture deskew 已裁图 --faces top bottom --output 已摆正 --process 过程/摆正
```

统一左右侧尺寸并创建标注页；默认尺寸按基准图确定，也可显式指定 `--width`、`--height`：

```bash
phone-texture prepare 已标准化 --reference SKU_01 --process 过程/结构标注
```

打开生成的 `标注按钮.html`，标记上下天线各两条边界、按钮上下沿。再结合 Blender 实际 UV 完成逐 SKU 天线目标，参见 [标定说明](docs/calibration.md)。源标记必须基于本次 `normalized` 图片，不能套用旧裁切图的坐标。

```bash
phone-texture apply 过程/结构标注 --landmarks 已标定.json --output 最终贴图 --process 过程/结构对齐
```

检查 `过程/结构对齐/overview.png`，所有 left 排在一起、所有 right 排在一起。天线上下边界及宽度还需放回当前 Blender 场景核对；总览和公式校验不能代替实际接缝验收。

## 六面照片含光照的一体化 Skill

使用 `$phone-six-view-process 处理这个六面图文件夹`，由 [统一工作流 Skill](skills/phone-six-view-process/SKILL.md) 调用几何工具与对应光照 Skill。支持单个六面文件夹和批量 SKU，包含输入快照、光照基线、保护范围验证及六面发布。几何、铝金属、不锈钢四个 Skill 的说明、脚本和测试均在 `skills/` 内；安装与运行见 [说明](skills/phone-six-view-process/references/run.md)。

## Codex 入口

仓库包含 `.codex-plugin/plugin.json` 和 `skills/phone-texture-preprocessor/SKILL.md`，可作为插件使用。原标准化 Python API 和 `phone-normalizer`、`phone-normalizer-uv` 命令保留兼容；交付批次请用新的 `phone-texture` 入口，确保过程文件分开。

## 验证与来源

```bash
pip install -e '.[test]'
python scripts/test_all.py
```

代码来源及本批验证见 [整合记录](docs/integration.md)。真实业务图片、模型、全量业务标定和模型权重不随仓库提交；本次回归只保存三张原图的指纹与边界检查标注。

正背面图片参考使用 `alignment_basis: front_back_images`，无需先提供模型，步骤见[结构标定](docs/calibration.md)。只保护贴纸使用 `sticker_only`；局部暗斑纳入独立暗部检查，见[光照验收](skills/phone-six-view-process/references/perimeter-review.md)。

## 单库内容

- `skills/phone-six-view-process`：六面处理统一入口与验收发布。
- `skills/phone-texture-preprocessor`：抠图、精裁、摆正、结构对齐。
- `skills/phone-aluminum-frame-restore`：铝金属光照处理。
- `skills/phone-six-view-restore`：不锈钢高光修复。

安装器默认使用 `$CODEX_HOME/skills`，未设置时使用 `~/.codex/skills`。已有旧版可运行 `python scripts/install_skills.py --replace`，安装器将旧目录或链接留作带时间戳的备份。安装后向 agent 发送：

> 使用 phone-six-view-process 处理这个六面图文件夹：裁切、摆正、按材质优化光照，按本次要求校准按钮和天线，检查轮廓与局部暗斑。六面成品与过程分开，并用一张图展示全部结果。

旧光照仓库仅保留历史来源，后续改动在本库维护；来源版本记录于 [lighting-origins.json](docs/lighting-origins.json)。

## 模型贴图发布检查（1.4）

六面入口默认交付模型贴图，结构对齐是必需步骤。光照候选生成后执行 `workspace.py align-structure`，核对上下天线四边界、宽度、两端与按钮，再以 `--structure-process` 和 `--structure-review` 发布。缺步骤或使用过期结果会被拒绝；只有明确的照片任务可在 intake 使用 `--delivery-goal photos --goal-reason ...`。完整字段和命令见[结构发布门禁](skills/phone-six-view-process/references/structure-delivery.md)。图片检查通过不替代当前模型 UV 接缝检查。
