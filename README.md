# 六面工坊 0.2.1

本地浏览器批量处理手机六面照片。选择机型、原图文件夹、导出文件夹和并发数，即可处理并预览六面透明贴图。支持 iPhone 17 Pro / Pro Max 本项目铝材标准拍摄照片，包含方向识别、一键命名副本、光照优化、按本机背面对齐天线及固定模型四角接缝校色。

### 安装与启动（macOS）

需要 Python 3.9 以上。首次安装联网下载依赖与抠图权重，后续图片处理在本机完成。

```bash
git clone --branch v0.2.1 https://github.com/tobiyoinoyy/phone-texture-preprocessor.git
cd phone-texture-preprocessor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[cutout,browser]'
# 首次准备模型：同时缓存权重和模型实现，供浏览器后台离线加载。
python -c "from scripts.segmentation import _load_birefnet_model; _load_birefnet_model('ZhengPeng7/BiRefNet_dynamic', 'cpu')"
phone-texture web --home "$HOME/Documents/六面工坊"
```

也可下载 Release 源码包并解压，从创建虚拟环境一步开始。请保留完整源码目录，光照引擎也位于其中。

可选：执行 `python scripts/package_browser.py "$HOME/Documents/六面工坊"` 创建本机双击启动器。它复用当前源码与环境，并非独立安装包。无需安装 Codex 或 Agent。

[浏览器使用说明](docs/browser-app.md) · [0.2.1 更新记录](CHANGELOG.md)

六面工坊使用独立产品版本 **0.2.1**；同库处理引擎与 Codex 插件版本为 **1.6.1**。发布包不包含原始照片、业务模型、运行记录或模型权重；包含运行所需的方向特征库、固定 UV 接缝配置及测试夹具。结果仍需检查裁切、材质与所用模型的匹配效果。

## 处理引擎

图片标准化、边框角度矫正、贴图结构对齐合并为一个入口：`phone-texture`。

- **原图标准化**：BiRefNet_dynamic 抠图、去支架、精裁；四个边框面默认摆正。
- **边框摆正**：已裁好透明图只做原像素刚性旋转，其他面原样保留。
- **结构对齐**：按钮以指定 SKU 为基准；上下天线可按每个 SKU 正背面图片标定，或按实际模型 UV 接缝换算。沿用逐列天线倾斜修正和透明边缘插值。

每步都能单独使用。原图标准化已经摆正时，直接进入结构标注，不重复旋转。铝金属光照优化和不锈钢高光修复也包含在本仓库，由统一 Skill 按材质调用。

## 无 Agent 的原图自动处理

本项目 iPhone 17 铝材采集规范使用 `phone-texture auto 原图目录 --model iphone17pro --jobs 2`，由程序完成六面分割、标定、材质与贴纸分析、光照、背面天线对齐和检查。无需 GPT/Agent 补点、画蒙版或填参数。顶面只保留边框本体，摄像头台阶与镜头必须排除；不支持的输入明确失败。结果交由用户验收，不能冒充任意照片或真实模型 UV 均已验证。详见 [程序边界与顶面规则](docs/top-body-calibration.md)。

## 显式配置与签收入口（1.5）

双击 `手机图片一键处理.command` 打开六面处理窗口，或使用同一套程序：

```bash
phone-texture start 原图目录 --work 本轮过程 --output 六面成品 --material aluminum
phone-texture run 本轮过程
phone-texture status 本轮过程
```

现在桌面、命令行和四个 Skill 统一调度几何、按材质光照、结构对齐及验收发布。默认每个待处理 SKU 使用自己的背面作为上下天线基准，按钮独立标定。缺少蒙版、标定或真实复核时保存待办，补齐后继续，不重复旋转或校色。

完整参数、逐面光照任务及可选固定模型坐标门禁见 [统一运行流程](docs/workflow.md)。固定 UV 模型仍须先将新图片配准到参考坐标；图片自身背面对齐不能替代固定模型适配。下面各命令保留为同程序的子步骤。

同库另提供正在验证的原图自动候选入口，面向已命名、采用本项目采集方向和质检贴纸样式的 iPhone 17 铝材照片：

```bash
phone-texture auto 原图目录 --model iphone17pro --jobs 2
```

它从当前原图新建轮廓、纸面/材料范围和结构标记，不读取历史 SKU 蒙版或标定。省略保存路径时自动创建独立目录。只有全部 SKU 完成六面计算与数值检查才生成整批候选；`processed_needs_visual_review` 明确表示仍需最终视觉验收，不能当作正式发布或模型 UV 已通过。适用范围和状态含义见[自动候选入口](docs/workflow.md#原图自动候选入口)。桌面及原 `start/run` 的正式复核发布流程保持原有门禁。

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

## 本地浏览器版

运行 `.venv/bin/phone-texture web --home /工具与结果目录`，支持文件夹选择、批量勾选、1–8 并发、逐台进度、内存保护、停止与失败重试、六面预览和 ZIP 下载。调用相同的原图自动入口，保留原图；输出仍须视觉与模型适配验收。[使用与部署说明](docs/browser-app.md)。

当前 Mac 可用 `.venv/bin/python scripts/package_browser.py /交付目录` 安装双击入口。它复用本仓库环境，默认结果保存在交付目录的「处理结果」。
