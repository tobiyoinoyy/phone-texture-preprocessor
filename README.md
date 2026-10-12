# 手机贴图预处理工具

图片标准化、边框角度矫正、贴图结构对齐合并为一个入口：`phone-texture`。

- **原图标准化**：BiRefNet_dynamic 抠图、去支架、精裁；四个边框面默认摆正。
- **边框摆正**：已裁好透明图只做原像素刚性旋转，其他面原样保留。
- **结构对齐**：按钮以指定 SKU 为基准；上下天线以每个 SKU 的背板和实际模型 UV 接缝标定。沿用逐列天线倾斜修正和透明边缘插值。

每步都能单独使用。原图标准化已经摆正时，直接进入结构标注，不重复旋转。铝金属光照优化和不锈钢高光修复仍是独立 Skill。

## 安装

Python 3.9 以上，在本仓库运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[cutout]'
phone-texture --help
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

相机凸起、支架遮挡使自动边缘判断失败时，流程会停止。可在查看原图后提供 `--frame-guides 裁切标定.json`，为对应 SKU/边框面指定 `source_sha256`、原图坐标 `crop_xyxy: [左, 上, 右, 下]` 和 `rotation_ccw_degrees`。程序按经核对的范围裁切、按角度旋转，不再重复猜测边缘；原图哈希改变即拒绝复用。标定文件格式为 `{ "SKU_01": { "top": { ... } } }`，不要把某批坐标当作所有机型的默认值。

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

## Codex 入口

仓库包含 `.codex-plugin/plugin.json` 和 `skills/phone-texture-preprocessor/SKILL.md`，可作为插件使用。原标准化 Python API 和 `phone-normalizer`、`phone-normalizer-uv` 命令保留兼容；交付批次请用新的 `phone-texture` 入口，确保过程文件分开。

## 验证与来源

```bash
pip install -e '.[test]'
python -m pytest tests -q
```

代码来源及本批验证见 [整合记录](docs/integration.md)。真实业务图片、模型、标定文件和模型权重不随仓库提交。
