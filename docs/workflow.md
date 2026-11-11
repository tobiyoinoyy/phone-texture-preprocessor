# 六面图统一运行流程（1.5）

开发独立运行优先使用下述 `auto`：无 LLM/Agent 运行依赖，不要求逐套补点或填表。相机排除、适用规范、失败处理和验收边界见 [顶面本体标定](top-body-calibration.md)。

桌面 `手机图片一键处理.command`、命令行与四个 Skill 均调用 `phone_texture`。
几何、铝材和不锈钢算法仍分别维护；不再维护另一套桌面处理流程。
原 `skills/phone-six-view-process/scripts/workspace.py` 仅转发到包内实现，保留历史调用兼容。

## 原图自动候选入口

`phone-texture auto 原图目录 --model iphone17pro --jobs 2` 面向本项目 iPhone 17 铝材质检采集规范，输入为命名完整的六面目录，或多 SKU 子目录。它自动创建过程/候选目录，也可指定 `--work` 和 `--output`；`--skus` 限定本批范围。每 SKU 重新运行 BiRefNet_dynamic 和当前图像分析，不输入或载入历史蒙版、裁切坐标、天线/按钮标记、旧成品与复核记录。不锈钢、任意材质和未命名照片不在此自动入口的适用范围内。

该采集规范要求原始侧面横向、正背面直立、背面未经镜像；back/bottom 使用本批同款印字质检纸。底部标签的红色印字定位点和当前边框厚度只给出有界搜索邻域，纸边仍从本次原图测量。没有可靠印字/纸面、露出的机身边界、按钮或天线时失败并保留诊断，不用全黑保护蒙版或复制原图替代处理。不同贴纸样式、明显透视、其他型号需要另行验证适用性。

流程包括六面分割、原图轮廓/支架分析、四面摆正、当前材料域与源色光照校正、正背面整圈/侧面天线端点效果检查、当前背面天线四边界与独立硬件标记、两端与中央重映射检查、全部六面输出。几何只处理 alpha 和一次刚性旋转；光照锁定尺寸/alpha，只保护识别出的真实纸面，按钮和天线分别分析。原图快照、算法参数、源哈希、蒙版、原尺寸/4倍证据与耗时另存。

自动采集规范 v2 对每张 front 使用同一模型的 1536 长边上限，其余面 1024；此设置针对窄金属边可靠取样，不按 SKU 读取旧配置。正背面的旋转角由机身两侧长边拟合并用本次原图 RGB 边缘支持验证，不能由支架接触线推断。背面塑料天线与金属分别估计光照，倒角的法向亮暗轮廓从当前源图估计，再校正相对该轮廓的切向照明，不能把亮倒角和深色窄带压成一种亮度。

自动范围中的明度上下限是处理前冻结的源图及源参考相对损坏检查，防止异常压黑或提亮；真实磨损、不同倒角和反光仍需逐处视觉判断。该数值通过不证明黑边已去除或材料分类正确，不能据此填写视觉签收。几何中的外推轮廓和去支架同样保留未复核状态。

自动入口输出的是完整处理的**待验收候选**。`processed_needs_visual_review` 只表示六面算法与相应数值检查运行完成，不表示视觉已通过。自动程序不勾选 `source_overlay`、`paper_edges`、`native_and_4x` 等视觉项，也不伪造 reviewer。正式接受仍须实际查看轮廓、纸边、整圈金属、天线两端和完整按钮，再进入下文同一套复核/发布门禁。缺少视觉或模型证据时保持相应状态；不得以自动候选代替固定 UV 模型适配或真实 Blender 接缝验收。

`--jobs` 是同时执行的工作进程数，目前限制为 1–4；其他 SKU 排队。每个任务独立加载并释放模型、重新生成参数和蒙版，默认每进程 2 个 CPU 计算线程。队列容量与实际模型并发分开；接受 30 个排队任务不代表启动 30 个模型。只把完整六面且数值检查通过的任务纳入处理吞吐，停止/失败不计成功；首次资料自动分析与证据生成均计时，最终人员视觉验收另记。

分割另有批内共享的 2 个执行名额（单任务为 1）：1536 正面推理及其清理完成后才释放名额，其他工作进程可并行做光照、结构和证据输出。`jobs=3/4` 不表示同时进行 3/4 次 GPU 推理；报告同时记录任务并发与最大分割并发。

macOS 的首次处理压测可运行 `python scripts/benchmark_automatic.py 原图目录 --records 新目录 --levels 1 3 4`，可选依赖为 `.[benchmark]`。每档重建全部输入分析与六面结果，包含模型加载、证据与文件输出；已安装的模型权重和系统文件缓存可以预热。报告分别列出整批时间/数量、摊销吞吐与单任务延迟，失败不报告成功速度。系统压力与交换空间由主机采样；进程 RSS 求和包含共享页，MPS 数值是推理返回后的单进程快照，不是真实显存峰值。代码指纹中途改变即停止对比；不把自动候选耗时称为最终人员验收耗时。

每台任务的 `result.json` 中，`peak_process_rss_bytes` 统一为字节：Linux 的 `ru_maxrss` 原值按 KiB 乘 1024，macOS 原值已是字节。`peak_process_rss_measurement` 保留原值、原单位、平台与采集范围；未核实单位的平台返回 `null` 字节值。该数是工作进程自启动以来的 RSS 高水位，复用工作进程时可能包含前一台任务，不是本 SKU 独立峰值、进程树总峰值或 GPU 显存。修复前的 Linux 报告在这个 bytes 字段内存的是 KiB；历史文件不自动改写，比较时须按运行平台和代码版本明确换算。单位依据：[Linux getrusage(2)](https://man7.org/linux/man-pages/man2/getrusage.2.html)、[Apple XNU getrusage(2)](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/getrusage.2)。

## 默认基准

默认 `back_image`：每个待处理 SKU 的背面分别约束该 SKU 左右侧上下天线四条边界，侧面天线保持横向平直。按钮独立标定，不能从天线位置推断。SPU 批次包含多 SKU 时也不共享背面坐标。

确实需要正背面两端独立约束的任务，在**建批时**指定 `--alignment-basis front_back_images`。基准写入批次契约；不能因某步失败改成另一种基准继续。真实模型 UV 换算仍可用 `prepare/apply` 的 `model_uv` 子流程，并另存当前 Blender 场景证据；完整发布的结构门禁验收的是图片空间。

关联固定 UV 模型时，使用下方的模型坐标门禁。每 SKU 自身背面对齐只证明图片空间关系，不能替代固定模型坐标配准。

## 新建、继续与查看进度

在仓库根目录使用 `.venv/bin/phone-texture`（激活环境后可省略前缀）：

```bash
phone-texture start 原图目录 --work 本轮过程 --output 六面成品 --material aluminum
phone-texture run 本轮过程
phone-texture status 本轮过程
```

输入支持一个六面目录，或多个 SKU 子目录。`--skus SKU_01 SKU_02` 限定批次选择；单台可用 `--single-name` 命名。

- `--input-kind raw`（默认）：原始横向边框照片，BiRefNet_dynamic 抠图、四面摆正。
- `--input-kind cutout`：已抠图且左右已竖向的六面图，摆正四个边框面，不重新抠图。
- `--input-kind existing`：六张已有 RGBA PNG，原样进入本轮几何复核，不再旋转。
- `--material aluminum/stainless/mixed`：材料方法；mixed 按逐面光照范围分流。
- `--model-reference 清单.json`：可选固定模型坐标清单，不能传旧版 `--uv-profile`。

默认目标为 `model_textures`。只有明确的纯照片任务才使用 `--delivery-goal photos --goal-reason 用户限定的范围`；不得用它跳过结构失败。

新建会冻结原图快照、输出目标、输入类型、材质、对齐基准与模型指纹。
`run` 使用独立锁避免桌面与命令行同时处理一批，已有步骤验证后续接，不重复摆正或重复校色。
参数、源图、蒙版或输出改变后，旧结果不能复用；新建过程版本，不覆盖原图和正式成品。

## 资料与阶段

`workflow.json` 是批次配置。路径字段可为相对过程目录的路径；**材质任务内部的图片和蒙版路径使用绝对路径**。桌面“补充本步资料”会把选择的文件登记到对应字段。Agent 可直接编辑配置并继续，无需让用户逐个填写。

```json
{
  "geometry": {"guides": null, "review": null, "device": "auto"},
  "lighting": {"scope": null, "jobs": null},
  "structure": {"spec": null, "review": null},
  "model_review": null
}
```

上面只展示可补充的部分，保留程序生成的 schema、版本及冻结字段。

1. **几何候选**在 `geometry-records/staged`；按[几何复核](geometry-review.md)检查真实轮廓及支架区域。填写 `review-template.json` 的副本，把路径写入 `geometry.review`。必要的原图坐标标定先填 `geometry.guides`，然后再开始该批几何处理。
2. 通过复核后，程序发布到过程内 `geometry` 并冻结光照基线。
3. 建立逐面材料范围，路径写入 `lighting.scope`。格式与检查见[整圈与局部效果](../skills/phone-six-view-process/references/perimeter-review.md)。铝材使用 `sticker_only`，真实纸外不能冻结；整圈目标、独立样点、贴纸边界与 local_checks 都保留。
4. `lighting.jobs` 负责逐面调度下述材质任务，输出到 `lighting/SKU/面.png`；蒙版与验证存于 `lighting-records`。引擎不会自动判定材料或生成 AI 补片。
5. 光照效果和像素约束通过后，从**本次光照输出**标定侧面与本机背面，写入 `structure.spec`。格式见[结构门禁](../skills/phone-six-view-process/references/structure-delivery.md)。默认 basis 可不填，即 back_image；两侧必须齐全。
6. `structure-records` 保存重映射、报告和对照。实际视觉复核后将复核记录路径写入 `structure.review`。
7. 如关联固定模型，还须通过下方模型坐标门禁。
8. 全部通过才发布到冻结的成品目录，每 SKU 仅六张 PNG；一张分面总览及验证记录保留在过程目录。

`needs_review` 表示等待资料或视觉复核，`failed` 表示输入或验证失败，`completed` 才表示本批已发布。缺少材料或 AI 工具时保留待办，不改为 copy。Agent 填复核记录必须基于实际检查，测试中的合成签收不能用于真实照片。

## 光照任务格式

`lighting.jobs` 指向完整的 `SKU -> 六面` 清单；每面 action 必须履行冻结的光照范围。仅示例两个面，实际六面均须登记：

```json
{
  "SKU_01": {
    "front": {"action": "copy"},
    "left": {
      "action": "aluminum",
      "job": "light-inputs/SKU_01-left.json",
      "contains_sticker": false,
      "local_review": "/过程/left-local-review.json"
    }
  }
}
```

`copy` 仅适用于已登记的范围依据，程序从几何基线按字节复制。
含真实贴纸的处理面照常登记 `contains_sticker` 与独立 `boundary_checks`。
整圈面登记 `perimeter_review`，含局部检查的其他面登记 `local_review`。首次运行可先生成候选，再补这两个复核路径并继续；不重跑材质引擎。

**铝材** job 采用 [material-harmonization](../skills/phone-aluminum-frame-restore/references/material-harmonization.md) 的原有 schema，每任务一面，`views[0].name` 必须为对应面名、source 为该面几何基线。默认 `source_lightness`。统一流程管理 output_dir 和 records_dir，任务中可省略它们；保持材料域、独立参考、真实纸面与 required_editable_mask 的实际审核。

**不锈钢** job 使用合成 API 参数：

```json
{
  "source": "/过程/geometry/SKU_01/right.png",
  "candidate": "/过程/ai/right-candidate.png",
  "highlight_mask": "/过程/masks/right-highlight.png",
  "protect_mask": "/过程/masks/right-protected.png",
  "confirm_alignment": true,
  "confirm_texture_references": true,
  "texture_reference_boxes": [[10, 300, 40, 400]]
}
```

坐标仅为字段说明，必须按当前源图量取；示例不能直接执行。AI 补片按[不锈钢 Skill](../skills/phone-six-view-restore/SKILL.md)由可用的图像编辑工具生成，再交给同库合成器。没有候选时不能声称完成。统一流程设置 source 对应输出及 records_dir，严格保持高光区外原像素。独立调用两个引擎均使用 `phone-texture lighting --material ... --job ...`，独立任务要自行提供输出与记录路径。

## 固定 UV 模型

建批时关联已验收的 `phone-model-uv-reference-v1` 清单。程序核对 GLB、六张参考图和材质引用指纹；back、back-camera、backdetal 共用 back.png。

最后的模型复核模板由 `run` 生成，也可显式生成：

```bash
phone-texture model-review-template --input 结构候选 --model-reference 模型坐标清单.json --output 模型复核模板.json
phone-texture verify-model --input 结构候选 --model-reference 模型坐标清单.json --review 已复核.json
```

在已验收参考图和本次图片上独立量取四角轮廓、上下天线两端四边界、镜头中心、每个按钮两角和顶底接口边界。填写 reference_points/output_points、逐面视觉项和 notes、reviewer，绑定原尺寸及 4 倍证据（path/sha256/scale）。像素中心 convention 与模型清单一致；不同分辨率按归一化坐标换算，误差按参考图像素检查（2.5 px），左右侧天线保持横向平直。将记录路径写入 `model_review`。

这是对**已配准图片**的验收，不是自动语义配准算法；位置不匹配时先回到几何配准并新建过程版本，不能移动目标清单或改模型 UV 来让新 SKU 通过。轮廓和接口仍需从参考实图独立核对，不能仅靠 JSON 尺寸。
图片固定坐标检查与当前 Blender 实际场景检查分别记录；`model_uv_verified` 保持 false，不把离线坐标门禁称作实时模型无缝验收。

## 兼容入口

原 `standardize/deskew/approve-geometry/prepare/apply` 是几何子步骤；`intake/checkpoint/lighting-scope/perimeter-evidence/align-structure/publish` 也已并入 `phone-texture`。旧 normalizer API、GUI 脚本与旧 workspace 调用保留历史兼容，新任务使用上述完整入口。模型发布使用同一套几何、光照、结构及可选固定坐标门禁，不能仅以子步骤成功交付。

## 显式极限并发实验

常规 `auto` 仍限制 1–4 个任务，默认推理名额 `min(jobs, 2)`。`--experimental-concurrency` 可将任务数提高到 10；`--inference-slots` 显式指定 1 至 `min(jobs, 4)`。这是压力实验配置，不是日常稳定性承诺。直接 `auto` 的实验开关本身不提供主机内存中止器。

压测使用 `scripts/benchmark_extreme.py 原图目录 --records 独立过程目录 --jobs 任务数 --inference-slots 名额 --reference 已核验六面目录`：参考图片只用于输出后的字节比较，不能作为处理输入。该工具冻结代码和原图哈希，重新生成所有处理结果；严重内存压力、交换空间新增超过 8 GiB 或超过 600 秒即终止测试进程组。失败保留实际已完成数量与中止耗时，不计算成功吞吐。系统起始状态可能变化，报告必须保留失败与复测记录，不能只列最快成功一次。
