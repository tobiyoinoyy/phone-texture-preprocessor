# 标准化兼容接口

本项目只使用 BiRefNet_dynamic。

- [统一入口、安装和测试](../README.md)
- Python 接口保留 `phone_normalizer.api.Normalizer`、`NormalizerConfig` 和 `process_many`。
- [六面精裁与数值 UV 基准](uv-profile.md)：保留 `phone-normalizer-uv`，内置历史机型参数不能跨机型套用。
- `手机图片一键处理.command` 为原标准化图形入口，保留兼容；完整三阶段工作流使用 `phone-texture`。

发布 Python 安装包可运行 `python -m pip wheel --no-deps . -w dist`；Codex 插件使用仓库根目录的 `.codex-plugin/plugin.json`。
