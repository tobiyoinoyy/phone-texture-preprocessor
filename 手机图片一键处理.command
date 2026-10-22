#!/bin/bash
# 双击打开窗口，或把图片/文件夹直接拖到这个文件上。
# phone_texture/ shares the complete workflow with the CLI and Skills.

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

# BiRefNet_dynamic is deliberately run at a bounded input size by the Python
# pipeline.  This allocator guard is an additional macOS safety net: MPS uses
# unified memory, so an uncapped cache can otherwise compete with the desktop.
# Keep both allocator watermarks below the default unified-memory ceiling.
# Respect an existing setting so developers can override them explicitly.
if [[ -z "${PYTORCH_MPS_HIGH_WATERMARK_RATIO+x}" ]]; then
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.7"
fi
# The Python core derives a compatible LOW value from HIGH when a caller has
# supplied a custom high watermark.  Only inject the standard pair here; this
# avoids creating LOW=0.6 > HIGH when a developer intentionally chooses e.g.
# HIGH=0.3.  An explicit LOW is always preserved.
if [[ -z "${PYTORCH_MPS_LOW_WATERMARK_RATIO:-}" && "${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-}" == "0.7" ]]; then
  export PYTORCH_MPS_LOW_WATERMARK_RATIO="0.6"
fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

say() {
  printf '%s\n' "$*"
}

pause_before_exit() {
  # Finder opens .command files in a Terminal window.  Keep that window open
  # long enough for a designer to read the result; automated/terminal calls
  # can opt out with PHONE_NORMALIZER_NO_PAUSE=1.
  if [[ -t 0 && "${PHONE_NORMALIZER_NO_PAUSE:-0}" != "1" ]]; then
    printf '\n按回车关闭此窗口…'
    read -r _ || true
  fi
}

install_dependencies() {
  local answer
  if ! command -v python3 >/dev/null 2>&1; then
    say "找不到 python3。请先安装 Python 3.9 或更高版本，再重新双击此文件。"
    return 1
  fi

  if [[ -t 0 ]]; then
    printf '首次运行需要安装 BiRefNet 依赖（约数分钟，需联网），现在安装吗？[Y/n] '
    read -r answer || answer=""
    case "$answer" in
      [Nn]*)
        say "已取消。稍后再次双击此文件即可安装。"
        return 1
        ;;
    esac
  else
    say "尚未安装 BiRefNet 依赖。请在可交互终端运行此文件，或执行："
    say "  python3 -m venv \"$VENV_DIR\""
    say "  \"$VENV_DIR/bin/python\" -m pip install -e \"$PROJECT_DIR[cutout]\""
    return 1
  fi

  if [[ ! -x "$PYTHON" ]]; then
    say "正在创建独立 Python 环境：$VENV_DIR"
    if ! python3 -m venv "$VENV_DIR"; then
      say "创建 Python 环境失败，请确认 Python 自带 venv 组件。"
      return 1
    fi
  fi
  # A few macOS Python distributions ship an old pip/setuptools pair.  Keep
  # the isolated environment current so modern wheels and the optional
  # developer package can be installed consistently on a fresh machine.
  say "正在更新安装工具（pip / setuptools / wheel）…"
  if ! PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install --upgrade pip setuptools wheel; then
    say "安装工具更新失败。请检查网络后重试；源文件和项目代码没有被修改。"
    return 1
  fi
  say "正在安装依赖，首次下载可能需要一些时间…"
  if ! PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install -e "$PROJECT_DIR[cutout]"; then
    say "依赖安装失败。请检查网络后重试；源文件和项目代码没有被修改。"
    return 1
  fi
  say "依赖安装完成。首次处理图片时还会自动下载 BiRefNet 模型权重。"
  return 0
}

dependencies_ready=0
if [[ -x "$PYTHON" ]] && "$PYTHON" -c 'import PIL, numpy, scipy, torch, torchvision, transformers, accelerate, timm, kornia, einops, cv2' >/dev/null 2>&1; then
  dependencies_ready=1
fi

if [[ "$dependencies_ready" -ne 1 ]]; then
  if ! install_dependencies; then
    pause_before_exit
    exit 1
  fi
fi

# Finder may start outside the repository. Resolve imports and installation from
# the same checkout that owns all four Skills.
cd "$PROJECT_DIR" || exit 1
say "打开六面处理窗口：几何、材质光照、本机背面对齐与验收发布…"
"$PYTHON" -m phone_texture gui "$@"
status=$?
pause_before_exit
exit "$status"
