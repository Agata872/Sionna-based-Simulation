# 激活 Sionna 仿真环境（Python 虚拟环境 .venv）
# 用法：source env.sh   （可在任意目录下执行，路径含空格时请加引号）
#   例如：source "/home/user/Sionna-based Simulation/env.sh"
# 退出环境：deactivate

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 'source env.sh'，而不是直接执行" >&2
    exit 1
fi

SIONNA_SIM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIONNA_SIM_HOME
source "$SIONNA_SIM_HOME/.venv/bin/activate"

# GPU 射线追踪 Sionna RT 需要额外的 OptiX 库，通过 LD_LIBRARY_PATH 生效
# 只影响本 shell 启动的程序；VS Code 的 Jupyter 内核通过项目根目录的 .env 设置同一变量
# 两种情况下 libcuda 都用系统提供的，目录中只有 OptiX 相关的库
if [[ -d /usr/lib/wsl/lib ]]; then
    # WSL：Windows 驱动不提供可用的 OptiX，补上从 Linux 驱动中提取的 OptiX 库，见 README 注意事项 5
    NVIDIA_USERSPACE_LIB="$HOME/.local/nvidia/wsl-optix-615.78.08"
else
    # 服务器：系统驱动 595.91.07 只装了计算库，补上同版本的 OptiX 库，见 README 注意事项 2
    NVIDIA_USERSPACE_LIB="$HOME/.local/nvidia/optix-595.91.07"
fi
NVIDIA_USERSPACE_PROBE="libnvoptix.so.1"
if [[ ! -e "$NVIDIA_USERSPACE_LIB/$NVIDIA_USERSPACE_PROBE" ]]; then
    echo "警告：未找到 $NVIDIA_USERSPACE_LIB，Sionna RT 将无法使用 GPU（安装方法见 README）" >&2
elif [[ ":$LD_LIBRARY_PATH:" != *":$NVIDIA_USERSPACE_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_USERSPACE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
