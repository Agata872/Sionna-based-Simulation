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
