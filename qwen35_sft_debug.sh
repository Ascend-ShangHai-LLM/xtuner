set -ex

ray stop --force
# 2. 清理临时目录（默认 /tmp/ray）
rm -rf /tmp/ray/*
sysctl -w net.ipv4.tcp_max_syn_backlog=65536
sysctl -w net.core.somaxconn=65536
sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015

# cd ../../mojo_opset/
# pip install -e .
cd /mnt/huawei/rl_qwen35/code/xtuner

config_file=${1}
datetime=$(date +%Y%m%d_%H%M%S)
log_dir="logs/${datetime}"

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export MULTI_STREAM_MEMORY_REUSE=2 # 多流内存复用
export XTUNER_ACTIVATION_OFFLOAD=1 # 激活值offload
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True # 虚拟内存128M
unset TORCH_HCCL_ZERO_COPY # 关闭HCCL zero copy

export TASK_QUEUE_ENABLE=2 # 算子二级流水
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
# 绑核

export XTUNER_TOKENIZE_WORKERS=1

# 兼容clusterx训练和单机训练
export XTUNER_USE_FA3=${XTUNER_USE_FA3-"1"}
export TORCH_LOGS=${TORCH_LOGS-"recompiles"}

NNODES=$WORLD_SIZE  # WORLD_SIZE是clusterx训练时注入的环境变量
if [ "x${NNODES}" == "x" ]; then
NNODES=${NODE_COUNT-"1"}  # NODE_COUNT等是仪电训练任务调度器注入的环境变量
fi
NRANK=${RANK}  # clusterx
if [ "x${NRANK}" == "x" ]; then
NRANK=${NODE_RANK-"0"}  # yidian
fi
NPROC_PER_NODE=${GPUS_PER_NODE}  # clusterx
if [ "x${NPROC_PER_NODE}" == "x" ]; then 
NPROC_PER_NODE=${PROC_PER_NODE-"8"}  # yidian
fi

# hyf
export WORLD_SIZE=${WORLD_SIZE:-$NODE_COUNT}
export RANK=${RANK:-$NODE_RANK} 


export MODEL_PATH=/mnt/huawei/weight/Qwen3.5-35B-A3B
export MEDIA_ROOT=''
# xtuner_dir="/mnt/huawei/hyf/xtuner_0403"
# export PYTHONPATH=$xtuner_dir:$PYTHONPATH

NRANK=${NRANK-"0"}
NNODES=2
NPROC_PER_NODE=16
MASTER_ADDR=${MASTER_ADDR-"127.0.0.1"}
MASTER_PORT=23452
DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --node_rank $NRANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# bash hyf_test/bind_irq.sh # 绑核
run_cmd="torchrun $DISTRIBUTED_ARGS -m xtuner.v1.train.cli.sft --config ${config_file}"

env

# 支持日志同时输出到文件，可以替代仪电页面上查看日志。这通过 log_dir 是否传入决定
if [ "x${log_dir}" == "x" ]; then
    # console日志不输出到文件
    echo "run_cmd: ${run_cmd}" 
    eval ${run_cmd}

else
    # console日志输出到文件
    mkdir -p ${log_dir}
    env | tee -a "${log_dir}/node_${NRANK}.txt"
    echo "---------------------------START--------------------------------" | tee -a "${log_dir}/node_${NRANK}.txt"
    echo "run_cmd: ${run_cmd}" | tee -a "${log_dir}/node_${NRANK}.txt"
    set -o pipefail
    eval ${run_cmd} 2>&1 | tee -a "${log_dir}/node_${NRANK}.txt"
    status=$?
    if [ $status -ne 0 ]; then exit $status; fi
    echo "---------------------------END--------------------------------" | tee -a "${log_dir}/node_${NRANK}.txt"
fi