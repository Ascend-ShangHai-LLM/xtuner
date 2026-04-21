      
config_file=${1}
datetime=$(date +%Y%m%d_%H%M%S)
log_dir="logs/${datetime}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh


export MULTI_STREAM_MEMORY_REUSE=2 # 多流内存复用
export XTUNER_ACTIVATION_OFFLOAD=1 # 激活值offload
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:128 # 虚拟内存128M

export TASK_QUEUE_ENABLE=2 # 算子二级流水
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
# 绑核
export CPU_AFFINITY_FORCE=True
export CPU_AFFINITY_CONF=1,npu0:12-23,npu1:26-37,npu2:52-63,npu3:66-77,npu4:92-103,npu5:106-117,npu6:132-143,npu7:146-157,npu8:172-183,npu9:186-197,npu10:212-223,npu11:226-237,npu12:252-263,npu13:266-277,npu14:292-303,npu15:306-317
export TRITON_ALWAYS_COMPILE=1

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

export TRITON_ALL_BLOCKS_PARALLEL=1 # triton算子需要
export WORLD_SIZE=${WORLD_SIZE:-$NODE_COUNT}
export RANK=${RANK:-$NODE_RANK} 


export MEDIA_ROOT=''

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

    
