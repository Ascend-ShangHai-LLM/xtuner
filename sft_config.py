      
import os

from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config

from xtuner.v1.train import TrainerConfig

from xtuner.v1.config import (

    AdamWConfig,

    FSDPConfig,

    LRConfig,

)

from xtuner.v1.datasets import FTDPTokenizeFnConfig

from xtuner.v1.loss.ce_loss import CELossConfig

try:

    from xtuner.v1.loss.moe_loss import ZLossConfig

except:

    from xtuner.v1.model.moe.moe import ZLossConfig

from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig

from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config

try:

    from xtuner.v1.model.moe.moe import MTPConfig

except:

    MTPConfig = None

    

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig



from xtuner.v1.float8.config import Float8Config, ScalingGranularity

# from xtuner.v1.loss.layer_moe_loss import LayerBalancingLossConfig



QWEN3_MOE_PATH = "/mnt/huawei/weight/Qwen3.5-35B-A3B"

# QWEN3_MOE_PATH = "/mnt/shared-storage-user/llmrazor-share/yehaochen/tmp/asdfasfasdf/20260320160600/hf-2"

ALPACA_PATH = "/mnt/huawei/wtg/sampled_jsonls" #"/mnt/huawei/wsl/datasets"



float8_cfg = Float8Config(

    scaling_granularity_gemm=ScalingGranularity.TILEWISE,

    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,

)





moe_cfg = Qwen3_5_VLMoE35BA3Config()

moe_cfg.text_config.ep_size = 1
moe_cfg.text_config.num_hidden_layers = 32
if MTPConfig is not None:

    moe_cfg.text_config.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=1.0)

optim_cfg = AdamWConfig(lr=6e-05, foreach=False)  

lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)

fsdp_cfg = FSDPConfig(

    torch_compile=True,

    cpu_offload=False,

    ep_size=1

)



dataset_config = [

    {

        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),

        # "tokenize_fn": FTDPTokenizeFnConfig(max_length=16*1024),

        "tokenize_fn": PretrainTokenizeFunctionConfig(),

    },

]



dataloader_config = DataloaderConfig(

    pack_max_length=27*1024,

    pack_level="hard"

)



loss_cfg = CELossConfig(mode="chunk",chunk_size=512)





trainer = TrainerConfig(

    total_step=1000,

    load_from=QWEN3_MOE_PATH,

    model_cfg=moe_cfg,

    optim_cfg=optim_cfg,

    fsdp_cfg=fsdp_cfg,

    dataset_cfg=dataset_config,

    dataloader_cfg=dataloader_config,

    lr_cfg=lr_cfg,

    loss_cfg=loss_cfg,

    tokenizer_path=QWEN3_MOE_PATH,

    global_batch_size=32,

)




    