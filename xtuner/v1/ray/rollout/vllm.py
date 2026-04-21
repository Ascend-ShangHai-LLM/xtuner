import asyncio
import os
import traceback
from argparse import Namespace
from typing import Any, Dict, List, Union

import ray
import requests
import torch
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, RolloutState
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.ray.rollout.worker import RolloutWorker
from xtuner.v1.utils.device import get_device, get_torch_device_module
import numpy as np
import time
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from xtuner.v1.ray.base import AutoAcceleratorWorkers

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class WorkerWrap:

    def update_weight_npu_ipc(self, data):
        import base64
        import json
        from multiprocessing.reduction import ForkingPickler

        if isinstance(data, str):
            data = json.loads(data)

        def _construct(item):
            func, args = item
            args = list(args)
            args[6] = DEVICE_MODULE.current_device()
            return func(*args)

        serialized_data = data["serialized_named_tensors"]
        finished = data["finished"]
        if isinstance(serialized_data, list):
            serialized_data = serialized_data[self.global_rank]
        weights = ForkingPickler.loads(base64.b64decode(serialized_data))
        weights = [(k, _construct(v)) for k, v in weights]
        DEVICE_MODULE.synchronize()
        self.model_runner.model.load_weights(weights=weights)
        del weights
        
        if finished:
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
        DEVICE_MODULE.synchronize()
        DEVICE_MODULE.empty_cache()

    def get_worker_pids(self):
        current_pid = os.getpid()
        return current_pid


@ray.remote
class VllmServerWrapper:
    def __init__(self, server_namespace: Namespace):
        cli_env_setup()
        server_args = getattr(server_namespace, "args", Namespace())
        env = getattr(server_namespace, "env", {})
        for k, v in env.items():
            os.environ[k] = str(v)
        for k in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "NODES_PER_LOGIC_NODE", "HCCL_LOGIC_SUPERPOD_ID"]:
            if k in os.environ:
                del os.environ[k]
        try:
            asyncio.run(run_server(server_args))
        except Exception as e:
            error_msg = f"Failed to start server in VllmServerWrapper: {type(e).__name__}: {str(e)}"
            stack_trace = traceback.format_exc()
            print(error_msg)
            print(stack_trace)
            raise  # Re-raise the exception to prevent silent failure

    def actor_health(self):
        return "healthy"


# Add a dummy task.
#def run_vllm_server_wrapper(server_namespace: Namespace):
#    return ray.get(VllmServerWrapper.remote(server_namespace).actor_health.remote())  # type: ignore


class vLLMWorker(RolloutWorker):
    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        self.router_func = ""
        #self.server_func = run_vllm_server_wrapper
        self.endpoints["health_generate"] = "health"
        self.endpoints["v1/chat/completions"] = "v1/chat/completions"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wake_up"
        self.endpoints["models"] = "models"
        self.endpoints["update_weights"] = "update_weights"
        # self.endpoints['abort_request'] = "abort_request"
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name
        self.enable_return_routed_experts = self.config.enable_return_routed_experts
        self.dp_size = self.config.data_parallel_size
        assert self.dp_size > 0, "data_parallel_size must be > 0"
        assert self.config.tensor_parallel_size % self.dp_size == 0, (
            f"tensor_parallel_size ({self.config.tensor_parallel_size}) must be divisible by data_parallel_size ({self.dp_size})"
        )
        self.tp_size = self.config.tensor_parallel_size // self.dp_size

    def launch_server(self):
        """Launch the inference server as a separate process or Ray task.

        It waits for the server to become healthy before returning.

        Raises:
            TimeoutError: If the server fails to start within the specified
                timeout.
            Exception: If the server task terminates unexpectedly.
        """
        server_configs = self._transform_rollout_config_to_server_configs()
        timeout = 3600.0
        start_time = time.perf_counter()
        last_log_time = start_time
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {server_configs.api_key}",
        }

        self.logger.info(f"Launch server task on server_url: {self.server_url}")

        # launch the server as ray task
        # so that the lmdeploy backend could get externl pg
        current_pg = ray.util.get_current_placement_group()
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=current_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=self.engine_bundle_idxs[0],
        )
        assert ray.is_initialized()
        ray_kwargs = (
            {"runtime_env": server_configs.ray_runtime_env} if hasattr(server_configs, "ray_runtime_env") else {}
        )
        self.server_task = (
            VllmServerWrapper.options(
                scheduling_strategy=scheduling_strategy,
                **AutoAcceleratorWorkers.get_pg_options(current_pg),
                **ray_kwargs,
            )
            .remote(server_configs)
        )

        with requests.Session() as session:
            while time.perf_counter() - start_time < timeout:
                try:
                    response = session.get(
                        f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                    )
                    if response.status_code == 200:
                        return
                except requests.RequestException:
                    pass

                try:
                    ray.get(self.server_task.actor_health.remote(), timeout=0.1)
                    raise Exception("Server task terminated unexpectedly.")
                except ray.exceptions.GetTimeoutError:
                    pass
                except Exception as e:
                    raise e

                current_time = time.perf_counter()
                if current_time - last_log_time >= 15:
                    self.logger.info(
                        f"Waiting for server to start... Elapsed time: {current_time - start_time:.2f}s"
                    )
                    last_log_time = current_time

            ray.cancel(self.server_task)
    
    async def _create_request(
        self,
        url: str,
        prompt: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,  # reserved for agent tool use
        tool_choice: str,  # reserved for agent tool use
        sample_params: dict,
        extra_params: dict,
        extra_info: dict,
    ):
        if len(tools) > 0:
            raise NotImplementedError

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",
        }

        if "image_data" in extra_info:
            if not isinstance(prompt, list):
                raise ValueError("image_data requires prompt to be a list of messages")

            image_index = 0
            for message in prompt:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "user":
                    new_content = []
                    for content_part in message.get("content", []):
                        if not isinstance(content_part, dict):
                            new_content.append(content_part)
                            continue
                        if content_part.get("type") == "image_url":
                            content_part["image_url"]["url"] = f"file://{extra_info['image_data'][image_index]}"
                            content_part["image_url"].pop("image_wh", None)
                            image_index += 1
                            new_content.append(content_part)
                        else:
                            new_content.append(content_part)

                    message["content"] = new_content

            assert image_index == len(extra_info["image_data"]), (
                f"Expected {len(extra_info['image_data'])} images, but processed {image_index}."
            )

        
        payload = {
            "model": self.config.model_path,
            "messages": prompt,
            #"tools": tools,
            #"tool_choice": tool_choice if tool_choice else 'none',
        }

        if input_ids is not None:
            payload["input_ids"] = input_ids
        if "partial_rollout_input_ids" in extra_info:
            payload["input_ids"] = extra_info["partial_rollout_input_ids"]
            assert len(payload["input_ids"]) <= self.config.context_length, (
                f"Total input length {len(payload['input_ids'])} exceeds context length {self.config.context_length}."
            )
        
        vllm_sample_params = self._transform_sample_params(sample_params, extra_params)
        payload.update(vllm_sample_params)
        return await self._safe_post_request(url, headers, payload)

    def _transform_sample_params(self, sample_params: Dict, extra_params: Dict = {}):
        import copy

        vllm_sample_params = copy.deepcopy(sample_params)
        if extra_params:
            vllm_sample_params.update(extra_params)
        if "stops" in vllm_sample_params:
            vllm_sample_params["stop"] = vllm_sample_params.pop("stops")
        if "no_stop_trim" in vllm_sample_params:
            vllm_sample_params["include_stop_str_in_output"] = vllm_sample_params.pop("no_stop_trim")
        if "top_logprobs" in vllm_sample_params and "return_logprob" in vllm_sample_params:
            vllm_sample_params["logprobs"] = vllm_sample_params.pop("return_logprob")
        return vllm_sample_params

    def get_logprobs(self, input_ids, sampling_params):
        pass

    def generate(self, input_ids, sampling_params):
        pass

    def sleep(self, level=1):
        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json"}
        params = {}
        params["level"] = level
        response = requests.post(url, headers=headers, params=params)
        assert response.status_code == 200, response.status_code
        return response.text

    def wake_up(self, tags: List[str] | None = None):
        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json"}
        params = {}
        if tags is not None:
            params["tags"] = tags
        response = requests.post(url, headers=headers, params=params)
        assert response.status_code == 200, response.status_code
        return response.text

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def onload_weights(self):
        """Onloads the model weights by waking up the model."""
        return self.wake_up(tags=["weights"])

    def onload_kvcache(self):
        """Onloads the KV cache by waking up the model."""
        return self.wake_up(tags=["kv_cache"])

    def offload(self):
        """Offloads the model weights and KV cache."""
        return self.sleep(level=2)

    def reset_prefix_cache(self, tags: List[str] | None = None):
        raise NotImplementedError("The 'reset_prefix_cache' API is not yet implemented in the vLLM server.")

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        # use vllm FlexibleArgumentParser to parse the config
        # and return the args as the default server config
        # vllm server_args: vllm/vllm/engine/arg_utils.py
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args_ = parser.parse_args([])

        args = {}
        args["host"] = self.host
        args["port"] = self.server_port
        args["api_key"] = self.api_keys
        args["api_keys"] = self.api_keys
        args["model"] = self.config.model_path
        args["log_level"] = "info"
        args["data_parallel_size"] = self.dp_size
        args["tensor_parallel_size"] = self.tp_size
        args["enable_expert_parallel"] = False

        args["distributed_executor_backend"] = "ray"
        args["max_model_len"] = self.config.context_length
        args["enforce_eager"] = False
        args["enable_sleep_mode"] = True
        args["worker_extension_cls"] = "xtuner.v1.ray.rollout.vllm.WorkerWrap"
        args["trust_remote_code"] = True
        args["enable_prefix_caching"] = False
        args["allowed_local_media_path"] = "/"
        args["mm_processor_cache_gb"] = 0
        args["max_num_batched_tokens"] = 4096
        args["max_num_seqs"] = self.config.rollout_max_batch_size_per_instance // self.dp_size
        args["block_size"] = 128
        args["gpu_memory_utilization"] = self.config.gpu_memory_utilization
        args["compilation_config"] = {
            "cudagraph_capture_sizes": [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],
            #"cudagraph_capture_sizes": [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
        args["additional_config"] = {"enable_cpu_binding": True}
        args["limit_mm_per_prompt"] = {"image": 20, "video": 5}
        args["enable_log_requests"] = False
        args["uvicorn_log_level"] = "error"
        args["enable_return_routed_experts"] = True
        args["api_server_count"] = 1
        #args["enable_auto_tool_choice"] = True
        #args["tool_call_parser"] = "qwen3_coder"
        
        env = {
            "VLLM_WORKER_MULTIPROC_METHOD":"spawn",
            "VLLM_VERSION": "0.18.0",
            "TASK_QUEUE_ENABLE": "0",
            "CPU_AFFINITY_CONF": "2",
            "VLLM_USE_V1": "1",
            "VLLM_RAY_PER_WORKER_GPUS": "0.1",
            "VLLM_RAY_BUNDLE_INDICES": ",".join(map(str, self.engine_bundle_idxs)),
            #"VLLM_MONITOR": "1",
            #"VLLM_ACCU_MONITOR": "0",
            #"CUSTOM_SCHEDULE_KV_LIMIT": "0.9",
            "HCCL_BUFFSIZE": "512",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "SHM_BARRIER": "true",
            "USE_TOKEN_IN": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
            #"HCCL_OP_EXPANSION_MODE": "AIV",
            #"INTERNS1_VIT_USE_TP": "1",
            #"VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
            "NCCL_URL": self.nccl_url,
        }

        # Apply extra_rollout_config overrides for vLLM parameters (prefix: "vllm_")
        extra_cfg = getattr(self.config, "extra_rollout_config", None) or {}
        for key, value in extra_cfg.items():
            if key.startswith("vllm_"):
                real_key = key[5:]
                args[real_key] = value

        args_.__dict__.update(args)
        validate_parsed_serve_args(args_)

        return Namespace(
            args=args_,
            env=env,
            api_key=self.api_keys,
            api_keys=self.api_keys,
            ray_runtime_env={"env_vars": env},
        )

    async def _handle_stream_response(self, uid, sample_params, extra_params, response) -> RLRolloutResponseItem:
        raise NotImplementedError

    async def _handle_non_stream_response(
        self, root_id, action_id, sample_params, extra_params, response, input_extra_info
    ) -> RLRolloutResponseItem:
        uid = action_id
        response = response.json()
            
        last_logprobs = []
        extra_info = {}
        
        finish_reason = response["choices"][0]["finish_reason"]
        if finish_reason == "abort" and self.receive_abort_request.is_set() is False:
            self.receive_abort_request.set()
            self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")
        
        last_token_ids = response["choices"][0]["token_ids"]
        if len(response["choices"][0]["logprobs"]["content"]) > 0:
            last_trajectory = "".join([item["token"] for item in response["choices"][0]["logprobs"]["content"]])
            last_logprobs = [item["logprob"] for item in response["choices"][0]["logprobs"]["content"]]
            assert len(last_token_ids) == len(last_logprobs)
            assert len(last_token_ids) <= sample_params["max_tokens"], (
                f"Generation length exceeds limit: generated {len(last_token_ids)}, limit {sample_params['max_tokens']}"
            )
        else:
            last_trajectory = response["choices"][0]["message"]["content"]
        
        prompt_tokens = response["usage"]["prompt_tokens"]
        response_tokens = response["usage"]["completion_tokens"]
        if self.enable_return_routed_experts and not extra_params.get("disable_routed_experts", False):
            assert "routed_experts" in response["choices"][0], (
                "enable_return_routed_experts is True, but routed_experts is not in meta_info"
            )
            exist_history_routed_experts = (
                "routed_experts" in input_extra_info and input_extra_info["routed_experts"] is not None
            )
            routed_experts = response["choices"][0]["routed_experts"]  # token[layer[expert]]#
            if routed_experts is not None:
                assert len(routed_experts) == prompt_tokens + response_tokens - 1
            if routed_experts is not None and not exist_history_routed_experts:
                routed_experts = torch.tensor(routed_experts)  # n,layer,expert
                extra_info["routed_experts"] = ray.put(routed_experts)
            elif routed_experts is not None and exist_history_routed_experts:
                routed_experts = torch.tensor(routed_experts)  # n,layer,expert
                cur_routed_experts = routed_experts

                history_routed_experts = await input_extra_info["routed_experts"]  # n, layer, expert
                ray.internal.free(input_extra_info["routed_experts"], local_only=False)
                del input_extra_info["routed_experts"]

                assert (history_routed_experts.shape[0] - 1) > 0 and history_routed_experts.shape[
                    0
                ] - 1 <= cur_routed_experts.shape[0], (
                    f"Existing routed_experts shape: {history_routed_experts.shape}, current routed_experts shape: {cur_routed_experts.shape}"
                )
                init_cur_roued_experts = cur_routed_experts.shape[0]
                cur_routed_experts = cur_routed_experts[history_routed_experts.shape[0] :, :, :]
                concat_routed_experts = np.concatenate((history_routed_experts, cur_routed_experts), axis=0)
                assert concat_routed_experts.shape[0] == prompt_tokens + response_tokens - 1, (
                    f"Routed experts shape {concat_routed_experts.shape[0]} does not match total tokens {prompt_tokens + response_tokens - 1}"
                )
                self.logger.debug(
                    f"[{root_id}/{action_id}] Partial Rollout Stats: "
                    f"Tokens(prompt={prompt_tokens}, response={response_tokens}, total={prompt_tokens + response_tokens}) | "
                    f"Experts(exist={history_routed_experts.shape}, init_cur={init_cur_roued_experts}, cur={cur_routed_experts.shape}, concat={concat_routed_experts.shape})"
                )
                extra_info["routed_experts"] = ray.put(concat_routed_experts)
                del history_routed_experts
                del cur_routed_experts
            else:
                assert finish_reason == "abort", (
                    f"routed_experts is None, but finish_reason is {finish_reason}, expected abort. response: {response}"
                )

        if finish_reason != "abort" and (len(last_token_ids) == 0 or len(last_logprobs) == 0):
            self.logger.error(f"Invalid rollout response for request {uid}: finish_reason: {finish_reason} len(last_token_ids) {len(last_token_ids)} len(last_logprobs) {len(last_logprobs)} {response}")
            return RLRolloutResponseItem(state=RolloutState.SKIPPED)
        else:
            rollout_response = RLRolloutResponseItem(
                response=last_trajectory,
                response_ids=last_token_ids,
                num_return_tokens=len(last_token_ids),
                finish_reason=finish_reason,
                logprobs=last_logprobs,
                extra_info=extra_info,
                state=RolloutState.ABORTED if finish_reason == "abort" else RolloutState.COMPLETED,
            )

            return rollout_response
