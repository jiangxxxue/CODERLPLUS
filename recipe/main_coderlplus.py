import os
import sys
import socket
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

verl_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(verl_path))

from .ray_trainer_coderlplus import ResourcePoolManager, Role, RayPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor


@hydra.main(config_path="config", config_name="coderlplus_trainer", version_base=None)
def main(config):
    """Main entry point for execution semantics alignment GRPO training with merged validation."""
    run_coderlplus(config)


def run_coderlplus(config) -> None:
    """Initialize Ray cluster and run distributed execution semantics alignment GRPO training process."""
    
    # Check if Ray is not initialized
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.get("ray_init", {}).get("num_cpus", None),
        )

    # Create remote task runner
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available
        assert is_nvtx_available(), "nvtx is not available in CUDA platform"
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = CodeReasoningTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = CodeReasoningTaskRunner.remote()
    
    # Start training
    ray.get(runner.run.remote(config))
    
    # Optional timeline generation
    timeline_json_file = config.get("ray_init", {}).get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class CodeReasoningTaskRunner:
    """Ray remote class for executing execution semantics alignment GRPO training with merged validation."""
    
    def run(self, config):
        """Execute the main training workflow."""
        
        from pprint import pprint
        
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Download model to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Setup tokenizer and processor
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Define worker classes based on the actor strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
            
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
            
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")
        
        # Map roles to worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        # Define resource pool specification
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        
        # Add reward model worker if needed
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Add reference policy if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id
        
        # Load reward functions
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # Create datasets using RolloutEnhancedDataset if rollout integration is enabled
        if config.data.get("rollout_integration", {}).get("enable", False):
            from .rollout_enhanced_dataset import RolloutEnhancedDataset
            
            print("[DEBUG] Using RolloutEnhancedDataset for training dataset")
            train_dataset = RolloutEnhancedDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                config=config.data,
                processor=processor,
            )
            
            # Print initial rollout integration configuration
            rollout_config = config.data.rollout_integration
            print(f"[DEBUG] Rollout Integration Configuration:")
            print(f"  - Enabled: {rollout_config.get('enable', False)}")
            print(f"  - Ratio: {rollout_config.get('ratio', 0.5)}")
            print(f"  - Buffer size: {rollout_config.get('buffer_size', 20000)}")
            print(f"  - Sampling strategy: {rollout_config.get('sampling_strategy', 'recent')}")
            print(f"  - Batch size: {config.data.train_batch_size}")
            print(f"  - Target rollout samples per batch: {int(config.data.train_batch_size * rollout_config.get('ratio', 0.5))}")
        else:
            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True
            )
        
        # Validation dataset always uses standard dataset
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, is_train=False
        )
        
        # Create custom sampler for rollout integration if enabled
        if config.data.get("rollout_integration", {}).get("enable", False):
            from .rollout_batch_sampler import RolloutBatchSamplerWrapper
            import torch
            
            # Create generator for reproducibility
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(config.data.get("seed", 1))
            
            # Calculate min_reasoning_samples based on batch_size and ratio
            rollout_config = config.data.rollout_integration.copy()
            min_reasoning_samples = int(config.data.train_batch_size * rollout_config.get("ratio", 0.5))
            
            train_sampler = RolloutBatchSamplerWrapper(
                dataset=train_dataset,
                batch_size=config.data.train_batch_size,
                min_rollout_samples = min_reasoning_samples,
                rollout_config=rollout_config,
                drop_last=config.data.get("drop_last", False),
                generator=train_dataloader_generator,
            )
        else:
            train_sampler = create_rl_sampler(config.data, train_dataset)
        
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        
        trainer.init_workers()
        
        trainer.fit()


if __name__ == "__main__":
    main()
