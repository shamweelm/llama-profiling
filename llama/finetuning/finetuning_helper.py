from datetime import datetime
import json
import os
import dataclasses
from pathlib import Path
import time
import fire
import random
from llama.tokenizer import Tokenizer
import torch
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
import autonvtx
from llama.configs import fsdp_config as FSDP_CONFIG
from llama.configs import train_config as TRAIN_CONFIG
from llama.data.concatenator import ConcatDataset
from llama.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama.utils import fsdp_auto_wrap_policy
from llama.utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama.utils.dataset_utils import get_preprocessed_dataset
from llama.utils.fsdp_utils import hsdp_device_mesh
from llama.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama.model import ModelArgs, Transformer, TransformerBlock
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def setup_model_parallel_group():
    if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        if world_size > 1:
            initialize_model_parallel(world_size, rank)
        else:
            print("Single process detected, skipping model parallel group initialization.")
    else:
        print("Distributed environment variables not set, skipping model parallel group initialization.")
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)


def load_model_and_tokenizer(
    train_config
):    
    start_time = time.time()
    checkpoints = sorted(Path(train_config.ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"No checkpoint files found in {train_config.ckpt_dir}"

    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    with open(Path(train_config.ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=train_config.max_seq_len,
        max_batch_size=train_config.max_batch_size,
        **params,
    )
    
    print("Starting the tokenizer setup at : ", datetime.now())
    torch.cuda.nvtx.range_push("Tokenizer Setup")
    tokenizer = Tokenizer(model_path=train_config.tokenizer_path)
    # Set the pad_id to eos_id for packing batching strategy
    tokenizer.pad_id = tokenizer.eos_id

    print("Tokenizer setup complete at : ", datetime.now())
    torch.cuda.nvtx.range_pop()
    
    model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model.load_state_dict(checkpoint, strict=False)
    model = autonvtx(model)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
        
    return model, tokenizer


def main(**kwargs):
    print("Starting the training process at : ", datetime.now())
    torch.cuda.nvtx.range_push("Setup")
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    if not train_config.enable_fsdp:
        setup_model_parallel_group()

    clear_gpu_cache(0)
    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    torch.cuda.nvtx.range_pop()
    print("Configuration setup complete at : ", datetime.now())

    print("Starting the model setup at : ", datetime.now())
    torch.cuda.nvtx.range_push("Model Setup")

    model, tokenizer = load_model_and_tokenizer(
        train_config
    )
    
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        if rank == 0:
            model = model.cuda()
        else:
            model = model.to(torch.device("meta"))
    else:
        model = model.cuda()

    torch.cuda.nvtx.range_pop()
    print("Model setup complete at : ", datetime.now())

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    print("Starting the model preparation at : ", datetime.now())
    torch.cuda.nvtx.range_push("Model Preparation")

    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    hsdp_device_mesh_plan = None
    if (
        fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, TransformerBlock)

        device_id = 0
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if torch.cuda.is_available():
            model.to("cuda")
    
    torch.cuda.nvtx.range_pop()
    print("Model preparation complete at : ", datetime.now())

    print("Starting the data preparation at : ", datetime.now())
    torch.cuda.nvtx.range_push("Data Preparation")
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        if len(eval_dataloader) == 0:
            raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    torch.cuda.nvtx.range_pop()
    print("Data preparation complete at : ", datetime.now())

    print("Starting the optimizer and scheduler setup at : ", datetime.now())
    torch.cuda.nvtx.range_push("Optimizer and Scheduler Setup")
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    torch.cuda.nvtx.range_pop()
    print("Optimizer and Scheduler setup complete at : ", datetime.now())

    print("Starting the training process at : ", datetime.now())
    torch.cuda.nvtx.range_push("Training")
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
    )
    torch.cuda.nvtx.range_pop()
    print("Training process complete at : ", datetime.now())
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
