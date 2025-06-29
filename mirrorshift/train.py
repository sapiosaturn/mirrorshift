"""
This file contains the core training logic.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from typing import Tuple
from collections import deque
from torch.utils.data import (
    DataLoader,
    Subset,
    RandomSampler,
    SequentialSampler
)
from torch.utils.tensorboard import SummaryWriter
import time

from mirrorshift.models import CausalTransformer
from mirrorshift.data import TiktokenTxtDataset
from mirrorshift.inference import sample
from mirrorshift.utils import (
    read_model_config,
    read_training_config,
    ModelConfig,
    TrainingConfig,
    get_lr_schedule,
)

BatchType = Tuple[torch.Tensor, torch.Tensor]
Logits = torch.Tensor

def sample_and_log(
    model: CausalTransformer,
    global_step: int,
    writer: SummaryWriter,
    device: str,
    subset: Subset,
    model_config: ModelConfig,
    sampling_length: int
) -> None:
    model.eval()
    random_token = torch.randint(
        low=0, high=model_config.vocab_size, size=(1, 1)
    )
    tokens, generated_text = sample(
        model=model,
        context=random_token,
        num_tokens=sampling_length,
        context_length=model_config.context_length,
        device=device,
        subset=subset
    )
    print("\n╭─ Generated Text Sample ──────────────")
    print("│")
    for line in generated_text.split('\n'):
        print(f"│ {line}")
    print("│")
    print("╰──────────────────────────────────────")
    writer.add_text("Sampled Text", generated_text, global_step)
    model.train()

def val_eval(
    model: CausalTransformer,
    writer: SummaryWriter,
    global_step: int,
    val_loader: DataLoader,
    device: str,
) -> None:
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            x, y = val_batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_value = F.cross_entropy(
                logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                y.view(y.size(0) * y.size(1)),
            )
            loss_scalar = loss_value.item()
            total_val_loss += loss_scalar
            val_batches += 1
    avg_val_loss = total_val_loss/val_batches
    writer.add_scalar("Loss/val", avg_val_loss, global_step)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
    writer.add_scalar("Perplexity/val", val_perplexity, global_step)

    print("\n╭─ Validation Metrics ─────────────────")
    print(f"│ Step:                  {global_step}")
    print(f"│ Validation Loss:       {avg_val_loss:.5f}")
    print(f"│ Validation Perplexity: {val_perplexity:.5f}")
    print("╰──────────────────────────────────────")

    model.train()

def train(
    model: CausalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    opt: optim.AdamW,
    device: str,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    writer: SummaryWriter,
) -> None:
    global_step: int = 0

    steps_per_epoch = len(train_loader.dataset) // training_config.batch_size
    total_steps = steps_per_epoch * training_config.num_epochs

    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=total_steps
    )

    step_times = deque(maxlen=100) # sliding window average
    step_start_time = time.time()

    for e in range(training_config.num_epochs):
        print("\n╭─ Starting Epoch ─────────────────────")
        print(f"│ Epoch: {e}")
        print("╰──────────────────────────────────────")

        running_loss: float = 0.0

        for i, batch in enumerate(train_loader):
            step_start_time = time.time()

            x: torch.Tensor
            y: torch.Tensor
            batch: BatchType = batch
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()

            for param_group in opt.param_groups:
                param_group['lr'] = lr_schedule(global_step)

            writer.add_scalar("Learning Rate", opt.param_groups[0]['lr'], global_step)

            logits: Logits = model(x)
            loss_value = F.cross_entropy(
                logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                y.view(y.size(0) * y.size(1)),
            )

            loss_value.backward()
            loss_scalar = loss_value.item()
            opt.step()
            global_step += 1

            step_time = time.time() - step_start_time
            step_times.append(step_time)

            avg_step_time = sum(step_times) / len(step_times)
            steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0

            writer.add_scalar("Loss/train", loss_scalar, global_step)
            writer.add_scalar(
                "Perplexity/train",
                torch.exp(torch.tensor(loss_scalar)).item(),
                global_step
            )
            writer.add_scalar("Performance/seconds_per_step", avg_step_time, global_step)
            writer.add_scalar("Performance/steps_per_second", steps_per_second, global_step)
            running_loss += loss_scalar

            if global_step % training_config.reporting_steps == 0:
                last_loss = running_loss / training_config.reporting_steps
                last_perplexity = torch.exp(torch.tensor(last_loss)).item()
                print("\n╭─ Training Progress ────────────────────────")
                print(f"│ Batch Size:              {training_config.batch_size}")
                print(f"│ Step:                    {global_step}")
                print(f"│ Loss:                    {last_loss:.5f}")
                print(f"│ Perplexity:              {last_perplexity:.5f}")
                print(f"│ Seconds/Step (per GPU):  {avg_step_time:.3f}")
                print(f"│ Steps/Second (per GPU):  {steps_per_second:.3f}")
                print("╰────────────────────────────────────────────")
                running_loss = 0.0

            if global_step % training_config.validation_eval_steps == 0:
                step_times.clear()
                val_eval(
                    model=model,
                    writer=writer,
                    global_step=global_step,
                    val_loader=val_loader,
                    device=device
                )

            if global_step % training_config.sampling_steps == 0:
                step_times.clear()
                sample_and_log(
                    model=model,
                    global_step=global_step,
                    writer=writer,
                    device=device,
                    subset=train_loader.dataset,
                    model_config=model_config,
                    sampling_length=training_config.sampling_length_multiplier * model_config.context_length
                )

def main():
    """Entry point for the mirrorshift-train command."""
    # default values are for tiny model
    parser = argparse.ArgumentParser(description="Train a mirrorshift transformer model")
    parser.add_argument('--model-config', type=str, default='mirrorshift/config/model_configs/small.json',
                      help='Path to model configuration file')
    parser.add_argument('--training-config', type=str, default='mirrorshift/config/training_configs/small.json',
                      help='Path to training configuration file')
    parser.add_argument('--dataset', type=str, default='mirrorshift/datasets/coqa_stories.txt',
                      help='Path to training text file')
    args = parser.parse_args()

    model_config: ModelConfig = read_model_config(args.model_config)
    training_config: TrainingConfig = read_training_config(args.training_config)
    
    # Continue with the rest of the training process
    writer: SummaryWriter = SummaryWriter()

    full_dataset: TiktokenTxtDataset = TiktokenTxtDataset(
        args.dataset, model_config.context_length
    )

    # this is mainly useful with char-level tokenizers
    assert full_dataset.get_vocab_size() == model_config.vocab_size, \
    f"dataset vocab size is {full_dataset.get_vocab_size()}, model_config vocab size is {model_config.vocab_size}"

    train_size = int(len(full_dataset) * (1-training_config.validation_split))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator()
    )

    model = CausalTransformer(model_config=model_config)

    device: str
    if training_config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = training_config.device

    model = model.to(device)

    opt: optim.AdamW = optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    if training_config.compile:
        if device == "mps":
            print("INFO: torch.compile not compatible with device mps, choosing not to compile")
        else:
            model = torch.compile(model)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=train_sampler
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        sampler=val_sampler
    )

    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n╭─ Training Info ──────────────────────")
    print(f"│ Total dataset size:   {len(full_dataset)}")
    print(f"│ Trainable parameters: {trainable_params}")
    print(f"│ Training set size:    {len(train_dataset)}")
    print(f"│ Training on device:   {device}")
    print(f"│ Validation set size:  {len(val_dataset)}")
    print(  "╰──────────────────────────────────────")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        opt=opt,
        device=device,
        training_config=training_config,
        model_config=model_config,
        writer=writer,
    )

    writer.flush()
    
    return 0

if __name__ == '__main__':
    main()