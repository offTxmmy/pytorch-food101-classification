import time

import torch
from torch import nn

from data import create_dataloaders
from model import FoodCNNV2

BENCHMARK_BATCHES = 300
WARMUP_BATCHES = 20

CONFIGURATIONS = [
    {
        "name": "AMP standard",
        "batch_size": 32,
        "num_workers": 4,
        "use_amp": True,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking": False,
    },
    {
        "name": "AMP optimized I/O",
        "batch_size": 32,
        "num_workers": 4,
        "use_amp": True,
        "pin_memory": True,
        "persistent_workers": True,
        "non_blocking": True,
    },
]


def benchmark_configuration(
    batch_size,
    num_workers,
    use_amp,
    pin_memory,
    persistent_workers,
    non_blocking,
    device,
):
    amp_enabled = use_amp and device.type == "cuda"

    train_loader, _, _ = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = FoodCNNV2().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    iterator = iter(train_loader)

    # Warm-up
    for _ in range(WARMUP_BATCHES):
        images, labels = next(iterator)

        images = images.to(
            device,
            non_blocking=non_blocking,
        )
        labels = labels.to(
            device,
            non_blocking=non_blocking,
        )

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    processed_images = 0

    for _ in range(BENCHMARK_BATCHES):
        images, labels = next(iterator)

        images = images.to(
            device,
            non_blocking=non_blocking,
        )
        labels = labels.to(
            device,
            non_blocking=non_blocking,
        )

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        processed_images += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    images_per_second = processed_images / elapsed_time

    if device.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )
    else:
        peak_memory_gb = 0.0

    return {
        "time": elapsed_time,
        "images_per_second": images_per_second,
        "peak_memory_gb": peak_memory_gb,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Benchmark batches: {BENCHMARK_BATCHES}")
    print(f"Warm-up batches: {WARMUP_BATCHES}")
    print()

    results = []

    for config in CONFIGURATIONS:
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("=" * 60)
        print(config["name"])
        print("=" * 60)

        print(
            f"Batch size: {config['batch_size']} | "
            f"Workers: {config['num_workers']} | "
            f"AMP: {config['use_amp']}"
        )

        print(
            f"Pin memory: {config['pin_memory']} | "
            f"Persistent workers: "
            f"{config['persistent_workers']} | "
            f"Non-blocking: {config['non_blocking']}"
        )

        result = benchmark_configuration(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            use_amp=config["use_amp"],
            pin_memory=config["pin_memory"],
            persistent_workers=config["persistent_workers"],
            non_blocking=config["non_blocking"],
            device=device,
        )

        result["name"] = config["name"]
        results.append(result)

        print()

        print(
            f"{BENCHMARK_BATCHES} batches time: "
            f"{result['time']:.2f}s"
        )

        print(
            f"Throughput: "
            f"{result['images_per_second']:.1f} images/s"
        )

        print(
            f"Peak CUDA memory: "
            f"{result['peak_memory_gb']:.2f} GB"
        )

        print()

    if len(results) == 2:
        baseline = results[0]
        optimized = results[1]

        speedup = (
            optimized["images_per_second"]
            / baseline["images_per_second"]
        )

        throughput_improvement = (
            (
                optimized["images_per_second"]
                - baseline["images_per_second"]
            )
            / baseline["images_per_second"]
            * 100
        )

        time_reduction = (
            (
                baseline["time"]
                - optimized["time"]
            )
            / baseline["time"]
            * 100
        )

        print("=" * 60)
        print("AMP I/O OPTIMIZATION RESULTS")
        print("=" * 60)

        print(
            f"Standard AMP throughput: "
            f"{baseline['images_per_second']:.1f} images/s"
        )

        print(
            f"Optimized I/O throughput: "
            f"{optimized['images_per_second']:.1f} images/s"
        )

        print()

        print(f"Speedup: {speedup:.2f}x")

        print(
            f"Throughput improvement: "
            f"{throughput_improvement:.1f}%"
        )

        print(
            f"Benchmark time reduction: "
            f"{time_reduction:.1f}%"
        )

        print()

        print(
            f"Standard AMP peak memory: "
            f"{baseline['peak_memory_gb']:.2f} GB"
        )

        print(
            f"Optimized I/O peak memory: "
            f"{optimized['peak_memory_gb']:.2f} GB"
        )


if __name__ == "__main__":
    main()