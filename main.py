import torch
import argparse
import time
from thop import profile

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()

def setup():
    torch.distributed.init_process_group("nccl", init_method="env://")

def cleanup():
    torch.distributed.destroy_process_group()

def get_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2048, 2048)
        self.out = torch.nn.Linear(2048, 1)

    def forward(self, x):
        x = self.block(x)
        x = self.out(x)
        return x

def effective_flops(macs: float, elapsed: float, batch_size: int) -> float:
    return (macs * 2 * 3 * batch_size) / elapsed

def training_test(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, steps: int) -> float:
    def step(data):
        optimizer.zero_grad()
        f_start = time.process_time()
        output = model(data)
        f_end = time.process_time()
        loss = output.sum()
        loss.backward()
        optimizer.step()
        b_end = time.process_time()
        torch.distributed.all_reduce(loss)
        r_end = time.process_time()

        # f_start - start of forward pass
        # f_end - end of forward pass
        # b_end - end of backward pass
        # r_end - end of reduction

        return r_end - f_start, f_end - f_start, b_end - f_end, r_end - b_end
    step(data[0]) # run first step to allocate memory for more accurate timing
    times = []
    latencies = []
    for i in range(steps):
        step_latency, ff_latency, bp_latency, red_latency = step(data[i])
        times.append(step_latency)
        latencies.append((ff_latency, bp_latency, red_latency))
    latencies = [sum(x) / len(x) for x in zip(*latencies)]
    return sum(times) / steps, latencies

def inference_test(model: torch.nn.Module, data: torch.Tensor, steps: int) -> float:
    def step(data):
        with torch.no_grad():
            f_start = time.process_time()
            output = model(data)
            return time.process_time() - f_start
    step(data[0]) # run first step to allocate memory for more accurate timing
    times = []
    for i in range(steps):
        times.append(step(data[i]))
    return sum(times) / steps

def device_printout():
    print('\n--Device Info--')
    world_size = get_world_size()
    devs = []
    for i in range(world_size):
        devs.append(torch.cuda.get_device_properties(i))
    for i in range(world_size):
        print(f"Rank {i}: {devs[i].name}; {devs[i].total_memory/1e9} GB")

def main():
    rank = get_rank()
    torch.cuda.set_device(rank)
    dummy_model = DummyModel()
    dummy_input = torch.randn(1, 2048, 2048)
    macs, params = profile(dummy_model, inputs=(dummy_input, ))
    if rank == 0:
        print(f"macs: {macs}, Params: {params}")

    steps = args.steps
    batch_size = args.batch_size
    
    # create a "steps" amount of dummy data with batch size 8
    inputs = [torch.randn(batch_size, 2048, 2048).cuda() for _ in range(steps)]

    training_time, latencies = training_test(
        model=dummy_model.to('cuda'),
        optimizer=torch.optim.SGD(dummy_model.parameters(), lr=0.001, momentum=0.9),
        data=inputs,
        steps=steps,
    )

    inference_time = inference_test(
        model=dummy_model.to('cuda'),
        data=inputs,
        steps=steps,
    )

    if rank == 0:
        device_printout()
        print("\n--Training Latencies--")
        print(f"fwd latency: {latencies[0]}")
        #print(f"fwd flops: {}")
        print(f"bwd latency: {latencies[1]}")
        print(f"red latency: {latencies[2]}")
        print(f"step: {training_time}\n")
        print("--Training FLOPS Metrics--")
        print(f"Effective Global TFLOPS: {(effective_flops(macs, training_time, batch_size)*get_world_size())/1e12}")
        print(f"Effective Rank TFLOPS: {(effective_flops(macs, training_time, batch_size))/1e12}\n")
        print("--Inference FLOPS Metrics--")
        print(f"Effective Rank TFLOPS: {(effective_flops(macs, inference_time, batch_size)/3)/1e12}\n")

if __name__ == "__main__":
    setup()
    main()
    cleanup()