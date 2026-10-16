import timeit
from typing import Any

import numpy as np
import torch
from numpy import memmap
from torch import nn, optim
from torch._tensor import Tensor
from tqdm import tqdm

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import data_load
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.train import evaluate_fn
from cs336_basics.transformer import transformer_lm


def single_train(
    model: nn.Module,
    hyper_params: dict,
    optimizer: optim.Optimizer,
    data_array: np.memmap,
    global_train_times: int,
    sync: bool = False,
):

    model.train()

    batch_size = hyper_params["batch_size"]
    train_context_length = hyper_params["train_context_length"]
    train_device = hyper_params["device"]
    max_lr = hyper_params["max_lr"]
    min_lr = hyper_params["min_lr"]
    t_w = hyper_params["t_w"]
    t_c = hyper_params["t_c"]
    l2_norm_max = hyper_params["l2_norm_max"]
    # 声明我们所需要用到的超参数

    x, y = data_load(data_array, batch_size, train_context_length, train_device)
    ad_lr = get_lr_cosine_schedule(global_train_times, max_lr, min_lr, t_w, t_c)

    for group in optimizer.param_groups:
        group["lr"] = ad_lr
    # 优化器内部参数储存在param_groups
    # 我们这里使用了正弦调整学习率，因此需要更新

    if sync == True:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_start = timeit.default_timer()

        logistic = model(x)

        forward_end = timeit.default_timer()

        forward_time = forward_end - forward_start

        loss = cross_entropy(logistic, y)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        backward_start = timeit.default_timer()

        loss.backward()

        backward_end = timeit.default_timer()

        backward_time = backward_end - backward_start

        gradient_clipping(model.parameters(), l2_norm_max)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        opt_start = timeit.default_timer()

        optimizer.step()

        opt_end = timeit.default_timer()

        opt_time = opt_end - opt_start

        forward_backward_time = backward_end - forward_start

        forward_backward_opt_time = opt_end - forward_start

        return (
            loss,
            forward_time,
            backward_time,
            opt_time,
            forward_backward_time,
            forward_backward_opt_time,
        )
    else:
        optimizer.zero_grad()
        logistic = model(x)
        loss = cross_entropy(logistic, y)
        loss.backward()
        gradient_clipping(model.parameters(), l2_norm_max)
        optimizer.step()
        return loss

    # 返回交叉熵损失，当loss降低时储存下来
    #


def analysis_time(
    train_data_load_path: str,
    vaild_data_load_path: str,
    total_time: int,
    warm_up_time: int,
    hyper_params: dict,
) -> None:

    forward_time_ = []
    backward_time_ = []
    opt_time_ = []
    forward_backward_time_ = []
    forward_backward_opt_time_ = []

    assert total_time > warm_up_time, ValueError("Total times must be BIGGER than warm_up_time")
    assert train_data_load_path is not None, ValueError("Path does not exit")
    assert vaild_data_load_path is not None, ValueError("Path does not exit")

    model: transformer_lm = transformer_lm(
        hyper_params["vocab_size"],
        hyper_params["context_length"],
        hyper_params["num_layers"],
        hyper_params["d_model"],
        hyper_params["num_heads"],
        hyper_params["d_ff"],
        hyper_params["theta"],
        hyper_params["device"],
    )

    opt: AdamW = AdamW(
        model.parameters(),
        hyper_params["max_lr"],
        hyper_params["weight_decay"],
        hyper_params["betas"],
        hyper_params["eps"],
    )

    train_data_list: memmap = np.memmap(train_data_load_path, dtype=np.uint32, mode="r")
    vaild_data_list: memmap = np.memmap(vaild_data_load_path, dtype=np.uint32, mode="r")

    for t in tqdm(range(total_time)):
        if warm_up_time < t <= total_time:
            (
                train_loss,
                forward_time,
                backward_time,
                opt_time,
                forward_backward_time,
                forward_backward_opt_time,
            ) = single_train(model, hyper_params, opt, train_data_list, t, True)

            forward_time_.append(forward_time)
            backward_time_.append(backward_time)
            opt_time_.append(opt_time)
            forward_backward_time_.append(forward_backward_opt_time)
            forward_backward_opt_time_.append(forward_backward_opt_time)

            print(
                f"forward time is {forward_time}\n backward time is {backward_time}\n opt time is {opt_time}\n forward_backward_time is {forward_backward_time}\n forward_backward_opt_time is {forward_backward_opt_time}"
            )

        else:
            train_loss = single_train(model, hyper_params, opt, train_data_list, t, False)

        with torch.no_grad():
            acc: Tensor = evaluate_fn(model, vaild_data_list, hyper_params)

            print(f"epoch is {t}, and the loss is {train_loss}, the acc is {acc}")

    forward_time_ = np.array(forward_time_)
    backward_time_ = np.array(backward_time_)
    opt_time_ = np.array(opt_time_)
    forward_backward_time_ = np.array(forward_backward_time_)
    forward_backward_opt_time_ = np.array(forward_backward_opt_time_)

    print(
        f"\nthe mean and vari forward time is {forward_time_.mean(), forward_time_.var()}\n backward is {backward_time_.mean(), backward_time_.var()}\n opt is {opt_time_.mean(), opt_time_.var()}\n forward_backward_time is {forward_backward_time_.mean(), forward_backward_time_.mean()}\n forward_backward_opt_time is {forward_backward_opt_time_.mean(), forward_backward_opt_time_.mean()}"
    )


if __name__ == "__main__":
    hyper_params: dict[str, Any] = {
        "vocab_size": 50257,
        "context_length": 128,
        "num_layers": 2,
        "d_model": 128,
        "num_heads": 4,
        "d_ff": 352,
        "theta": 10000.0,
        "train_times": 0,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "t_w": 5,
        "t_c": 100,
        "l2_norm_max": 1.0,
        "weight_decay": 0.1,
        "betas": (0.9, 0.99),
        "eps": (1e-8),
        "device": torch.device("cpu"),
        "batch_size": 4,
        "train_context_length": 64,
    }

    analysis_time(
        "/home/sleepinlava/Projects/python/cs336/assignment1-basics/checkpoints/test.bin",
        "/home/sleepinlava/Projects/python/cs336/assignment1-basics/checkpoints/vaild.bin",
        50,
        10,
        hyper_params,
    )
