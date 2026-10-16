import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import data_load, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.softmax import softmax
from cs336_basics.transformer import transformer_lm


@torch.no_grad()
def evaluate_fn(model: nn.Module, data_array: np.memmap, hyper_params: dict):
    model.eval()

    batch_size = hyper_params["batch_size"]
    context_length = hyper_params["context_length"]
    eval_device = hyper_params["device"]

    x, y = data_load(data_array, batch_size, context_length, eval_device)

    logits = model(x)

    preds = softmax(logits, dim=-1).argmax(dim=-1)

    acc = torch.eq(preds, y).sum() / y.numel()

    return acc


def single_train(
    model: nn.Module,
    hyper_params: dict,
    optimizer: optim.Optimizer,
    data_array: np.memmap,
    global_train_times: int,
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

    optimizer.zero_grad()

    logistic = model(x)
    loss = cross_entropy(logistic, y)
    loss.backward()
    gradient_clipping(model.parameters(), l2_norm_max)
    optimizer.step()
    # 按照参数根据梯度更新

    return loss
    # 返回交叉熵损失，当loss降低时储存下来
    #


def full_train(
    train_data_load_path: str,
    vaild_data_load_path: str,
    output_path: str,
    train_times: int,
    model: nn.Module,
    hyper_params: dict,
    checkpoint_load_path=None,
    load_from_former=False,
) -> None:

    plt.ion()  # plt的交互模式
    fig_1, ax = plt.subplots()

    fig_2, bx = plt.subplots()

    train_losses = []
    train_steps = []

    vaild_acc = []
    vaild_steps = []

    (line1,) = ax.plot([], [], label="Train Loss")
    ax.set_xlabel("train steps")
    ax.set_ylabel("train loss")
    ax.grid(True)
    ax.legend()

    (line2,) = bx.plot([], [], label="vaild acc")
    bx.set_xlabel("vaild steps")
    bx.set_ylabel("vaild acc")
    bx.grid(True)
    bx.legend()

    single_loss = 1e9
    total_acc = -1e5
    run_times = 0
    max_lr = hyper_params["max_lr"]
    weight_decay = hyper_params["weight_decay"]
    betas = hyper_params["betas"]
    eps = hyper_params["eps"]

    train_data_list = np.memmap(train_data_load_path, dtype=np.uint32, mode="r")
    vaild_data_list = np.memmap(vaild_data_load_path, dtype=np.uint32, mode="r")

    opt = AdamW(model.parameters(), max_lr, weight_decay, betas, eps)
    # 想一想正弦调整学习率的目的是什么

    if load_from_former == True:
        if checkpoint_load_path is not None:
            run_times = load_checkpoint(checkpoint_load_path, model, opt)
        else:
            raise ValueError("No Model Path was inputed")

    for t in tqdm(range(train_times - run_times)):
        train_loss = single_train(model, hyper_params, opt, train_data_list, t + run_times)

        with torch.no_grad():
            acc = evaluate_fn(model, vaild_data_list, hyper_params)

            train_steps.append(t)
            vaild_steps.append(t)
            train_losses.append(train_loss.item())
            vaild_acc.append(acc.item())

            line1.set_data(train_steps, train_losses)
            line2.set_data(vaild_steps, vaild_acc)

            # 动态调整坐标轴
            ax.relim()
            bx.relim()
            ax.autoscale_view()
            bx.autoscale_view()

            # refresh fig

            fig_1.canvas.draw_idle()
            fig_2.canvas.draw_idle()
            plt.pause(0.01)
            plt.pause(0.01)
            fig_1.savefig("./loss.png")
            fig_2.savefig("./acc.png")

        print(f"epoch is {t}, and the loss is {train_loss}, the acc is {acc}")
        # 使用t + run_times的目的
        # 因为我们这里实际上要实现在整个周期上进行学习率的正弦调节
        # 如果只使用t的话会丢失前面学习率的调节
        if single_loss > train_loss and acc > total_acc:
            single_loss = min(single_loss, train_loss)
            total_acc = max(acc, total_acc)
            save_checkpoint(model, opt, t, output_path)
            torch.save(model.state_dict(), "full_weights.pth")


if __name__ == "__main__":
    hyper_params: dict = {
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

    model = transformer_lm(50257, 128, 2, 128, 4, 352, 10000.0, device=hyper_params["device"])

    # print(evaluate_fn(model, ))

    full_train(
        "/home/sleepinlava/Projects/python/cs336/assignment1-basics/cs336_basics/test.bin",
        "/home/sleepinlava/Projects/python/cs336/assignment1-basics/cs336_basics/vaild.bin",
        "/home/sleepinlava/Projects/python/cs336/assignment1-basics/cs336_basics/test.pth",
        5000,
        model,
        hyper_params,
    )
