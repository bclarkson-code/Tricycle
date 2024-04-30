import torch

from nanoGPT.model import GPT, GPTConfig, RMSNorm
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.layers import RMSNormV2
from tricycle.loss import cross_entropy
from tricycle.models import GPTV2
from tricycle.optimisers import StochasticGradientDescent
from tricycle.scheduler import lr_schedule
from tricycle.tensor import to_tensor
from tricycle_datasets.shakespeare import ShakespeareChar


def load_model_andrej(config: SmolGPTConfig):
    gpt_config = GPTConfig()
    gpt_config.block_size = config.context_window
    gpt_config.vocab_size = config.vocab_size
    gpt_config.n_layer = config.n_layers
    gpt_config.n_head = config.n_heads
    gpt_config.n_embd = config.embedding_dim
    gpt_config.dropout = config.input_dropout_prob
    gpt_config.bias = False
    return GPT(gpt_config)


def load_model_tricycle(config: SmolGPTConfig):
    return GPTV2(config)


def test_tricycle_gpt_matches_andrej():
    config = SmolGPTConfig()
    config.batch_size = 16
    config.activation_fn = RMSNormV2()
    shakespeare = ShakespeareChar()
    shakespeare.vocab_size = 65

    dataset = (
        CausalLMDataset(
            tokens=shakespeare,
            vocab_size=shakespeare.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .to_tensor()
        .to_vector()
        .shuffle()
    )
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(
        learning_rate=1e-3,
        weight_decay=0,
        momentum=0,
    )

    tr_model = GPTV2(config)


def transfer_weights(a_model: GPT, tr_model: GPTV2):
    tr_model.position_embedding.weights._data = (
        a_model.transformer.wpe.weight.detach().numpy()
    )
    tr_model.token_embedding.weights._data = (
        a_model.transformer.wte.weight.detach().numpy()
    )
    for a_block, tr_block in zip(a_model.transformer.h, tr_model.blocks):
        tr_block.attention_block.in_projection.weights._data = (
            a_block.attn.c_attn.weight.detach().numpy().T
        )
        tr_block.attention_block.out_projection.weights._data = (
            a_block.attn.c_proj.weight.detach().numpy().T
        )
        tr_block.mlp_block.linear_1.weights._data = (
            a_block.mlp.c_fc.weight.detach().numpy().T
        )
        tr_block.mlp_block.linear_2.weights._data = (
            a_block.mlp.c_proj.weight.detach().numpy().T
        )
    tr_model.head.weights._data = a_model.lm_head.weight.detach().numpy().T
    return a_model, tr_model


if __name__ == "__main__":
    torch.manual_seed(0)
    import numpy as np

    np.random.seed(0)

    # load model
    config = SmolGPTConfig()
    # config.n_layers =
    config.vocab_size = 65
    config.batch_size = 1
    a_model = load_model_andrej(config)
    tr_model = load_model_tricycle(config)
    a_model, tr_model = transfer_weights(a_model, tr_model)
    tr_model.to_gpu(1)

    # load dataset
    shakespeare = ShakespeareChar()
    shakespeare.vocab_size = 65

    dataset = (
        CausalLMDataset(
            tokens=shakespeare,
            vocab_size=shakespeare.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .to_tensor()
        .to_vector()
        .shuffle()
    )

    # forward
    for _ in range(config.gradient_accumulation_steps):
        input, output = dataset[0]
        input.to_gpu(1)
        output.to_gpu(1)

        tr_out, tr_grads = tr_model(input)
        a_out, a_loss, a_grads = a_model(
            torch.tensor(input._data),
            targets=torch.tensor(output.xp.argmax(output._data, -1)),
        )
        a_out.retain_grad()

        tr_loss = cross_entropy(output, tr_out).from_vector().mean().mean()
        tr_loss /= config.gradient_accumulation_steps
        a_loss /= config.gradient_accumulation_steps

        tr_out_numpy = tr_out.from_gpu()
        a_out_numpy = a_out.detach().numpy()

        breakpoint()
        assert tr_out_numpy.close_to(a_out_numpy, rtol=1e-3, atol=1e-3)

        tr_loss.backward()
        a_loss.backward()

    for key in list(tr_grads.keys())[::-1]:
        if key == "logits":
            continue
        print(key)
        diff = (
            tr_grads[key].grad.from_gpu().numpy()
            - a_grads[key].grad.detach().numpy()
        )
        tr = tr_grads[key].grad
        aj = a_grads[key].grad
        ratio = abs(diff).mean() / abs(aj).mean()
        if ratio < 1e-3:
            print("✅")
            print()
            continue

        breakpoint()

        print()
        print(tr_grads[key].grad.from_gpu().numpy()[0][0][:10])
        print(a_grads[key].grad.detach().numpy()[0][0][:10])
        print(diff[0][0][:10])
        print(abs(diff).mean())
    breakpoint()
    # assert tr_out.grad.close_to(
    #     a_out.grad.detach().numpy(), rtol=1e-6, atol=1e-7
    # )
