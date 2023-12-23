import gradio as gr
import numpy as np
from tiktoken import get_encoding
import torch
from llm_from_scratch.demos.word2vec import Model, WindowDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "test_loss": 0.012575260954158172,
    "train_loss": 0.019275289789616765,
    "epoch": 9,
    "timestamp": 1702836996,
    "checkpoint_dir_name": "checkpoint_000009",
    "should_checkpoint": True,
    "done": True,
    "training_iteration": 10,
    "trial_id": "41d2a_00007",
    "date": "2023-12-17_18-16-37",
    "time_this_iter_s": 11.633716106414795,
    "time_total_s": 121.94960904121399,
    "pid": 718364,
    "hostname": "ben-desktop",
    "node_ip": "192.168.1.146",
    "config": {
        "model": {"embedding_size": 256, "weight_init": "xavier_norm"},
        "dataset": {
            "window_size": 5,
            "encoding": "p50k_base",
            "train_fraction": 0.8,
            "batch_size": 32,
        },
        "optimiser": {
            "type": "SGD",
            "lr": 0.07177823282340132,
            "momentum": 0.9671284106092733,
        },
        "loss": {"type": "cross_entropy"},
        "logger": {
            "project": "llm_from_scratch/word2vec",
            "tracking_url": "http://localhost:8080",
        },
        "training": {"epochs": 10},
    },
    "time_since_restore": 121.94960904121399,
    "iterations_since_restore": 10,
    "experiment_tag": "7_lr=0.0718",
}


def load_model():
    tokeniser = get_encoding("p50k_base")
    model = Model(tokeniser.n_vocab)
    model_checkpoint_path = "/home/ben/ray_results/objective_function_2023-12-17_18-08-18/objective_function_41d2a_00007_7_lr=0.0718_2023-12-17_18-08-19/checkpoint_000009/checkpoint.pt"

    model_state, _ = torch.load(model_checkpoint_path)
    parsed = {k.replace("_orig_mod.", ""): v.cpu() for k, v in model_state.items()}
    model.load_state_dict(parsed)
    model.eval().to(DEVICE)
    return model, tokeniser


model, tokeniser = load_model()


def encode(text: str) -> torch.Tensor:
    tokens = tokeniser.encode(text)
    zeros = np.zeros(tokeniser.n_vocab, dtype=np.float32)
    zeros[tokens] = 1

    with torch.no_grad():
        return model.encoder(torch.Tensor(zeros).to(DEVICE))


def decode(embedding: torch.Tensor) -> str:
    with torch.no_grad():
        output = model.decoder(embedding)
        print(output)
        best_token = int(output.argmax().cpu().numpy())
        print([best_token])
        decoded = tokeniser.decode([best_token])
        print(decoded)
    return decoded


def combine_words(left: str, right: str) -> str:
    left_input = encode(left)
    right_input = encode(right)
    combined = torch.mean(torch.stack([left_input, right_input]), dim=0)
    return decode(combined)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        left = gr.Textbox(label="First word")
        right = gr.Textbox(label="Second word")
        output = gr.Textbox(label="Combined word")

        button = gr.Button(
            "Combine words",
        )
        button.click(combine_words, inputs=[left, right], outputs=output)

    demo.launch(show_api=False, server_name="0.0.0.0")
