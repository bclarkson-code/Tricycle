from andrej_model import GPT


def load_model_andrej():
    return model


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


if __name__ == "__main__":
    model = load_model_andrej()
    breakpoint()
