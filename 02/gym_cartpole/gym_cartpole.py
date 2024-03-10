#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.keras", type=str, help="Output model path.")
parser.add_argument("--hidden_layer", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--hidden_layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--activation", default="none", choices=["none", "relu", "tanh", "sigmoid"], help="Activation.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--val_split", default=0.0, type=float, help="How many validation data points to use in.")
parser.add_argument("--eval_after_train", default=True, action="store_true",
                    help="Make the model cartpole evaluation after training.")
parser.add_argument("--eval_after_train_steps", default=10, type=int,
                    help="Steps number to make the model cartpole evaluation after training.")


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        # print(" - ", logs, " (add logs)")
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        # print(" - ", logs, " (on epoch end)")
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)

    def on_train_end(self, logs=None):
        if args.eval_after_train:
            writer = self.writer("cartpole_evaluation")
            score = evaluate_model(self.model, seed=args.seed, episodes=args.eval_after_train_steps,
                                   report_per_episode=True, writer=writer)
            print("The average score was {}.".format(score))
            writer.add_scalar("score", score, args.eval_after_train_steps + 1)


def evaluate_model(
        model: keras.Model, seed: int = 42, episodes: int = 100, render: bool = False, report_per_episode: bool = False,
        writer=None
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            prediction = model.predict_on_batch(observation[np.newaxis])[0]
            if len(prediction) == 1:
                action = 1 if prediction[0] > 0.5 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
        if writer:
            writer.add_scalar("score", score, episode + 1)
        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


def main(args: argparse.Namespace) -> keras.Model | None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    if not args.evaluate:
        # Create logdir name
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ))

        # Load the data
        data = np.loadtxt("gym_cartpole_data.txt")
        np.random.shuffle(data)
        observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

        # TODO: Create the model in the `model` variable. Note that
        # the model can perform any of:
        # - binary classification with 1 output and sigmoid activation;
        # - two-class classification with 2 outputs and softmax activation.

        model = keras.Sequential()

        model.add(keras.layers.Input([observations.shape[1]]))
        activation = args.activation if args.activation != "none" else None
        for _ in range(args.hidden_layers):
            model.add(keras.layers.Dense(units=args.hidden_layer, activation=activation))
        model.add(keras.layers.Dense(units=1, activation="sigmoid"))

        if args.decay == 'linear':
            max_steps = (observations.shape[0] // args.batch_size) * args.epochs
            learning_rate = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.learning_rate,
                end_learning_rate=args.learning_rate_final,
                decay_steps=max_steps
            )
        elif args.decay == 'exponential':
            max_steps = (observations.shape[0] // args.batch_size) * args.epochs
            learning_rate = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate,
                decay_rate=args.learning_rate_final / args.learning_rate,
                decay_steps=max_steps
            )
        elif args.decay == 'cosine':
            max_steps = (observations.shape[0] // args.batch_size) * args.epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi))
            alpha = (args.learning_rate_final / args.learning_rate) / (1 - cosine_decay) - cosine_decay / (
                    1 - cosine_decay)
            learning_rate = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.learning_rate,
                alpha=alpha,
                decay_steps=max_steps
            )
        else:
            learning_rate = args.learning_rate

        if args.optimizer == 'SGD':

            if args.momentum is not None:
                kwargs = {'momentum': args.momentum, 'nesterov': True}
            else:
                kwargs = {}

            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # TODO: Prepare the model for training using the `model.compile` method.
        model.compile(optimizer=optimizer,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

        model.summary()

        tb_callback = TorchTensorBoardCallback(args.logdir)

        model.fit(x=observations, y=labels, batch_size=args.batch_size, epochs=args.epochs,
                  callbacks=[tb_callback], validation_split=args.val_split)
        # Save the model, without the optimizer state.
        model.save(args.model)

    else:
        # Evaluating, either manually or in ReCodEx
        model = keras.models.load_model(args.model, compile=False)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print("The average score was {}.".format(score))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
