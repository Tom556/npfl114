#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which processes the input using
        # a fully connected hidden layer with non-linear activation, and produces
        # one output (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        self._model = tf.keras.models.Sequential(
            [tf.keras.layers.InputLayer([4]),
             tf.keras.layers.Flatten(),
             #tf.keras.layers.Dense(args.hidden_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
             #tf.keras.layers.Dense(args.hidden_layer, activation='relu'),
             #tf.keras.layers.Dense(args.hidden_layer, activation='relu'),
             #tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dense(args.hidden_layer * 2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
             #tf.keras.layers.BatchNormalization(),
             #tf.keras.layers.Dense(args.hidden_layer, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
             tf.keras.layers.Dense(env.actions, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-5))]
        )

        self._baseline_model = tf.keras.models.Sequential(
            [tf.keras.layers.InputLayer([4]),
             tf.keras.layers.Flatten(),
             #tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dense(args.hidden_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
             #tf.keras.layers.BatchNormalization(),
             # tf.keras.layers.Dense(args.hidden_layer, activation='relu'),
             # tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-5))]

        )

        self._batch_size = args.batch_size
        self._model.compile(
            optimizer=self.get_optimizer(args),
            loss=tf.losses.SparseCategoricalCrossentropy(),
        )

        self._batch_size = args.batch_size
        self._baseline_model.compile(
            optimizer=self.get_optimizer(args),
            loss=tf.losses.MeanSquaredError(),
        )



    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)

        # TODO: Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        # - train the `baseline` model to predict `returns`
        advantages = returns - self._baseline_model.predict_on_batch(states).numpy().ravel()
        self._baseline_model.train_on_batch(states, advantages, sample_weight=advantages)
        advantages = returns - self._baseline_model.predict_on_batch(states).numpy().ravel()
        self._model.train_on_batch(states, actions, sample_weight=advantages)
        #self._baseline_model.train_on_batch(states, advantages, sample_weight=advantages)

    def predict(self, states):
        states = np.array(states, np.float32)

        # TODO: Predict distribution over actions for the given input states. Return
        # only the probabilities as NumPy arrays, not the baseline.
        preds = self._model.predict_on_batch(states)
        return preds.numpy()

    def get_optimizer(self, args):
        learning_rate_final = args.learning_rate_final
        decay_steps = int(args.episodes // args.batch_size)
        if args.decay == 'polynomial':
            self.learning_rate_schedule = tf.optimizers.schedules.PolynomialDecay(args.learning_rate,
                                                                             decay_steps=decay_steps,
                                                                             end_learning_rate=args.learning_rate_final)
        elif args.decay == 'exponential':
            decay_rate = learning_rate_final / args.learning_rate
            self.learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                              decay_steps=decay_steps,
                                                                              decay_rate=decay_rate, staircase=False)
        else:
            self.learning_rate_schedule = args.learning_rate

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate_schedule, clipnorm=.5)

        return optimizer


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--hidden_layer", default=512, type=int, help="Size of hidden layer.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Discount factor of the rewards.")

    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=50, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # optimizer
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.007, type=float)
    parser.add_argument("--decay", default="exponential", type=str)

    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False, seed=args.seed)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict([state])[0]
                # TODO(reinforce): Compute `action` according to the distribution returned by the network.
                # The `np.random.choice` method comes handy.
                action = np.random.choice(env.actions, p=probabilities)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute `returns` from the observed `rewards`.
            curr_return = 0.
            returns = []
            for reward in rewards[::-1]:

                curr_return = args.gamma * curr_return + reward
                returns.insert(0, curr_return)

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
