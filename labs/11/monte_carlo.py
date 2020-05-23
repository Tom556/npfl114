#!/usr/bin/env python3
import argparse

import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--episodes", default=600, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=0.3, type=float, help="Discount factor of the rewards.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=50, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=True, seed=args.seed)

    # Create Q, C and other variables
    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing number of observed returns of a given (state, action) pair.

    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions])


    for _ in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards = [], [], []
        returns = 0
        while True:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random actions (there are env.actions of them),
            # otherwise, choose and action with maximum Q[state, action].

            if np.random.uniform() < args.epsilon:
                action = np.random.randint(low=0, high=env.actions)
            else:
                action = np.argmax(Q[state,:])

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                rewards[-1] = -1.
                break

        # TODO: Compute returns from the observed rewards.
        returns = list()
        curr_return = 0.

        states.reverse()
        actions.reverse()
        rewards.reverse()
        for reward in rewards:
            #returns.append(curr_return)
            curr_return = args.gamma * curr_return + reward #* (1. - args.gamma)
            returns.append(curr_return)

        # TODO: Update Q and C
        for state, action, curr_return in zip(states, actions, returns):
            C[state, action] += 1
            Q[state, action] = Q[state, action] + (1. / C[state, action]) * (curr_return - Q[state, action])

        #args.epsilon = args.epsilon * 0.9

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
