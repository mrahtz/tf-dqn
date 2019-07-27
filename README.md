[![Build Status](https://travis-ci.com/mrahtz/tf-dqn.svg?branch=master)](https://travis-ci.com/mrahtz/tf-dqn)

# TensorFlow DQN

TensorFlow implementation of Deep Q-learning. Implements:
 
* [Vanilla DQN](https://www.nature.com/articles/nature14236)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Dueling DQN](https://arxiv.org/abs/1511.06581)
* [Double DQN](https://arxiv.org/abs/1509.06461)

## Results

Testing on Breakout across three seeds:

![](images/breakout_reward.png)

A cherry-picked episode:

![](images/breakout.gif)

## Setup

Install [`pipenv`](https://github.com/pypa/pipenv), then set up a virtual environment and install main dependencies with

```bash
$ pipenv sync
```

## Usage

To a train a policy, use [`train.py`](train.py).

`train.py` uses [Sacred](https://github.com/IDSIA/sacred) for managing configuration.
To specify training options overriding the defaults in [`config.py`](config.py), use `with` then a number of
`config=value` strings. For example:

```bash
$ python3 -m dqn.train with atari_config env_id=PongNoFrameSkip-v4 dueling=False
```

TensorBoard logs will be saved in a subdirectory inside the automatically-created `runs` directory.

## Implementation gotchas

A couple of details to be aware of with DQN:
* The hard hyperparameters to get right are a) the ratio of training to environment interaction, and b)
  how often to update the target network. I used the setup from
  [Stable Baselines' implementation](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html).
* As mentioned in the [blog post](https://openai.com/blog/openai-baselines-dqn/) accompanying the release of
  OpenAI Baselines, don't forget the Huber loss.
  
Also, a general point: be super careful about normalising observations twice. It turns out Baselines had a bug for
[several months](https://github.com/openai/baselines/issues/431) because of this. We use TensorFlow
[assertion functions](https://www.tensorflow.org/api_docs/python/tf/debugging/assert_greater_equal) to make sure
the observations have the right scale right at the point they enter the network.

## Tests

To run smoke tests, unit tests and an end-to-end test on Cartpole, respectively:

```bash
$ python tests.py Smoke
$ python tests.py Unit
$ python tests.py EndToEnd
```
