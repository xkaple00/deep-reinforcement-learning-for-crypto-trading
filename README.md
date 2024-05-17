# Deep Reinforcement Learning for Crypto Trading
[![DEMO LIVE TRADING](https://img.youtube.com/vi/elY9TdrdpgI/0.jpg)](https://www.youtube.com/watch?v=elY9TdrdpgI) <br />

This repository accompanies my blog series: https://medium.com/@sane.ai/deep-reinforcement-learning-for-crypto-trading-72c06bb9b04c <br />


## Part 0: Introduction
Set up: <br />
git clone https://github.com/xkaple00/deep-reinforcement-learning-for-crypto-trading.git <br />
cd deep-reinforcement-learning-for-crypto-trading <br />
conda env create -f environment.yml <br />

Add your API keys to keys.py <br />

## Part 1: Data preparation
Jupyter notebook to create dataset: <br />
dataset.ipynb <br />

## Part 2: Strategy:
Training environment: <br />
./envs/training_env.py <br />

## Part 3: Training
Command to start training: <br />
python train.py <br />

Tensorboard logs example: <br />
https://drive.google.com/file/d/12IyS3PKTx0KQr-J28vYOYJon5qpIRhsB/view <br />

## Part 4: Backtesting
Jupyter notebook to backtest on validation dataset: <br />
backtest.ipynb <br />
