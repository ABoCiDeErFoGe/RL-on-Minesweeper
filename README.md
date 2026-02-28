# RL-on-Minesweeper
The initial intention for this project was to train an agent that can plays the classic game Minesweeper. However, as I was working on the project, I realized that I need a convenient way of training, tweaking hyperparams, measuring performance and possibly visually compare the performance between differnet model. And therefore, this project is about an GUI application that does the above thing, making training agent much easier for everyone!

## Installation and Execution
Before running the application, some dependencies need to be installed:
```
pip install -r requirements.txt
```
Once completed, the application can be start with:
```
python main.py
```
Enjoy!

## Reward Function

For each **left-click action**, the agent receives:

(revealed_cells_after_click / total_cells) + 0.5 * (revealed_cells_after_click / unrevealed_cells_before_click)

and Terminal rewards:

- +1 for winning  
- -1 for losing  
- +0.01 for each non-terminal step  

This reward function is designed for the Hybrid agent, where deterministic rules perform all safe moves and the learning agent acts only when no rule-based move is available.

Since rule-based actions never cause a loss, the agent is incentivized to reveal cells that expand the board and enable further safe rule application. The reward therefore promotes exploration that increases the solvable region of the board.

(The effect of this reward function has no yet been tested, as training agent takes longer than I expected, will update if this is good or bad when the training and testing is done)

To modify the terminal award, simply go to `config.py` to change the value.

For modification to reward function, check out `step()` function in `MSEnv` class in `Game.py` 

## Algorithm and network
The agent is trained using DQN with a Convolution Neural Network, consists of 3 convolutional layer and 2 fully connected layer. The details of the network can be found on `RL_agent.py`. 

The training and optimization process is largely inspired by https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Future Work
Currently the application only supports tweaking batch size, gamma, epsilon, target network update rate (TAU), and the learning rate. Other hyperparameters such as number of layers as well as the reward function, should also be made avaliable to the user to provide a more accessibility to the user. 

When the application is loading, it lacks a clear indication, sometimes results in the application being unresponsive. A simple loading animation would helps a lot.
## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

