# shobu_rl

## Resources
- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- https://huggingface.co/blog/deep-rl-ppo
- https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
- https://arxiv.org/pdf/1707.06347 

## Some interesting stats
(computed by brandon)
- Branch factor: approx. 62.2
- Total number of legal positions: <= 2.62 * 10^25 (26190232295586403046560000)
- Number of legal moves in starting position: 232

compare with:
- chess: branch factor of 35, legal positions ~10^44, 20 first moves
- checkers: branch factor of 6.4, legal positions 5 * 10^20, 7 first moves
- go: branch factor 250, legal positions 2.1 * 10^170, 361 first moves
- ludo: just totally guessing, branch factor probably like 2ish, 1 first move

## Progress/design choices
- PPO
    - Board representation decisions
    - Model output decisions
    - PPO framework
    - Runs into instability issues due to lack of exploration
- MCTS
    - Board representation decisions
    - Model output decisions
    - Parallelized game simulations
    - Playout cap randomization
    - Parallel training and simulation
    - Rolling window of training samples
    - Increase complexity of policy/value heads
    - Dirichlet noise based on branch factor
    - Explorer for debugging mcts/model
    - Determining optimal rollout value during training
    - Exploration bonus hyperparameter for early exploration when values are poor
    - Loss function weighting to priortize value function