# shobu_rl

## Resources
PPO
- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- https://huggingface.co/blog/deep-rl-ppo
- https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
- https://arxiv.org/pdf/1707.06347

MCTS
- https://arxiv.org/pdf/1712.01815
- https://github.com/JoshVarty/AlphaZeroSimple/tree/master
- https://arxiv.org/pdf/1902.10565
- https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a

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