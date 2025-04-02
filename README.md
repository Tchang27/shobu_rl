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

## Training runs:
- Run 1:
    - Goal: Trial run
    - Intermediate reward is sum of model pieces - sum of opponent pieces, multiplied by 0.1
    - Win reward is 10
    - Draw penalty set to half of loss penalty, which is -win reward
    - Hyperparameters:
        - Batch size: 256
        - epochs: 4
        - training update: every 20 episodes
        - max turns: 100
        - C1 = 0.5
        - C2 = 0.005
        - EPSILON = 0.2
        - GAMMA = 0.99
        - LAMBDA = 0.9
        - Learning rates: 3e-4,3e-4,1e-4
    - Result: model got good at stalling

- Run 2:
    - Goal: Less draws, numerically stabilize logits and losses
    - Increase model complexity (inspired by AlphaGo)
    - No intermediate reward
    - Win reward is 10
    - Loss penalty is -10
    - Draw penalty is -15
    - Update opponent pool every 5000 episodes / 100 training steps
    - Hyperparameters:
        - Batch size: 512
        - epochs: 3
        - training update: every 50 episodes
        - max turns: 100
        - C1 = 0.5
        - C2 = 0.1
        - EPSILON = 0.2
        - GAMMA = 0.995
        - LAMBDA = 0.95
        - Learning rates: 3e-4,3e-4,3e-4
    - Result: model learning random actions (ie not capturing any useful information)

- Run 3:
    - Goal: model needs to capture information/strategies
    - Increase model complexity (inspired by AlphaGo)
    - Split up entropy loss into passive/aggressive
    - Fix action masking and sampling
    - Board representation overhaul
        - Different boards for each subboard
        - Different boards for each piece color
        - Store past 8 baord states (after each player move)
        - 64 x 4 x 4 image
    - No intermediate reward
    - Win reward is 1
    - Loss penalty is -1
    - Draw penalty is -1
    - Update opponent pool every 2000 episodes, start with random agent
    - Hyperparameters:
        - Batch size: 256
        - epochs: 4
        - training update: every 50 episodes
        - max turns: 50
        - C1 = 1
        - C2 = 0.03
        - C3 = 0.015
        - EPSILON = 0.2
        - GAMMA = 0.995
        - LAMBDA = 0.95
        - Learning rates: 2e-6,2e-6,2e-6
    - Result: 