# Thomas's Notes on Training Regime Timeline

## `Intial`
Epochs 0 - 4200
- Initial training from random self play
- Begin from starting position
- Warmup: 10000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 1 until move 30
    - Linear decay to 0 by move 60
- Exploration bonus coefficient: 1.5
- Value loss weight: 1.5
- Results
    - Some overfitting observed empirically, model plays very similarly each game
    - To mitigate this: increase exploration during trainig

## `Exploration emphasis`
Epochs 4200 - 6900
- Training from last checkpoint
- Begin from starting position
- Warmup: 40000 samples (bc I was limited in compute)
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1.5
- Value loss weight: 1.5
- Results
    - The priors are almost uniform now, not avoiding basic mate in 1's
    - The exploration is likely a little too high now
    - Value function still understands material advantage; unlearned overfitting but not basic understanding
    - Fix: turn down exploration bonus

## Major Bug Found
- Found a major bug in eval.py, model learning better than expected
- Will continue training the runs below

## `Start from noisy prior, learned value function`
Epochs 6900 - 8600
- Training from last checkpoint
- Begin from starting position
- Warmup: 12000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 8600
        - against 3200: 46-0-4
        - against 6200v2: 32-1-17
        - against 8300v2: 25-2-23
    - Model improved from `Intial`, even with `Start from sharper prior, learned value function`
    - Policy distribution still mostly flat
    - Continue from other run; it can only barely find mate in 1 with the test case, and doesn't search 
    into the checkmate position for the second candidate move

## `Start from sharper prior, learned value function`
Epochs 4200v2 - 8300v2
- Training from checkpoint 4200
- Begin from starting position
- Warmup: 12000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 4700v2
        - against 3200: 37-4-9
        - against 4200v2: 30-3-17
    - 6500v2
        - against 3200: 43-2-5
        - against 4200v2: 31-7-12
    - 8100v2
        - against 6200v2: 30-4-16
        - against 7200v2: 24-8-18
    - 8300v2
        - against 8600 (see above run): 23-2-25
    - Model is steadily improving, but plateauing in recent epochs
    - Even with `Start from noisy prior, learned value function`
    - Try curriculum learning: introduce random positions and play out from there

## Bug Fix
- Positions where there are no valid moves is a loss for the current player
- Check implemented for training and MCTS

## `Incorporating random positions in training`
Epochs 8600 - ?
- Training from last checkpoint from `Start from noisy prior, learned value function`
- 0.5 probability to start from random position, 0.5 probability to start from starting position
    - To keep the model from forgetting opening board states
- Warmup: 25000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 0: play optimally from a random position
    - or same temp scheduler as before
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 

## `Continuing training from start positions`
Epochs 8600 - ?
- Training from last checkpoint from `Start from noisy prior, learned value function`
- Begin from starting position
- Warmup: 25000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 

# Future runs
## `Increased learning rate`
Epochs ? - ?
- Training from last checkpoint
- Begin from starting position
- Warmup: 40000 samples
- Max moves: 128 (64 per player)
- LR: 6e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 