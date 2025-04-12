# Thomas's Notes on Training Regime

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

## `Exploration balance`
Epochs 6900 - 7800
- Training from last checkpoint
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
    - Model move distribution still low
    - Fix: try from initial run checkpoint, but with new temp scheduler


## `Exploration balance pt 2`
Epochs 4200v2 - ?
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
    - Model move distribution still low
    - Fix: try from initial run checkpoint, but with new temp scheduler

# Future runs
## `Middlegame and Endgame`
Epochs ? - ?
- Training from last checkpoint
- Begin from random position
- Warmup: 40000 samples
- Max moves: 128 (64 per player)
- LR: 2e-5
- Temperature scheduling
    - 3 until move 6
    - 1 until move 10
    - Linear decay to 0 by move 20
- Exploration bonus coefficient: 1.5
- Value loss weight: 1.5
- Results
    - 

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
- Exploration bonus coefficient: 1.5
- Value loss weight: 1.5
- Results
    - 