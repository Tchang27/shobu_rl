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
    - Value function knows material differences
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
    - Policy distribution still mostly flat but competitive with sharper prior;
    could indicate that the prior doesn't need to be very sharp, just good enough
    - Try curriculum learning: introduce random positions and play out from there

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
    - Mostly even with `Start from noisy prior, learned value function`
        - Lost more positions as white, won less positions as black
    - These checkpoints are deprecated due to compute/time constraints:
    given that the other training run `Start from noisy prior, learned value function`
    had slightly better performance, we opted to continue training from that checkpoint

## Bug Fix
- Positions where there are no valid moves is a loss for the current player
- Check implemented for remainder of training

## `Incorporating random positions in training`
Epochs 8600 - 15600_random
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
    - 13200_random
        - against 8300v2: 29-4-17, with 18-0-7 as black and 11-4-10 as white
        - against 8600: 26-8-16, with 15-5-5 as black and 11-3-11 as white
        - against 12200_random: 27-5-18, with 16-3-6 as black and 11-2-12 as white
    - 13800_random
        - against 8600: 24-7-19, with 16-4-5 as black and 8-3-14 as white
        - against 13000_random: 30-6-15, with 22-0-3 as black and 8-6-11 as white
    - 14600_random
        - against 8600: 22-6-22, with 12-3-10 as black and 10-3-12 as white
        - against 13400_random: 18-7-25, with 13-3-9 as black and 5-4-16 as white
    - 15400_random
        - against 8600: 20-12-18, with 12-7-6 as black and 8-5-12 as white
        - against 13800_random: 22-4-24, with 14-1-10 as black and 8-3-14 as white
    - 15600_random
        - against 12400_start: 22-9-19, with 15-5-5 as black and 7-4-14 as white
    - Having random positions interspersed actually leads to more train steps 
    because of less locking overlap and more game results for the value network
    - improvement is plateauing, we should increase the learning rate for this regime

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
    - 10400_start
        - against 8600: 27-6-17, with 16-5-4 as black and 11-1-13 as white
    - 11400_start
        - against 8600: 25-7-18, with 16-4-5 as black and 9-3-13 as white
        - against 10400_start: 23-2-25 with 16-1-8 as black and 7-1-17 as white
    - 12400_start
        - against 8600: 28-7-15, with 19-2-4 as black an 9-5-11 as white
        - against 10400_start: 27-6-17, with 16-2-7 as black and 11-4-10 as white
        - against 12400_random: 25-8-17, with 14-3-8 as black and 11-5-9 as white
        - against 15600_random: 19-9-22, with 14-4-7 as black and 5-5-15 as white
    - for the same number of epochs, it performs better than the model trained on
    random positions, but that could be due to the random position task being 
    harder and thus requiring more epochs
    - the model still seems to be improving, keeping learning rate same for now
    - ?_start
        - against ?_start
        - against ?_random

## `Increased learning rate, random positions`
Epochs 15600_random - ?
- Training from last checkpoint from `Incorporating random positions in training`
- 0.5 probability to start from random position, 0.5 probability to start from starting position
- Warmup: 25000 samples
- Max moves: 128 (64 per player)
- LR: 6e-5
- Temperature scheduling
    - 0: play optimally from a random position
    - or same temp scheduler as before
- Exploration bonus coefficient: 1
- Value loss weight: 1.5
- Results
    - 16400_random:
        - against 8600: 24-8-18, with 17-3-5 as black and 7-5-13 as white
        - against 13800_random: 23-5-22, with 17-2-6 as black and 6-3-16 as white
        - against 15600_random: 25-7-18, with 14-5-6 as black and 11-2-12 as white
    - 18600_random
        - against 12400_start: 50-11-39, with 32-5-13 as black and 18-6-26 as white
        - against 16800_random: 50-8-42, with 33-7-10 as black and 17-1-32 as white
    - 20500_random
        - against 12400_start: 57-4-39, with 38-1-11 as black and 19-3-28 as white
        - against 18600_random: 63-7-30, with 35-3-12 as black and 28-4-18 as white


# Future runs
## `Increased learning rate, starting position`
Epochs ? - ?
- Training from last checkpoint from `Continuing training from start positions`
- Begin from starting position
- Warmup: 25000 samples
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