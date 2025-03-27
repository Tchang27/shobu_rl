# shobu_rl
## Training runs:
- Run 1:
    - Intermediate reward is sum of model pieces - sum of opponent pieces, multiplied by 0.1
    - Win reward is 10
    - Draw penalty set to half of loss penalty, which is -win reward
    - Result: model got good at stalling