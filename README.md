# ShabuShabu:  Learning Shobu By Self-Play Through Reinforcement Learning

## Outline of Repo
- `CS1470 Final Presentation`: PDF of capstone presentation slides
- `ShabuShabu_Final_Paper`: PDF of final report
- `CS1470_Final_Outline.pdf`: PDF of outline

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

## shobu_ui
1. Make sure you have npm installed, and python packages `flask`, `flask-cors`, and
`gunicorn`.
2. We need to compile the UI. `cd shobu_ui`, and `npm i`. Then `npm run build`
3. `cd ..` back out to the root of the repo and run `gunicorn app:app`. (make sure you have the checkpoints
also in the root dir, or edit `app.py` to fix those paths)
4. For production, can process requests in parallel via `gunicorn --workers=16 app:app -b 0.0.0.0:1470`.

<!-- ## Progress/design choices
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
    - Loss function weighting to priortize value function -->