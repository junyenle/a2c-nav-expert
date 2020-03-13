# a2c-nav-expert

instructions

1. pip install the gym-map environment
2. configure parameters in a2c/main.py and model in a2c/a2c/model.py
3. configure rewards and game parameters in gym-map/gym_map/envs/map_env.py
4. run a2c/main.py to train the agent
5. run game_only.py for JUST the game

notes

currently configured for SARL + heuristic expert (not described in paper)
but you can swap to any of the paper models with minimal code changes
