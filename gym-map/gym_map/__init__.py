from gym.envs.registration import register
 
register(id='Map-v0', 
    entry_point='gym_map.envs:MapEnv', 
)