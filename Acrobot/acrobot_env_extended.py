from gym.envs.classic_control.acrobot import AcrobotEnv

class AcrobotEnvExtended(AcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(AcrobotEnvExtended, self).__init__(*args, **kwargs)
        self.max_steps = 700
        self.actual_steps = 0
        
    def step(self, action):
        self.actual_steps += 1
        obs, reward, done, _, _ = super().step(action)
        done = done or self.actual_steps >= self.max_steps
        return obs, reward, done, False, {}
    
    def reset(self, *args, **kwargs):
        self.actual_steps = 0
        return super().reset(*args, **kwargs)