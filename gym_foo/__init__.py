from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
    max_episode_steps=1000,
    reward_threshold=975.0,
)
