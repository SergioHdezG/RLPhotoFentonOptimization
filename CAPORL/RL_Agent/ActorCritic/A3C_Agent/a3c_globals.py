# These three variables will be initialized in a3c_problem in solve method
coord = None  # Train coordinator
global_raw_rewards = None  # deque(maxlen=100) to track 100 last reward values
global_episodes = None  # global episodes counter
