import yaml

# Load the configuration parameters from config.yaml
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Swarm parameters
N = config['swarm']['N']
SPACE_DIM = config['swarm']['space_dim']
NUM_ANCHORS = config['swarm']['num_anchors']

# Drone parameters
CUTOFF_DISTANCE = config['drone']['cutoff_distance']

# Consensus algorithm parameters
ROUNDS = config['consensus']['rounds']