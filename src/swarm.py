import numpy as np
from drone import Drone

class Swarm:
    def __init__(self, N, space_dim=(100,100,100), num_anchors=3):
        self.N = N
        self.space_dim = space_dim
        self.num_anchors = num_anchors
        self.drones = self.initialize_drones()

    def initialize_drones(self):
        drones = []
        positions = np.random.uniform(low=0, high=self.space_dim, size=(self.N, 3))
        anchor_indices = np.random.choice(self.N, self.num_anchors, replace=False)
        for i in range(self.N):
            drones.append(Drone(i, positions[i], i in anchor_indices))
        return drones