import numpy as np
from drone import Drone

from .settings import N, SPACE_DIM, NUM_ANCHORS


class Swarm:
    def __init__(self):
        self.N = N
        self.space_dim = SPACE_DIM
        self.num_anchors = NUM_ANCHORS
        self.drones = self.initialize_drones()

        # Calculate distances and perform MDS for all drones
        for drone in self.drones:
            drone.calculate_distances(self.drones)
            drone.perform_mds()

    def initialize_drones(self):
        drones = []
        positions = np.random.uniform(low=0, high=self.space_dim, size=(self.N, 3))
        anchor_indices = np.random.choice(self.N, self.num_anchors, replace=False)
        for i in range(self.N):
            drones.append(Drone(i, positions[i], i in anchor_indices))
        return drones

    def run_consensus_algorithm(self, rounds=100):
        """Run the consensus algorithm for a certain number of rounds."""
        for _ in range(rounds):
            for drone in self.drones:
                # Select another drone at random.
                other = np.random.choice(self.drones)
                drone.share_positions(other)