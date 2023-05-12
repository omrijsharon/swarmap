import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

class Drone:
    def __init__(self, id, position, is_anchor=False, cutoff_distance=None):
        self.id = id
        self.position = position
        self.is_anchor = is_anchor
        self.cutoff_distance = cutoff_distance
        self.distance_matrix = None

    def calculate_distances(self, drones):
        """Calculate distances to other drones and store in self.distance_matrix."""
        positions = np.array([drone.position for drone in drones])
        distances = euclidean_distances([self.position], positions)[0]

        # If cutoff_distance is specified, set distances greater than the cutoff to infinity.
        if self.cutoff_distance is not None:
            distances[distances > self.cutoff_distance] = np.inf

        self.distance_matrix = distances

    def perform_mds(self):
        """Perform multidimensional scaling based on self.distance_matrix."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix is not calculated.")

        mds = MDS(dissimilarity='precomputed')
        positions = mds.fit_transform(self.distance_matrix.reshape(-1, 1))

        return positions

    def share_positions(self, other):
        """Share current positions with another drone."""
        other.receive_positions(self.positions)

    def receive_positions(self, positions):
        """Receive positions from another drone and update own positions."""
        self.positions = self.combine_positions(self.positions, positions)

    def combine_positions(self, positions1, positions2):
        """Combine two sets of positions."""
        # For simplicity, just average the positions.
        return (positions1 + positions2) / 2