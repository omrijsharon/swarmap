import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

from .settings import CUTOFF_DISTANCE


class Drone:
    def __init__(self, id, position, is_anchor=False):
        self.id = id
        self.absolute_position = position
        self.is_anchor = is_anchor
        self.cutoff_distance = CUTOFF_DISTANCE
        self.distance_matrix = None
        self.relative_positions = None

    def calculate_distances(self, drones):
        """Calculate distances to other drones and store in self.distance_matrix."""
        positions = np.array([drone.absolute_position for drone in drones])
        distances = euclidean_distances([self.absolute_position], positions)[0]

        # If cutoff_distance is specified, set distances greater than the cutoff to infinity.
        if self.cutoff_distance is not None:
            distances[distances > self.cutoff_distance] = np.inf

        self.distance_matrix = distances

    def perform_mds(self):
        """Perform multidimensional scaling based on self.distance_matrix."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix is not calculated.")

        mds = MDS(dissimilarity='precomputed')
        self.relative_positions = mds.fit_transform(self.distance_matrix.reshape(-1, 1))
        return self.relative_positions

    def share_positions(self, other):
        """Share current relative positions with another drone."""
        other.receive_positions(self.relative_positions)

    def receive_positions(self, positions):
        """Receive relative positions from another drone and update own relative positions."""
        self.relative_positions = self.combine_positions(self.relative_positions, positions)

    def combine_positions(self, positions1, positions2):
        """Combine two sets of relative positions."""
        # For simplicity, just average the relative positions.
        return (positions1 + positions2) / 2