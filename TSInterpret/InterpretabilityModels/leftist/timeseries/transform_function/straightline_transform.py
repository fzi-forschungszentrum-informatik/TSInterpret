import numpy as np

from TSInterpret.InterpretabilityModels.leftist.transform import Transform

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class StraightlineTransform(Transform):
    """
    Method to create neighbors representation in explained instance data space.
    """

    def __init__(self, explained_instance):
        """
        Must inherit Transform class.

        Parameters:
            explained_instance (Instance): instance to explain.
        """
        Transform.__init__(self, explained_instance)
        self.segments_interval = None

    def apply(self, neighbor_mask):
        """
        Transform a neighbor into the explained instance data space from its interpretable features representation (mask).

        Parameters:
            neighbor_mask (np.ndarray): representation of the neighbor in the explained instance data space.

        Returns:
            neighbor_values (np.ndarray): representation of the neighbor in the explained instance data space.
        """

        # get interval of consecutive segment where neighbor mask equal 0
        zeroes_interval = np.array([(i, i) for i in np.where(neighbor_mask == 0)[0]])
        neighbor_values = self.explained_instance.copy().astype(float)
        neighbor_values = self._compute_segment(zeroes_interval, neighbor_values)
        return neighbor_values

    def _compute_segment(self, zeroes_interval, neighbor_values):
        """
        Compute the new segments by connecting the first and last element of consecutive segment in instance to explain when neighbor mask is 0.

        Parameters:
            zeroes_interval (np.ndarray): list of pair which represent interval of consecutive segment associated to cosnecutive 0 in neighbor mask.
            neighbor_values (np.ndarray): the initial value of the neighbor.

        Returns:
            neighbor_representation (np.ndarray): representation of the neighbor in the explained instance data space.

        WARNING: straightline transform,
        if it's create a constant subseries use instead a slighty slope with a delta, because FastShapelet doesn't work with subseries of contant values

        """
        for zero_interval in zeroes_interval:
            first_segment_interval = self.segments_interval[zero_interval[0]]
            end_segment_interval = self.segments_interval[zero_interval[1]]
            coordinate_start = (
                first_segment_interval[0],
                self.explained_instance[first_segment_interval[0]],
            )
            coordinate_end = (
                end_segment_interval[-1],
                self.explained_instance[end_segment_interval[-1]],
            )
            if coordinate_end[1] == coordinate_start[1]:
                coordinate_end = (
                    end_segment_interval[-1],
                    self.explained_instance[end_segment_interval[-1]] + 0.001,
                )
            neighbor_values[
                first_segment_interval[0] : end_segment_interval[-1] + 1
            ] = self._compute_line_values(coordinate_start, coordinate_end)
        return neighbor_values

    def _compute_line_values(self, coordinate_start, coordinate_end):
        """
        Compute the intermediate values of a straight line between two coordinates.

        Parameters:
            coordinate_start (float,float): coordinates of the first point of the line.
            coordinate_end (float,float): coordinates of the last point of the line.

        Returns:
            intermediate_values (np.ndarray): intermediate values of the straight line between the two coordinates.
        """
        x1, y1 = coordinate_start
        x2, y2 = coordinate_end
        slope = (y1 - y2) / (x1 - x2)
        b = y1 - slope * x1
        return np.array([(slope * x) + b for x in range(x1, x2 + 1)])
