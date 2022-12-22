import numpy as np

from TSInterpret.InterpretabilityModels.leftist.timeseries.segmentator.segmentator import (
    Segmentator,
)

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class UniformSegmentator(Segmentator):
    """
    Uniform time series segmentator.

    Attribute:
        nbsegments (int): number of segments to compute with similar size.
    """

    def __init__(self, nb_segments):
        """
        Must inherit TsSegmentator class.
        """
        Segmentator.__init__(self)
        if isinstance(nb_segments, int):
            self.nb_segments = nb_segments
        else:
            raise TypeError("nb_segments must be an integer")

    def segment(self, time_series):
        """
        Segment the time series.

        Returns:
            nb_segments, segments_interval (int,np.ndarray): the number and the intervals of the segments.
        """
        # print(time_series)
        len_time_series = len(time_series)
        len_segment_base = (len_time_series + self.nb_segments - 1) // self.nb_segments
        rest = (len_time_series + self.nb_segments - 1) % self.nb_segments
        start_position_current_segment = 0
        segments_interval = []
        for i in range(self.nb_segments):
            len_segment = len_segment_base
            if rest > 0:
                len_segment += 1
                rest -= 1
            segments_interval.append(
                (
                    start_position_current_segment,
                    start_position_current_segment + len_segment - 1,
                )
            )
            start_position_current_segment = (
                start_position_current_segment + len_segment - 1
            )
        return self.nb_segments, np.array(segments_interval)
