from TSInterpret.InterpretabilityModels.leftist.timeseries.segmentator.segmentator import (
    Segmentator,
)

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class ProvidedSegmentator(Segmentator):
    """
    Time series segmentator with provided segmentation.

    Attribute:
        segmentation (int): the size of the segments.
    """

    def __init__(self, segments):
        """
        Must inherit TsSegmentator class.
        """
        Segmentator.__init__(self)
        self.segments = segments

    def segment(self, time_series):
        """
        Segment the time series.

        Returns:
            nb_segments, segments_interval (int,np.ndarray): the number and the intervals of the segments.
        """

        return len(self.segments), self.segments
