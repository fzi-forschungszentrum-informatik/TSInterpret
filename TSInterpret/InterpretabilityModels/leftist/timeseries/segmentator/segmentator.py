from abc import ABC, abstractmethod

__author__ = "Mael Guilleme mael.guilleme[at]irisa.fr"


class Segmentator(ABC):
    """
    Abstract class of the time series segmentator.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract __init__ must be called by concrete class.
        """

    @abstractmethod
    def segment(self, time_series):
        """
        Segment the time series.

        Paramaters:
            time_series (np.ndarray): the time series to segment.

        Returns:
            nb_segment, segments_interval (int,np.ndarray): the number and the intervals of the segments.
        """
        pass
