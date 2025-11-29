"""
degrees.py
A helper module providing a Degrees class for convenient conversion
between degrees and radians.

"""

from se3kit.utils import deg2rad, rad2deg


class Degrees:
    """
    Represents an angle in degrees with convenient conversion to/from radians.

    Provides .deg and .rad properties for getting/setting the value
    in degrees or radians, respectively.
    """

    def __init__(self, x):
        """
        Initializes the Degrees object with a value in degrees.

        :param x: Angle value in degrees.
        :type x: float
        """
        self.x = x

    @property
    def deg(self):
        """
        Gets the stored angle in degrees.

        :return: Angle in degrees.
        :rtype: float
        """
        return self.x

    @deg.setter
    def deg(self, val):
        """
        Sets the angle value in degrees.

        :param val: Angle in degrees.
        :type val: float
        """
        self.x = val

    @property
    def rad(self):
        """
        Gets the stored angle converted to radians.

        :return: Angle in radians.
        :rtype: float
        """
        return deg2rad(self.x)

    @rad.setter
    def rad(self, val):
        """
        Sets the angle value in degrees.

        :param val: Angle in degrees.
        :type val: float
        """
        self.x = rad2deg(val)

    def __str__(self):
        """
        Returns the string representation of the angle.

        :return: Angle as a string.
        :rtype: str
        """
        return str(self.x)
