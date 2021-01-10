class Angle:
    """
    Class used to store x,y,z values of an angle.
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Overriding str() method
    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}]"
