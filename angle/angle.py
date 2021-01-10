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

    # Overriding left multiplication method
    def __mul__(self, other):
        return self.__multiplication(other)

    # Overriding left multiplication method
    def __rmul__(self, other):
        return self.__multiplication(other)

    def __multiplication(self,other):
        """
        Used for overriding left and right multiplication.
        """

        if isinstance(other,Angle):
            # Multiplication between same format
            self.x = self.x * other.x
            self.y = self.y * other.y
            self.z = self.z * other.z
        else:
            # Scalar multiplication
            self.x = other * self.x
            self.y = other * self.y
            self.z = other * self.z

        return self

    def add(self, x_value, y_value, z_value):
        """
        Adds x,y,z values to existing ones.
        """
        self.x = self.x + x_value
        self.y = self.y + y_value
        self.z = self.z + z_value


