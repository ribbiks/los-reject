import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __add__(self, other):
        return Point(self.x + other[0], self.y + other[1])

    def __sub__(self, other):
        return Point(self.x - other[0], self.y - other[1])

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        else:
            raise TypeError("Can only multiply Point by a scalar (int or float)")

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError("Cannot divide by zero")
            return Point(self.x / scalar, self.y / scalar)
        else:
            raise TypeError("Can only divide Point by a scalar (int or float)")

    def __rmul__(self, scalar):
        return self * scalar

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.x,self.y))

    def __getitem__(self, key):
        if key == 0 or key == -2:
            return self.x
        elif key == 1 or key == -1:
            return self.y
        else:
            raise IndexError("Point index out of range")

    def __setitem__(self, key, value):
        if key == 0 or key == -2:
            self.x = value
        elif key == 1 or key == -1:
            self.y = value
        else:
            raise IndexError("Point index out of range")

    def __iter__(self):
        yield self.x
        yield self.y

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def rotate(self, angle):
        # rotate the point by the given angle (in radians)
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        x = self.x * cos_theta - self.y * sin_theta
        y = self.x * sin_theta + self.y * cos_theta
        return Point(x, y)


class LineSegment:
    def __init__(self, start, end, metadata=None):
        self.start = Point(start[0], start[1])
        self.end = Point(end[0], end[1])
        self.metadata = metadata

    def __getitem__(self, key):
        if key == 0 or key == -2:
            return self.start
        elif key == 1 or key == -1:
            return self.end
        else:
            raise IndexError("LineSegment index out of range")

    def __setitem__(self, key, value):
        if key == 0 or key == -2:
            self.start = value
        elif key == 1 or key == -1:
            self.end = value
        else:
            raise IndexError("LineSegment index out of range")

    def __str__(self):
        return f"LineSegment({self.start}, {self.end})"

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        yield self.start
        yield self.end

    def flip(self):
        self.start, self.end = self.end, self.start

    def angle_with_x_axis(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.atan2(dy, dx)

    def length(self):
        return (self.end - self.start).magnitude()


class BoundingBox:
    def __init__(self, x, y, width, height):
        self.x = x # bottom left x
        self.y = y # bottom left y
        self.width = width
        self.height = height

    def contains(self, point):
        return (self.x <= point.x < self.x + self.width and
                self.y <= point.y < self.y + self.height)

    def intersects(self, other):
        return not (other.x > self.x + self.width or
                    other.x + other.width < self.x or
                    other.y > self.y + self.height or
                    other.y + other.height < self.y)
