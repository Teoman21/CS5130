import math

class Shape:
    """Base class for all shapes."""
    
    def __init__(self, name):
        self.name = name
    
    def area(self):
        """Calculate the area of the shape."""
        raise NotImplementedError("Subclass must implement area()")
    
    def perimeter(self):
        """Calculate the perimeter of the shape."""
        raise NotImplementedError("Subclass must implement perimeter()")
    
    def __str__(self):
        return f"{self.name}"


class Circle(Shape):
    """Class representing a circle."""
    
    def __init__(self, name, radius):
        super().__init__(name)
        self.radius = radius
    
    def area(self):
        
        return math.pi * (self.radius ** 2)
    
    def perimeter(self):
        return 2 * math.pi * self.radius    
    
    
    def __str__(self):
        return f"Circle(radius={self.radius})"
    

circle = Circle("MyCircle", 5)
print(circle.area())     
print(circle.perimeter()) 
print(circle)             

class Rectangle(Shape):
    
    def __init__(self, name, width, height):
        super().__init__(name)
        self.width = width
        self.height = height
        
    def area(self):
        
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
     
    def __str__(self):
        return f"Rectangle(width={self.width}, height={self.height})"

     
     
rect = Rectangle("MyRect", 4, 6)
print(rect.area())      # 24
print(rect.perimeter()) # 20
print(rect)             # Rectangle(width=4, height=6)
        
        
        
class Triangle(Shape):
    
    def __init__(self, name, side_a, side_b, side_c):
        super().__init__(name)
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c
        
    def perimeter(self):
        return self.side_a + self.side_b + self.side_c
    
    def area(self):
        s = (self.side_a + self.side_b + self.side_c) / 2
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
    
    def __str__(self):
        return f"Triangle(sides={self.side_a},{self.side_b},{self.side_c})"
    
    

triangle = Triangle("MyTriangle", 3, 4, 5)
print(triangle.area())      # 6.0
print(triangle.perimeter()) # 12
print(triangle)             # Triangle(sides=3,4,5)


class ShapeCollection:
    
    def __init__(self, ):
        self.shapes = []
        
    def add_shape(self, shape):
         self.shapes.append(shape)
         
    def total_area(self):
        return sum(shape.area() for shape in self.shapes)
    
    def total_perimeter(self):
        return sum(shape.perimeter() for shape in self.shapes)
       
    
collection = ShapeCollection()
collection.add_shape(Circle("C1", 5))
collection.add_shape(Rectangle("R1", 4, 6))

print(collection.total_area())      # 102.53975
print(collection.total_perimeter()) # 51.4159


print("-"*20)

print("Tests")

# Test Circle
c = Circle("test", 10)
assert abs(c.area() - 314.159) < 0.01
assert abs(c.perimeter() - 62.8318) < 0.01

# Test Rectangle
r = Rectangle("test", 5, 10)
assert r.area() == 50
assert r.perimeter() == 30

# Test Triangle
t = Triangle("test", 3, 4, 5)
assert abs(t.area() - 6.0) < 0.01
assert t.perimeter() == 12

# Test ShapeCollection
collection = ShapeCollection()
collection.add_shape(c)
collection.add_shape(r)
assert abs(collection.total_area() - 364.159) < 0.01

print("All tests passed!")