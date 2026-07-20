import math

my_tuple = (1, 2, 2, 3, 4, 4, 4)
print("The number of 4 in the tuple is: ", my_tuple.count(4))


class TestClass:
    
    name = "AppleOrg"

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def __str__(self):
        return f"TestClass: {self.name}"

    @classmethod
    def showClassMethod(self):
        print("The class method is " + self.name)

    @staticmethod
    def showStaticMethod(self):
        print("The static method is " + self.name)

test_class = TestClass("Apple")
print("The name is " + test_class.get_name())

TestClass.showClassMethod()
# test_class.showStaticMethod()

print("Execute in pythonABC.py, name: " + __name__)