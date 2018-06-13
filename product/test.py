class A:
    def __init__(self):
        self._value = 10
    
    @property
    def value(self):
        return self._value

a = A()
print(a.value)