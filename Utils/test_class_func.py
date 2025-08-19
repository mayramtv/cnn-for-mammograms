# Test
class Test_py:
    def __init__(self, t_name="test0"):
        self.t_name = t_name

    def print_(self):
        return "Class -> Try if python utils connects to notebook: " + self.t_name

def test_py(t_name):
    return "Function -> Try if python utils connects to notebook: " + t_name