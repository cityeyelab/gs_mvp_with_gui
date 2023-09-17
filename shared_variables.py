from multiprocessing import Manager

class SharedVariables():
    def __init__(self) -> None:
        self.manager = Manager()
        self.my_bool = self.manager.Value(bool, True)
        self.my_bool2 = self.manager.Value(bool, True)
        self.args = {'my_bool': self.my_bool, 'my_bool2': self.my_bool2}

