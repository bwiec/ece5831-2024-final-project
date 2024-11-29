class debug:
    def __init__(self, enable_debug):
        self.enable_debug = enable_debug

    def print(self, s):
        if self.enable_debug:
            print(s)