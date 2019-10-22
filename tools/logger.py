
class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input, newline=True):
        input = str(input)
        if newline:
            input += '\n'

        with open(self.logger_path, 'a') as f:
            f.write(input)
            f.close()

        print(input)
