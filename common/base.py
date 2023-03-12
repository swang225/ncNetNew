import torch
import time


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class Counter:
    def __init__(self, total=None):
        self.total = total
        self.start_time = None
        self.count = 0

    def start(self):
        print("START!")
        self.start_time = time.process_time()

    def update(self):
        self.count += 1
        cur_time = time.process_time()

        log = f"time elapsed: {sec_to_time(cur_time - self.start_time)}"
        if self.total is not None:
            log += f", progress: {int(self.count * 100 / self.total)}%"

        print(log )


def sec_to_time(sec):

    ms = max(0, int(sec * 1000))

    sec = ms // 1000
    ms = ms % 1000

    min = sec // 60
    sec = sec % 60

    hrs = min // 60
    min = min % 60

    return f"{hrs} hours, {min} mins, {sec} seconds, {ms} milliseconds"


def normalize(text):
    return ' '.join(text.replace('"', "'").split())


def is_matching(label, prediction):
    return normalize(label) == normalize(prediction)