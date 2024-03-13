import time


def tic():
    global tic_time
    tic_time = time.time()


def toc(name="Last operation"):
    print(f"{name} took {time.time() - tic_time:.4f} seconds")