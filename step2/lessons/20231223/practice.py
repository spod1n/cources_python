import threading
import time


def rest(filename, dir):
    for x in range(0, filename):
        with open(f'proc{dir}/{x}.txt', 'w') as f:
            f.write("hello")


if __name__ == "__main__":
    s = time.time()
    proc1 = threading.Thread(target=rest, args=(500, 1))
    proc2 = threading.Thread(target=rest, args=(500, 2))
    proc1.start()
    proc2.start()
    print(time.time() - s)