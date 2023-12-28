import threading
import sys


def job(number):
    for x in range(10000):
        x**x
    sys.stdout.flush()


def run(count):
    threads = [threading.Thread(target=job, args=(i,),name=f"Потік №{i}",) for i in range(0, count)]
    for x in threads:
        print(x.name)
        x.start()
    for x in threads:
        x.join()

run(80)