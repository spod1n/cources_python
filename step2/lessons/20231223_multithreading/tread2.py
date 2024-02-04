import threading
import time


def job(number):
    time.sleep(2)
    print(f'Stop job #{number}')


def run(count):
    threads = [threading.Thread(target=job, args=(i,), name=f'Thread # {i}') for i in range(0, count)]
    for x in threads:
        print(f"{x.name}")
        x.start()
    for x in threads:
        x.join()


run(6)
print("ERROR")