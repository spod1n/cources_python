from multiprocessing import Process
import time


def rest(sec, proc_name):
    print(f"{proc_name} is going to sleep")
    time.sleep(sec)
    print(f"{proc_name} is alive")
    return sec


if __name__ == "__main__":
    proc = Process(target=rest, args=(3, "pr1"))
    proc1 = Process(target=rest, args=(5, "pr2"))
    print(proc)
    proc.start()
    proc1.start()
    print(proc)
    print(proc1)
    proc.kill()