import multiprocessing
import numpy as np


def worker(queue: multiprocessing.Queue, path):
    while True:
        if queue.empty():
            return
        function, image_set, kernel, name = queue.get()
        parsed_data = function(image_set, kernel)
        np.savez_compressed(path + name, parsed_data)


def multiprocess_starter(data, path, number_of_processes: int):
    processes = []
    for i in range(number_of_processes):
        process = multiprocessing.Process(target=worker, args=(data, path), name=f"Process {i}")
        processes.append(process)
        process.start()
    for p in processes:
        p.join()
