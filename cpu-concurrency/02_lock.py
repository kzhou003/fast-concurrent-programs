import threading
import time
import os

from threading import Thread
from random import randint

threadLock = threading.Lock()

class myThreadClass(Thread):
    def __init__(self, name, duration): 
        Thread.__init__(self)
        self.name = name
        self.duration = duration 
        
    def run(self):
        threadLock.acquire()
        print(f"{self.name} running, belonging to process ID + {str((os.getpid()))} \n")
        time.sleep(self.duration)
        print(f"{self.name} completed after {self.duration} seconds\n")
        threadLock.release()
        
def main():
    start_time = time.time()
    thread1 = myThreadClass("Thread #1", randint(1, 10))
    thread2 = myThreadClass("Thread #2", randint(1, 10))
    thread3 = myThreadClass("Thread #3", randint(1, 10))
    thread4 = myThreadClass("Thread #4", randint(1, 10))
    thread5 = myThreadClass("Thread #5", randint(1, 10))
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    
    print("END \n")
    print(time.time() - start_time)
    
if __name__ == "__main__":
    main()
    