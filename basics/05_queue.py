from threading import Thread
from queue import Queue
import time
import random

class Producer(Thread):
    def __init__(self, queue): 
        Thread.__init__(self)
        self.queue = queue
        
    def run(self): 
        for i in range(5): 
            item = random.randint(0, 256)
            self.queue.put(item)
            print(f"producer notify: item appended to queue by {item, self.name}")
            time.sleep(1)
    
class Consumer(Thread):
    def __init__(self, queue): 
        Thread.__init__(self)
        self.queue = queue
        
    def run(self): 
        while True:  
            item = self.queue.get()
            print(f"consumer notify: item popped from queue by {item, self.name}")
            self.queue.task_done()
            
if __name__ == "__main__":
    queue = Queue()
    t1 = Producer(queue)
    t2 = Consumer(queue)
    t3 = Consumer(queue)
    t4 = Consumer(queue)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    t1.join()
    t2.join()
    t3.join()
    t4.join()
