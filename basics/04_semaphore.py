import logging
import threading
import time
import random

logging.basicConfig(level = logging.INFO)

semaphore = threading.Semaphore(0)
item = 0 

def consumer(): 
    logging.info("Consumer waiting")
    semaphore.acquire()
    logging.info(f"Consumer notify: item number {item}")

def producer(): 
    global item 
    time.sleep(3)
    item = random.randint(0, 1000)
    logging.info(f"Producer notify: item number {item}")
    semaphore.release()
    
def main(): 
    for i in range(10): 
        t1 = threading.Thread(target=consumer)
        t2 = threading.Thread(target=producer)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
if __name__ == '__main__': 
    main()