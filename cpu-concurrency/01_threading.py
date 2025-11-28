import threading

def my_func(thread_number: int): 
    return print(f'my_func called by thread: {thread_number}')

def main(): 
    threads = []
    for idx in range(10):
        t = threading.Thread(target=my_func, args=(idx, ))
        threads.append(t)
        t.start()
        t.join()
        
if __name__ == "__main__":
    main()