from time import time

def TimeCounter_Process(function):
    def warp_func(*args, **kwargs):
        start_time = time()
        result = function(*args, **kwargs)
        end_time = time() 
        print(f'Function {function.__name__!r} executed in {(end_time-start_time ):.4f}s')
        return result
    return warp_func 

