import timeit


# a decorator to comcalcute the time
def time(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"FPS: {1 / (end - start):.2f}")
    return wrapper