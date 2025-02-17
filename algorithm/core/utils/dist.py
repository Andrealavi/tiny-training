
try:
    import torchpack.distributed as dist
except:
    dist = None

__all__ = ["size", "rank", "local_rank", "init"]

# Returns the number of processes currently running
def size():
    if dist:
        return dist.size()
    else:
        return 1
    
# Returns rank of the current process
def rank():
    if dist:
        return dist.rank()
    else:
        return 0

# Returns local rank of the current process
# Useful to separate the processes that are on the same machine
def local_rank():
    if dist:
        return dist.local_rank()
    else:
        return 0
    

# Initializes the distributed environment
def init():
    if dist:
        return dist.init()
    else:
        pass