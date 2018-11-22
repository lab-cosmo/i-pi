try:
    from mpi4py import MPI
    has_mpi = True
except:
    has_mpi = False

__all__ = ['is_master', 'has_mpi']


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if not cls in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

if has_mpi:
    
    class IPIMPI:
        __metaclass__ = Singleton
        
        def __init__(self):
            pass 
            
        def is_master(self):
            return (MPI.COMM_WORLD.rank == 0)
    
else:        
    class IPIMPI():
        
        __metaclass__ = Singleton
        
        def __init__(self):
            pass 
            
        def is_master(self):
            return False
        
        
