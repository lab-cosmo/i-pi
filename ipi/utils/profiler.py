"""
 a few utility wrappers to help using line_profiler. 
 the @profile decorator profiles a function only if the code is run through kernprof
 otherwise the decorators do nothing. also adds a @profile_class decorator that applies
 @profile to all member functions in a class. 

 Usage: 
 import ipi.utils.profiler

 @profile
 def function_to_profile():

 @profile_class
 class class_to_profile:
"""

import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func

    builtins.profile = profile


def profile_class(cls):
    for attr in cls.__dict__:  # there's propably a better way to do this
        print("attaching ", attr)
        if callable(getattr(cls, attr)):
            print("attached")
            setattr(cls, attr, profile(getattr(cls, attr)))
    return cls


builtins.profile_class = profile_class
