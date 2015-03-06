from test_util import *
from funkyyak import grad

# This checks if fanout causes non-constant overhead.
# If it does, the call to df will take forever.
#
# If a compiler is smart enough to rewrite (x+x)/2 as plain x,
# substitute (x+1.0000000001*x)/2.0000000001 or some such.

def test_fanout():
    def f(x):
        for _ in range(1,100000):
            x = (x+x)/2
        return x
    df = grad(f)
    check_equivalent(df(12.34), 1.0)
