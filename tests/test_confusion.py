from test_util import *
from autograd import grad

## See Siskind & Pearlmutter (2008), "Nesting forward-mode AD in a
## functional framework", Higher Order and Symbolic Computation
## 21(4):361-76, doi:10.1007/s10990-008-9037-1

def test_nest_hosc():
    shouldBeTwo = grad (lambda x: x * grad (lambda y: x*y) (2.0)) (1.0)
    check_equivalent(shouldBeTwo, 2.0)
