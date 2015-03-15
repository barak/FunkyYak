from test_util import *
from autograd import grad

def test_int_case():
    check_equivalent((lambda x:x*x)(2.0),         4.0)
    check_equivalent((lambda x:x*x)(2) + 0.0,     4.0)
    check_equivalent(grad(lambda x:x*x)(2.0),     4.0)
    check_equivalent(grad(lambda x:x*x)(2) + 0.0, 4.0)
