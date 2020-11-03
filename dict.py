import string
from collections import OrderedDict

def create_dictionary():
    s = string.ascii_lowercase
    odict = OrderedDict()
    for count, i in enumerate(s):
        odict[i] = count
    odict['\''] = 26
    odict[' ']  = 27
    return odict
