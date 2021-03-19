'''
String of binary digits of length n,
can we find out the probability that are at least 2 consequtive ones

2 --> (1,1) out of 4 possibilities
3 --> (0, 1, 1), (1, 1, 0) , (1,1,1)
4 --> (0, 1, 1, 1) (1, 1, 1,1) (1,1, 0, 0)...
'''

n = 8

def generate_strings(length):
    """
    (0, 0, 0), (1, 0, 0), (1, 0, 1)
    """


def is_there_a_one(string):
    for idx, el in enumerate(string[:-1]):
        if el == 0:
            continue
        if string[idx+1] == 1:
            return 1
    return 0