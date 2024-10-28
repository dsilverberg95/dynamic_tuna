### some possible n_ei_function

def linear(n, a=0, b=1, from_start=True):
    '''
    if from_start=True:
    - n is number of completed trials
    - n_ei_candidates = a * n + b
    - b is number of candidates in first trial (since n=0)
    if from_start=False: 
    - n is number of remaining trials
    - n_ei_candidates = (-a) * (n-1) + b
    - b is number of candidates in final trial (since n=1)
    by default, a=0 and b=1, i.e. perform random search
    '''

    if from_start:
        n_ei_candidates = (a * n) + b

    else:
        n_ei_candidates = b - (a * (n - 1))

    return max(1, n_ei_candidates)


def polynomial(n, a=0, b=1, c=1, from_start=True):
    '''
    if from_start=True:
    - n is number of completed trials
    - n_ei_candidates = (a * (n ** c)) + b
    - b is number of candidates in first trial (since n=0)
    if from_start=False: 
    - n is number of remaining trials
    - n_ei_candidates = b - (a * ((n - 1) ** c))
    - b is number of candidates in final trial (since n=1)
    by default, a=0,b=1,c=1 i.e. perform random search
    '''

    if from_start:
        n_ei_candidates = (a * (n ** c)) + b

    else:
        n_ei_candidates = b - (a * ((n - 1) ** c))

    return max(1, n_ei_candidates)






