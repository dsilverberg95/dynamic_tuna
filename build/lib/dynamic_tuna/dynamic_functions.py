### some possible n_ei_function

def linear_n_ei_candidates(n, a, b, from_start=True):
    '''
    if from_start=True:
    - n is number of completed trials
    - n_ei_candidates = a * n + b
    - b is number of candidates in first trial (since n=0)
    if from_start=False: 
    - n is number of remaining trials
    - n_ei_candidates = (-a) * (n-1) + b
    - b is number of candidates in final trial (since n=1)
    '''

    if from_start:
        n_ei_candidates = (a * n) + b

    else:
        n_ei_candidates = b - (a * (n - 1))

    return max(1, round(n_ei_candidates))


def polynomial_n_ei_candidates(n, a, b, c, from_start=True):
    '''
    if from_start=True:
    - n is number of completed trials
    - n_ei_candidates = (a * (n ** c)) + b
    - b is number of candidates in first trial (since n=0)
    if from_start=False: 
    - n is number of remaining trials
    - n_ei_candidates = b - (a * ((n - 1) ** c))
    - b is number of candidates in final trial (since n=1)
    '''

    if from_start:
        n_ei_candidates = (a * (n ** c)) + b

    else:
        n_ei_candidates = b - (a * ((n - 1) ** c))

    return max(1, round(n_ei_candidates))


def linear_xi(n, a, b, from_start=True):
    '''
    if from_start=True:
    - n is number of completed trials
    - xi = a * n + b
    - b is xi in first trial (since n=0)
    if from_start=False: 
    - n is number of remaining trials
    - xi = (-a) * (n-1) + b
    - b is xi in final trial (since n=1)
    '''
    if from_start:
        xi = (a * n) + b
    
    else:
        xi = b - (a * (n - 1))

    return max(.01, xi)


