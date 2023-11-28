import numpy as np

def make_matrix(fen):
    res = [] 
    rows = fen.split('/')
    for row in rows:
        row_list = []
        pieces = row.split(" ", 1)[0]
        for thing in pieces:
            if thing.isdigit():
                row_list += '.' * int(thing)
            else:
                row_list += thing
        res.append(row_list)
    return res

def extract_metadata(fen):
    res = [] 
    data = fen.split(' ')
    
    if data[1][0] == 'w': res.append(1)
    else: res.append(0)

    if "K" in data[2]: res.append(1)
    else: res.append(0)

    if "Q" in data[2]: res.append(1)
    else: res.append(0)

    if "k" in data[2]: res.append(1)
    else: res.append(0)

    if "q" in data[2]: res.append(1)
    else: res.append(0)
        
    return res

def vectorize(fen):
    table = {
        '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        'P': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    
        'p': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'b': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'n': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    
    res = []
    for i in make_matrix(fen):
        res.append(list(map(table.get, i)))
    return np.array(res)