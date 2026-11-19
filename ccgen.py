import math
from fractions import Fraction
from collections import defaultdict
from itertools import combinations, permutations
from math import factorial

def make_ops(bra, H, Ts):

    opes = []

    if bra == 1:
        bra1 = ("i", "+", "o", 0), ("a", "-", "v", 0)
        opes += (bra1)                   
    if bra == 2:
        bra2 = ("i", "+", "o", 0), ("j", "+", "o", 0), ("b", "-", "v", 0), ("a", "-", "v", 0)
        opes += (bra2)
    
   
    if H == "f":
        Hf = ("p", "+", "g", 1), ("q", "-", "g", 1)
        opes += (Hf)
    else:
        Hv = ("p", "+", "g", 1), ("q", "+", "g", 1), ("s", "-", "g", 1), ("r", "-", "g", 1)
        opes += (Hv)
    
    loop = 0
    oop = ["k", "l", "m", "n"]
    vop = ["c", "d", "e", "f"]
    for num, exc in enumerate(Ts):
        if exc == 1:
            exc1 = ((vop[num+loop], "+", "v", num+2), (oop[num+loop], "-", "o", num+2))
            opes += (exc1)
        if exc == 2:
            exc2 = ((vop[num+loop], "+", "v", num+2), (vop[num+loop+1], "+", "v", num+2), (oop[num+loop+1], "-", "o", num+2), (oop[num+loop], "-", "o", num+2))
            opes += (exc2)
            loop += 1
            
    return opes


def contraction(ops):
    result = []
    cre = [(i, op) for i, op in enumerate(ops) if op[1] == "+"]
    ana = [(i, op) for i, op in enumerate(ops) if op[1] == "-"]
    t_type = [op[3] for op in ops if op[3] >= 2]
    for anop in permutations(ana):
        pairing = list(zip(cre, anop))
        ok = True
        for (i, a), (j, b) in pairing:
            if a[3] != b[3] and i < j and a[2] in ("o", "g") and b[2] in ("o", "g"):
                continue
            elif a[3] != b[3] and i > j and a[2] in ("v", "g") and b[2] in ("v", "g"):
                continue
            else:
                ok = False
                break
        if ok:
            linked = []
            for i in pairing:
                H = [(j, op) for j, op in i if op[3] == 1]
                T = [(j, op) for j, op in i if op[3] >= 2]
                if len(H) and len(T) != 0:
                    linked.append(T[0][1][3])

            if set(linked) == set(t_type):
                result.append([tuple(sorted((i, j))) for (i, a), (j, b) in pairing])
    return result


def cross_counting(pairs):
    n = 0
    sign = 1
    for (i, j), (k, l) in combinations(pairs, 2):
        if i < k < j < l or k < i < l < j:
            n += 1
    if n % 2 == 1:
        sign *= -1
    return sign


def rename_table(ops, pairs):
    rename = {}
    for i, j in pairs:
        x, y = ops[i], ops[j]
        if x[3] == 0:
            rename[y[0]] = x[0]
        elif y[3] == 0:
            rename[x[0]] = y[0]
        elif x[2] == "g":
            rename[x[0]] = y[0]
        elif y[2] == "g":
            rename[y[0]] = x[0]

    return rename


def get_factor_and_c(ops_test2):
    factor = []
    creoph = [op[0] for op in ops_test2 if op[1] == "+" and op[3] == 1]
    anaoph = [op[0] for op in ops_test2 if op[1] == "-" and op[3] == 1]

    if len(creoph) == 2:
        name = "g"
    else:
        name = "f"
    c_H = factorial(len(creoph)) ** 2  
    get_f_ops = tuple(creoph + anaoph[::-1])
    factor.append((name, get_f_ops))
    
    coe_H = Fraction(1, (c_H))

    
    coet_list = []
    for i in range(2, 7):
        creopt = [op[0] for op in ops_test2 if op[1] == "+" and op[3] == i]
        anaopt = [op[0] for op in ops_test2 if op[1] == "-" and op[3] == i]

        if len(creopt) == 0:
            continue

        if len(creopt) == 2:
            name = "t_2"
        if len(creopt) == 1:
            name = "t_1"
        c_t = factorial(len(creopt)) ** 2  
        get_f_ops = tuple(anaopt[::-1] + creopt)
        factor.append((name, get_f_ops))
        
        coe_t = Fraction(1, (c_t))
        coet_list.append(coe_t)

       
    coe = coe_H * math.prod(coet_list)
    T2_n = 0
    T1_n = 0
    for j in coet_list:
        if j == Fraction(1, 4):
            T2_n +=1
        if j == Fraction(1, 1):
            T1_n += 1
    if T2_n != 0:
        coe *= Fraction(1, (factorial(T2_n)))
    if T1_n != 0:
        coe *= Fraction(1, (factorial(T1_n)))

            
    return factor, coe



def get_terms(ops):
    formula, coe = get_factor_and_c(ops)
    result = contraction(ops)
    term = []
    for i in result:
        sign = cross_counting(i)
        rename = rename_table(ops, i)
        new = []
        for name, idx in formula:
            new.append((name, tuple(rename.get(ch, ch) for ch in idx)))
        term.append((coe * sign, new))
    return term


def get_standard(term):
    c, index = term
    stad = []
    for name, idx in index:
        if len(idx) == 2:
            stad.append((name, idx))
        else:
            h1, h2, h3, h4 = idx[:4]
            if h1 > h2:
                h1, h2 = h2, h1
                c *= -1
            if h3 > h4:
                h3, h4 = h4, h3
                c *= -1
            idx = (h1, h2, h3, h4)
            stad.append((name, idx))

    stada = (c, tuple(stad))
    return stada


def combine(term):
    f_terms2 = defaultdict(Fraction)
    for i in term:
        stada = get_standard(i)
        f_terms2[(stada[1])] += stada[0]
    return f_terms2


def generate_einsum(f_terms2, ops_test2):
    name2arr = {}
    lines = []

    for op in ops_test2:
        name2arr[op[0]] = op[2] 
    cre1 = [op[0] for op in ops_test2 if op[1] == "+" and op[3] == 0 ]
    ana1 = [op[0] for op in ops_test2 if op[1] == "-" and op[3] == 0 ]
    out = "".join(cre1 + ana1[::-1])

    for factors, c in f_terms2.items():
        subs2 = []
        arr2 = []
        for name, idx in factors:
            subs1 = "".join(idx)
            arr1 =", ".join(name2arr[op] for op in idx)
            
            if name == "t_1":
                rhs = " t1"
            if name == "t_2":
                rhs = " t2"
            if name == "g" or name == "f":
                rhs = f"{name}[{arr1}]"
            subs2.append(subs1)
            arr2.append(rhs)


        subs3 = ",".join(subs2)
        arr3 = ",".join(arr2)
        lines.append(f"{float(c)} * np.einsum('{subs3}->{out}', {arr3}, optimize=True)")
    return lines


def get_residual(parts):
    coder = []
    for i in parts:
        term = get_terms(i)
        f_terms = combine(term)
        line = generate_einsum(f_terms, i)
        coder += line
    coder = " + ".join(coder)
    return coder
