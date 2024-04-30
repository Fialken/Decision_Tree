import math


columns = 1
lines = 1
data = [["" for _ in range(400)] for _ in range(400)]
arvore = [ ]
t_arvore = 0
n_arvore = 0



def entropia(aux_cont, l):
    res = 0
    for i in range(l):
        if aux_cont[i] == 0:
            break
        a = aux_cont[i] / l
        res += (-1 * a * math.log2(a))
    return res


def conta_aux1(indices_lines):
    t = len(indices_lines)
    aux = ["" for _ in range(t)]
    aux_cont = [0] * t
    ent = 0

    for i in range(t):
        indice = indices_lines[i]
        val = data[indice][columns - 1]

        for k in range(t):
            if val == aux[k]:
                aux_cont[k] += 1
                break
            if aux[k] == "":
                aux[k] = val
                aux_cont[k] = 1
                break

    ent = entropia(aux_cont, t)
    return ent

