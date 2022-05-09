# EA614 2020.1 - EFC1
# Marcos Diaz RA: 221525

from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

# transforma um sinal h em uma matriz de convolucao H
# entradas: sinal h e tamanho do sinal s (inteiro)
# saida: matriz H_(P x K)

def gerar_H (h, K):
    D = len(h)
    P = K+D-1
    H = (np.hstack([h, np.zeros(P-D)])).reshape(P,1)
    for i in range(1,K):
        temp = (np.hstack([np.zeros(i),h,np.zeros(P-D-i)])).reshape(P,1)
        H = np.hstack([H, temp])
    return H

# implementa a convolucao matricial
# entradas: matriz H e sinal s
# saida: sinal x

convolucao_matricial= lambda H, s : H @ s

# Resolucao

# (e)
# Do item (c)

h = np.array([1, -0.5])
w1 = np.array([1, 0.5, (0.5)**2, (0.5)**3, (0.5)**4])
w2 = np.array([1, 1.5, 0.7, -0.2, 0.3])

g1 = convolucao_matricial(gerar_H(h, len(w1)), w1)
g2 = convolucao_matricial(gerar_H(h, len(w2)), w2)

# (f)

#sinal de entrada
s = np.sign(np.random.randn(100))

# sinal transmitido pelo canal
x = convolucao_matricial(gerar_H(h, len(s)), s)

# distancia de correlacao
distancia = distance.correlation(x,np.hstack([s,np.array([0])]))

# (g)

# sinal equalizado por w1
y1 = convolucao_matricial(gerar_H(w1, len(x)), x)

print(y1)
# sinal equalizado por w2
y2 = convolucao_matricial(gerar_H(w2, len(x)), x)
'''
# grafico 1
plt.figure()
plt.stem(s, markerfmt='bo', linefmt=None, label="Entrada")
plt.stem(y1, markerfmt='r.', linefmt=None, label="Saida")
plt.xlabel('Tempo Discreto')
plt.ylabel("Sinais")
plt.title("Sistema com equalizador w1")
plt.legend(loc='upper left')
plt.savefig("grafico1.png")

# grafico 2
plt.figure()
plt.stem(s, markerfmt='bo', linefmt=None, label="Entrada")
plt.stem(y2, markerfmt='r.', linefmt=None, label="Saida")
plt.xlabel('Tempo Discreto')
plt.ylabel("Sinais")
plt.title("Sistema com equalizador w2")
plt.legend(loc='upper left')
plt.savefig("grafico2.png")
'''
