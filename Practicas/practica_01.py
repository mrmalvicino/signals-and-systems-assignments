import numpy as np
import matplotlib.pyplot as plt

# COMPLEJOS
'''
z_1 = complex(2,3)
z_2 = complex(4,5)
z_re = np.real(z_1)
z_im = np.imag(z_1)
z_ang = np.angle(z_1)
z_conj = np.conj(z_1)
suma = z_1 + z_2
resta = z_1 - z_2
multi = z_1 * z_2
divi = z_1 / z_2

print(z_1, z_re, z_im, z_ang, z_conj,)
print(np.absolute(z_1))
print('\n Suma: ', suma)
print('\n Resta: ', resta)
print('\n Multiplicacion: ', multi)
print('\n Division: ', divi)
'''

# MATRICES
'''
M = np.array([[1,2],[3,4]])
Mij = M[:,:]
v = np.array([1,2,3,4])

print('\n Dada la matriz M: \n', M)
print('\n El elemento Mij es: \n', Mij)
print('\n La dimensión de Mij es: \n', np.shape(Mij)) # ¿Qué significa el output "nada"? Con Mij=M[:,0] es uno, pero con Mij=M[0,0] es cero.
print(type(np.shape(Mij)))
print(M**2)
print(M*M) # ¡No está definida así la multiplicación matricial!
print(2**M)
print('\n v transformado: \n', np.reshape(v, (2,2)))
'''

# SENOIDALES
'''
T = 0.01
t = np.arange(-1,1+T,T)
f = 5
w = 2*np.pi*f
A = 2
x = A*np.sin(w*t)

plt.plot(t,x,"-o")
plt.grid
'''

# EJERCICIO 1.6
'''
T = 0.01

t = np.arange(-1,1+T,T)
t_i = -1

f = 1
w = 2*np.pi*f
A = 1

x = np.array([])

for t_i in t:
    x_i = A*np.sin(w*t_i)
    if x_i < 0:
        x = np.append(x, -x_i)
    else:
        x = np.append(x, x_i)

plt.plot(t,x,"-r")
plt.grid
'''

# EJERCICIO 1.07
'''
x = float(input('Insert number: \n'))

if x % 2 ==  0:
    print(x, 'es par')
else:
    print(x, 'es impar')
'''

# EJERCICIO 1.08
'''
numeros = [1,2,3,4,5]
for numero in numeros:
    print(numero)
'''

# EJERCICIO 1.09 (MAL)
'''
x = float(input('Evaluar primalidad de: \n'))
N = 0
c = np.array([2,3,4,5,6,7,8,9])

for i in range(0,8,1):
    if x % c[i] == 0:
        print(x, 'es divisible por', c[i])
        N += 1

# print('N =', N)

if (N == 0 and x > 9) or (N == 1 and x < 9):
    print(x, 'es primo')
else:
    print(x, 'no es primo')
'''

# EJERCICIO 1.09 BIS
'''
M = 0
P = []
S = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

for x in range(2,1000,1):
    N = 0
    c = np.array([2,3,4,5,6,7,8,9])
    
    for i in range(0,8,1):
        if x % c[i] == 0:
            N += 1
    
    if (N == 0 and x > 9) or (N == 1 and x < 9):
        M += 1
        P.append(x)

print('M =', M)
print('Cantidad de elementos en S =', len(S))
print('Cantidad de elementos en P =', len(P))

if(set(P) == set(S)):
    print("P y S son el mismo conjunto")
else:
    print("P - S =")
    print(set(P) - set(S))
'''

# EJERCICIO 1.09 FINAL
'''
N = 0 # Contador de divisores
c_i = 1

x = int(input('Evaluar primalidad de: '))
c = np.array([]) # Vector de cocientes

for c_i in range(1,x+1,1):
    c = np.append(c,c_i) # Agregar todos los naturales menores que x

for i in range(0,x,1):
    if x % c[i] == 0: # Si el resto de x/c_i es nulo para todo i
        print(x, 'es divisible por', c[i])
        N += 1

if N == 2:
    print(x, 'es primo')
else:
    print(x, 'no es primo')
'''

# EJERCICIO 1.10
'''
X = [1,'a',np.pi,True]
X_1 = [1,2,3]
X_2 = ['a','b','c','d']

print('La lista original es: \n X =', X)

Y = [] + ['y']*len(X)

print('La lista invertida tendrá la forma: \n Y =', Y)

for i in range(1,len(X)+1,1):
    Y[i-1] = X[-i]

print('La lista invertida es: \n Y =', Y)
'''






