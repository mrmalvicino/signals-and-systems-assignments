import numpy as np
import math as mt
from matplotlib import pyplot as plt
from scipy import signal


# EJERCICIO 2.01
'''
f_s = 44100 # Muestras por segundo
a = -0.5 # Inicio del intervalo
b = 0.5 # Final del intervalo
closedInterval = False # Cerrar intervalo con el mayorante

t = np.arange(a,b+int(closedInterval)/f_s,1/f_s)
N = int(np.sign(f_s%2) - round(a*f_s))
u = np.concatenate((np.zeros(N) , np.ones(int(len(t)-N))), axis=0)

n = 0
for t_i in range(0,len(t),1):
    n = n + np.sign(mt.ceil(t[t_i]))

# print(t)
# print(u)
print('Cantidad de elementos de t:', len(t))
print('Cantidad de positivos con for:', n)
print('Cantidad de negativos con fórmula:', N)

plt.plot(t,u,"-r")
'''

# EJERCICIO 2.04
'''
n = np.arange(-10,11)
y = np.exp(n*1j/5*np.pi - np.pi/3)
plt.figure()
plt.stem(n, y.real, 'r')
plt.stem(n, y.imag, 'b')
plt.grid()
'''

# EJERCICIO 2.05
'''
n = np.arange(-10,11)
y = np.cos(np.pi*n/5 - np.pi/3) # y = y_par + y_impar
y_par = 0.5*(y + np.flip(y))
y_impar = 0.5*(y - np.flip(y))

plt.figure()
plt.plot(n, y, 'r')
plt.plot(n, y_par, 'g')
plt.plot(n, y_impar, 'b')
plt.plot(n, y-y_par-y_impar, '-o')
plt.grid()

print((y==y_par+y_impar).all())
'''

# EJERCICIO 2.07
'''
f_s = 44100 # Muestras por segundo
a = -1 # Inicio del intervalo
b = 2 # Final del intervalo
closedInterval = True # Cerrar intervalo con el mayorante

t = np.arange(a,b+int(closedInterval)/f_s,1/f_s)

N = int(np.sign(f_s%2) - round(a*f_s))

def reverse(x):
    y = np.flipud(x)
    return y

def delay(x, t_0):
    M = int(t_0*f_s)
    y = np.zeros(len(x))
    
    if M > 0:
        for i in range(0,len(x)-M):
            y[int(i+M)] = x[i]
    else: # ¿Se podría reemplazar esta parte por un reverse?
        for i in range(0,len(x)+M):
            y[i] = x[int(i-M)]
    
    return y

def recPulse(t, tao):
    x = np.zeros(len(t))
    inicio = int(len(t)/2) - int(tao*f_s/2)
    fin = int(len(t)/2) + int(tao*f_s/2)
    x[inicio:fin] = 1
    return x

x_1 = recPulse(t, 1)
x_2 = np.sin(2*np.pi*t)
x_3 = np.sin(2*np.pi*t+np.pi/2)
x = x_3
y_1 = reverse(delay(x, 0.1))
y_2 = delay(reverse(x), 0.1)

plt.figure()
plt.plot(t, x, '-r')
plt.plot(t, y_1, '--c')
plt.plot(t, y_2, '--y')
plt.grid()

print('t =', t)
print('x =', x)
'''








































