from numpy import linalg as la
import numpy as np

SIZE =20

ones = np.ones(20)
V = np.transpose(np.array([0.5*ones,0.6*ones,0.8*ones]))
print('vshape', V.shape)
a = np.arange(60.).reshape(20,3)
b = np.arange(60.).reshape(20,3)
A=6
test = np.cross(a,(V+(b)))
print(test.shape)
print(a.shape, b.shape, V.shape)
# print(b)
# c = np.cross(a,b)
# print(c.shape)
# print(c)
up = np.exp(np.sum(V*(b), axis = 1)).reshape(SIZE,1)
do = la.norm(A*(a/A-b), axis = 1).reshape(SIZE,1)
print('updo', up.shape, do.shape)
res= a*((up)/(do))
print('rr', res.shape)

tt =(a)*((np.exp(np.sum(V*(b), axis = 1)))/(la.norm(A*(((a)/A)-(b)), axis = 1).reshape(SIZE,1)))
#test = np.sum(a*b, axis=1)
#print(test.shape)
#print(test)