import numpy as np
import scipy as s
import matplotlib.pyplot as plt

import mojmodul as mm
import funkcja15 as f15


arr = np.array([1, 2, 3, 4, 5])
print(arr)


print("\n")
A = np.array([[1, 2, 3], [7, 8, 9]])
print(A)

print("\n")
A = np.array([[1, 2, 3],
[7, 8, 9]])
print(A)


# =============================================================================
# print("\n")
# A = np.array([[1, 2, \#po backslash’u nie moze byc zadnego znaku!
# 3],
# [7, 8, 9]])
# print(A)
# =============================================================================


v = np.arange(1,7)
print(v,"\n")

v = np.arange(1,7)
print(v,"\n")

v = np.arange(-2,7)
print(v,"\n")

v = np.arange(1,10,3)# od 1 do 10 krok 3
print(v,"\n")

v = np.arange(1,10.1,3)
print(v,"\n")

v = np.arange(1,11,3)
print(v,"\n")


v = np.arange(1,2,0.1)
print(v,"\n")


v = np.linspace(1,3,4)
print(v,"\n")

v = np.linspace(1,10,4)
print(v,"\n")

X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2) # np.eye(2,2) # np.eye(2,3)
Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))

print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)


Z = np.eye(3)
U = np.block([[A], [Z]])
print(U)

V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
        ],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)


print("\n")
print( V[0,2] )
print("\n")
print( V[3,0] )
print("\n")
print( V[3,3] )
print("\n")
print( V[-1,-1] )
print("\n")
print( V[-4,-3] )
print("\n")
print( V[3,:] )
print("\n", V[:,2] )
print("\n", V[3,0:3] )
print("\n", V[np.ix_([0,2,3],[0,-1])] )
print("\n", V[3] )



Q = np.delete(V, 2, 0)
print(Q)
print("\n")
Q = np.delete(V, 2, 1)
print(Q)

v = np.arange(1,7)
print( np.delete(v, 3, 0) )

print("\n")
print(np.size(v))
print(np.shape(v))



print("\n")
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print("\n")
print( A-B )
print("\n")
print( 2*A )



print("\n")
MT1 = A*B
print(MT1)
print("\n")
DT1 = A/B
print(DT1)

print("\n")
C = np.linalg.solve(A,MT1)
print(C) # porownaj z macierza B


print("\n")
x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)


print("\n")
PM = np.linalg.matrix_power(A,2) # por. A@A
PT = A**2 # por. A*A
print(PM,"\n")
print(PT,"\n")

A.T # transpozycja
A.transpose()
A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
A.conj().transpose()


np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print( np.all(A) )
print( np.any(A) )

print( v > 4 )
print( np.logical_or(v>4, v<2))
print( np.nonzero(v>4) )
print( v[np.nonzero(v>4) ] )

print("\n")
print(np.max(A))
print(np.min(A))

print("\n")
print(np.max(A,0))
print(np.max(A,1))

print("\n")
# ’wektoryzacja’ macierzy
print( A.flatten() )
#print( A.flatten(’F’) )
print("\n")
# wymiary macierzy
print( np.shape(A) )
print("\n")
# liczba elementow macierzy
print( np.size(A) )


print("\n")
A = np.array([[1,1,1],
[1,1,0],
[0,1,1]])
b = np.array([[3],
[2],
[2]])
x = np.linalg.solve(A, b)
print(x)

# wyznacznik macierzy
print(np.linalg.det(A))
# uwarunkowanie macierzy
print(np.linalg.cond(A))
# macierz odwrotna
print(np.linalg.inv(A))

def f1(x):
    return x**2

print(f1(2))
print(mm.f2(2))

print("\n")
f3 = lambda a,x: a*x**3
print(f3(2,2))


print("\n")
# =============================================================================
# f_podcalkowa = lambda x: f1(x) + mm.f2(x) + f3(1,x)
# calka,blad = spint.quad(f_podcalkowa, 0, 1)  ???
# print("calka = "+str(calka))
# print("blad oszacowania = "+str(blad))
# =============================================================================

# =============================================================================
# x = [1,2,3]
# y = [4,6,5]
# plt.plot(x,y)
# #plt.show()
# =============================================================================



x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()


x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x,y1,'r:',x,y2,'g')
plt.legend(('dane y1','dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
help(plt.legend)
plt.show()


x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()

# zad 3
P1 = np.linspace(1,5,5)
P2 = np.linspace(5,1,5)
Z=np.zeros((3,2))
S= np.linspace(2,2,3)# dwujki
M= np.linspace(-90,-70,3)
D = np.ones((5,1))
T=D*10
#print(T)

A1 = np.block([[P1], [P2]])

A2 = np.block([[S],[S], [M]])
A3=np.block([Z,A2])
A4=np.block([[A1],[A3]])

A=np.block([A4,T])
print(A)




print("\n")
# zad 4
#print( A[1,:] )# 2 wiersz
#print( A[3,:] )# 4 wiersz

B= A[1,:]+A[3,:]
print(B)


print("\n")
# zad 5
#print(np.max(A,0)) # 0 - maksymalna wartosc z każdej kolumny  1 - max z kazdego wiersza
C=np.max(A,0)
print(C)


print("\n")
# zad 6
print(B)
D = np.delete(B, 5, 0)
D = np.delete(D, 0, 0)
print(D)



print("\n")
# zad 7

for i in range(0, np.size(D)):
    if  D[i]==4:
        D[i]=0
    
print(D)




print("\n")
# zad 8
print(C,"\n")
MAX = np.max(C)
MIN = np.min(C)
#print(MIN,"\n")
for i in range(0, np.size(C)):
    if (   C[i]==MIN ): 
        E = np.delete(C,i)

for i in range(0, np.size(E)):
    if ( E[i]==MAX ): 
        E = np.delete(E,i)        

print(E,"\n")




print("\n")
# zad 9
print(A,"\n")
MAX = np.max(A)
MIN = np.min(A)

#print(np.shape(A))
wiersze,kolumny=np.shape(A)
#print( A[0,:] )
print(" pętla: \n")
mi=0.1# mi i ma różne od siebie wartosci inne niż numery wierszy
ma=0.2
for i in range(wiersze):
    for j in range(kolumny):
        if ( A[i,j]==MIN ): 
            mi=i
        if ( A[i,j]==MAX ):
            ma=i
    if(mi==ma):
        print( A[mi,:] )
        




# zad 10
print("\n")
#print(E)
M10 = D@E
print("mnożenie macierzowe",M10 ,"\n")
             
W10 = D*E    
print("mnożenie wektorowe",W10 ,"\n")      
print("\n")




# zad 11
print("\n")

def macierz ():
    s=0
    J=np.ones((3,3))
    #wiersze,kolumny=np.shape(A)
    for i in range(3):
        for j in range(3):
            J[i,j]=np.random.randint(0, 11) #0 włączone 11 nie włączone
    #print(J,"\n")       
    for i in range(len(J)):
        s=s+J[i,i]
    return(J,s)      
         
print(macierz())




# zad 12
print("\n")

#print("k=",k)
def zerowanie (X):
    wielkosc=(len(X))
    for i in range(len(X)):
        for j in range(len(X)):
            if(j== wielkosc - (i+1)):
                X[i,j]=0
    
    for i in range(len(X)):
        X[i,i]=0
    return(X)

mac= np.ones((4,4))
print(zerowanie(mac))





# zad 13
print("\n")
def sumowanie (X):
    suma=0
    for i in range(len(X)):
        
        for j in range(len(X)):
            if (i%2==0):
                suma += X[i,j]

    return(suma)

mac= np.ones((4,4))               
print(sumowanie(mac))    



# zad 14
print("\n")           

f3 = lambda x: np.cos(2*x)
y1=f3(x)
x=np.arange(-10,10,0.1)

plt.plot(x,y1,'r--')


# zad 15
print("\n") 

x=np.arange(-10,10,0.1)
y2=f15.funkcja15(x)
plt.plot(x,y2,'g+')



# zad 17
print("\n") 
y2=f15.funkcja15(x)
plt.plot(x,3*y1,x,y2,'b*')



#zad 18
print("\n") 
A = np.array([[10,5,1,7],
              [10,9,5,5],
              [1,6,7,3],
              [10,0,1,3]])

b = np.array([[34],
              [44],
              [25],
              [27]])

x = np.linalg.solve(A, b)
print(x)
