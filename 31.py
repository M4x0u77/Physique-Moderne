def derivee(f,dx):
    """fonction qui renvoie la dérivée"""
    npts=len(f)
    df=[]
    if(npts>=2):
        for i in range(npts):
            df.append(0)
        df[0]=(f[1]-f[0])/dx
        for i in range(1,npts-1):
            df[i]=(f[i+1]-f[i-1])/(2*dx)
        df[npts-1]=(f[npts-1]-f[npts-2])/dx
    return df

def carre(x):
    """fonction qui renvoie le carré"""
    return x**2

def double(x):
    """fonction qui renvoie le double"""
    return 2*x

npts=100
xmin=1.0
xmax=10.0
dx=(xmax-xmin)/(npts-1)
x=[]
for i in range(npts):
    x.append(xmin+i*dx)
f=[]
for val in x:
    f.append(carre(val))

df_correct=[]
for val in x:
    df_correct.append(double(val))
df_numerique = derivee(f, dx)

erreurs=[]
for i in range(npts):
    erreurs.append(abs((df_numerique[i]-df_correct[i])/df_correct[i]))
moyenne_erreur=0
for i in range(npts):
    moyenne_erreur=moyenne_erreur+erreurs[i]
moyenne_erreur=moyenne_erreur/npts
print("L'erreur relative moyenne de l'algorithme est de : ",moyenne_erreur)
