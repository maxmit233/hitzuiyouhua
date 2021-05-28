from scipy.optimize import minimize
import numpy as np

def AugumentedLagrangre(x:np.ndarray,v:np.ndarray,theta,f,h,mu:np.ndarray,epslon,alpha,beta):
    '''
    x: init point ,np.ndarry
    v: lagrangre coefficiency,ndarray
    mu:penalty,np.ndarry
    theta:Augumented Lagrangre function,function
    f:target function,function
    h:constrains,list
    epsilon: exit condition
    alpha: change coefficiency
    beta : change coefficiency
    return xs,vs,hs,fs
    '''
    xs,vs,fs,hs = [],[],[],[]
    sigma1 = max([abs(h_i(x)) for h_i in h])
    xs.append(x.copy());vs.append(v.copy());hs.append([abs(h_i(x)) for h_i in h]);fs.append(f(x))
    while True:
        theta_ = lambda p:theta(p,v.copy(),mu.copy())
        x_i = minimize(theta_,x,method='CG')['x']
        sigma2 = max([abs(h_i(x_i)) for h_i in h])
        xs.append(x_i.copy());vs.append(v.copy());hs.append([abs(h_i(x_i)) for h_i in h]);fs.append(f(x_i))
        if sigma2<epslon:
            return xs,vs,hs,fs
        else:
            v_i= v+2*sum([h_i(x_i)*mu for h_i,mu_i in zip(h,mu)])
            if sigma2>alpha*sigma1:
                for i,h_i in enumerate(h):
                    if abs(h_i(x_i))>alpha*sigma1:
                        mu[i] = beta*mu[i]
            v,x,sigma1=v_i,x_i,sigma2
def Admm(x,z,y,rho,f,g,h,theta,epsilon):
    '''
    x:init_point for f,ndarray
    z:init_point for g,ndarray
    rho:penalty,ndarray
    f:target function
    g:target function
    h:constrains ,list
    theta: augumented lagrangre function
    y:lagrangre coefficient
    epsilon:stop condition
    '''
    xs,zs,fs,gs,rs = [x.copy()],[z.copy()],[f(x)],[g(z)],[[abs(h_i(x,z)) for h_i in h]]
    r = 1000
    while True:
        if abs(r)<=epsilon:
            break
        theta_ = lambda l:theta(l,z.copy(),y.copy(),rho)
        x_i = minimize(theta_,x.copy())['x']
        
        theta_ = lambda l:theta(x_i.copy(),l,y.copy(),rho)
        z_i = minimize(theta_,z)['x']
        
        y_i = y + np.dot(rho,np.array([h_i(x_i,z_i) for h_i in h]))
        
        r = np.abs(np.max(np.dot(rho,np.array([h_i(x_i,z_i) for h_i in h]))))
        x,z,y = x_i,z_i,y_i
        xs.append(x.copy());zs.append(z.copy());fs.append(f(x));gs.append(g(z));rs.append([abs(h_i(x,z)) for h_i in h])
    return xs,zs,fs,gs,rs
def Altest():
    def f(x):
        x1,x2 = x
        return -x1-x2
    def h1(x):
        x1,x2 = x
        return x1**2+x2**2-1
    def theta(x,v,mu):
        v1 = v[0]
        mu1 = mu[0]
        return f(x)+v1*h1(x)+mu1*h1(x)**2
    x = np.array([-0.1,-0.1])
    v = np.array([0])
    mu = np.array([5.5])

    xs,vs,hs,fs = AugumentedLagrangre(x.copy(),v.copy(),theta,f,[h1],mu.copy(),1e-6,0.25,10)
    print("iter x, f(x), h(x)")
    for i,(x,h,fv) in enumerate(zip(xs,hs,fs)):
        print(i,x,fv,h)
    

    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)

    X, Y = np.meshgrid(x, y)
    Z = -X-Y
    plt.gcf().gca().add_artist(plt.Circle((0,0),1,fill=False))
    plt.contour(X, Y, Z, colors='k')
    y1 = np.array(xs)

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='AL')
    plt.legend()
    plt.grid()
    plt.savefig("AL-test")
def AdmmTest():
    def f(x):
        x1 = x[0]
        return -x1
    def g(z):
        z1 = z[0]
        return -z1
    def h1(x,z):
        x1,z1 = x[0],z[0]
        return x1**2+z1**2-1
    def theta(x,z,y,rho):
        return f(x)+g(z)+y[0]*h1(x,z)+rho[0]*h1(x,z)**2
    x = np.array([-1.2])
    z = np.array([-1.2])
    y = np.array([0])
    rho = np.array([5.5])
    xs,zs,fs,gs,rs = Admm(x.copy(),z.copy(),y.copy(),rho.copy(),f,g,[h1],theta,1e-6)
    print("iter ,x ,z ,f(x,z) ,r")
    for i,(x,z,f,g,r) in enumerate(zip(xs,zs,fs,gs,rs)):
        print(i,x,z,f+g,r)
    
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)

    X, Y = np.meshgrid(x, y)
    Z = -X-Y
    plt.gcf().gca().add_artist(plt.Circle((0,0),1,fill=False))
    plt.contour(X, Y, Z, colors='k')
    y1 = np.array([xs,zs])

    plt.plot(xs, zs, '*-', label='admm')
    plt.legend()
    plt.grid()
    plt.savefig("admm-test")
if __name__ =="__main__":
    AdmmTest()
    Altest()