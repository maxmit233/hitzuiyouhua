import numpy as np
from scipy.optimize import  minimize
import  matplotlib.pyplot as plt
import time

# def bisearch(f,gf,alpha,beta,epsilon):
#     '''
#     :param f:函数
#     :param gf: 导数
#     :param alpha: 开始位置
#     :param beta: 结束位置
#     :param epsilon: 精度值
#     :return:
#     '''
#     iter = 1
#     while True:
#         if beta - alpha<= epsilon:
#             break
#         l = (beta+alpha)/2
#         gl = gf(l)
#         if gl == 0:
#             return  l
#         elif gl>0:
#             beta = l
#         else:
#             alpha = l
#         print(iter,alpha,beta)
#         iter += 1
#     return alpha,beta
def bisearch(gf,alpha,beta,epsilon):
    '''
    :param f:函数
    :param gf: 导数
    :param alpha: 开始位置
    :param beta: 结束位置
    :param epsilon: 精度值
    :return:
    '''
    iter = 1
    while True:
        if beta - alpha <= epsilon:
            break
        l = (beta+alpha)/2
        gl = gf(l)
        if gl == 0:
            return l
        elif gl > 0:
            beta = l
        else:
            alpha = l
        # print(iter,alpha,beta)
        iter += 1
    return alpha,beta
def fibonacciMethod(f,alpha,beta,epsilon):
    pre,cur = 5,8
    gamma = pre/cur
    l = -1
    mu = -1
    phi_l = -1
    phi_mu = -1
    zero = True
    iter = 0
    while True:
        if zero:
            l = alpha + (1-gamma)*(beta-alpha)
            mu = alpha + gamma*(beta - alpha)
            phi_l =f(l)
            phi_mu = f(mu)
            zero = False
        if beta - alpha<=epsilon:
            break
        if phi_l-phi_mu == 0:
            alpha = l
            beta = mu
            zero = True
        elif phi_l-phi_mu >0:
            alpha = l
            l = mu
            mu = alpha + gamma*(beta - alpha)
            phi_l = phi_mu
            phi_mu = f(mu)
        else:
            beta = mu
            mu = l
            l = alpha + (1-gamma)*(beta - alpha)
            phi_mu = phi_l
            phi_l = f(l)
        iter+=1
        pre,cur = cur,pre+cur
        gamma = pre/cur
        print(iter,alpha,beta)
    return [alpha,beta]
def goldenMethod(f,alpha,beta,epsilon):
    gamma = (np.sqrt(5)-1)/2
    l = -1
    mu = -1
    phi_l = -1
    phi_mu = -1
    zero = True
    iter = 0
    while True:
        if zero:
            l = alpha + (1-gamma)*(beta-alpha)
            mu = alpha + gamma*(beta - alpha)
            phi_l =f(l)
            phi_mu = f(mu)
            zero = False
        if beta - alpha<=epsilon:
            break
        if phi_l-phi_mu == 0:
            alpha = l
            beta = mu
            zero = True
        elif phi_l-phi_mu >0:
            alpha = l
            l = mu
            mu = alpha + gamma*(beta - alpha)
            phi_l = phi_mu
            phi_mu = f(mu)
        else:
            beta = mu
            mu = l
            l = alpha + (1-gamma)*(beta - alpha)
            phi_mu = phi_l
            phi_l = f(l)
        iter+=1
        print(iter,alpha,beta)
    return [alpha,beta]
def goldstein(f,gf,alpha,beta,rho,direct,init_point):
    phi1 = f(init_point)
    phi1g = np.dot(gf(init_point).T,direct)
    iter = 0
    l = 20
    while True:
        iter+=1
        phi2 = f(init_point+l*direct)
        if phi2<= phi1 + rho*phi1g*l:
            if phi2 >= phi1 + (1-rho)*phi1g*l:
                print(iter,l)
                return l
            else:
                l = alpha*l
        else:
            l = beta*l
        print(iter,l)
def wolfepowell(f,gf,rho,sigma,alpha,beta,direct,init_point):
    l = 30
    phi1 = f(init_point)
    phi1g = np.dot(gf(init_point).T,direct)
    iter = 0
    while True:
        iter+=1
        phi2 = f(init_point+l*direct)
        if phi2<=phi1+rho*phi1g*l:
            phi2g = np.dot(gf(init_point+l*direct).T,direct)
            if phi2g>=sigma*phi1g*l:
                print(iter,l)
                return l
            else:
                l = alpha*l
        else:
            l = beta*l
        if iter>500:
            break
        print(iter,l)
    return l
def SGDMomentum(gf,f,x,lr,epsilon,alpha,beta):
    v = 0
    iter = 0
    steps = [x]
    while True:
        if np.linalg.norm(gf(x),ord=2)<=epsilon:
            break
        v = beta*v - alpha*gf(x)
        x = x + v
        steps.append(x)
        iter += 1
        print(iter, x.reshape((-1)),f(x))
    return x,steps
def BFGS(x,epsilon,gf,lgf,f):
    '''
    :param x: 初始点
    :param epsilon:
    :param gf:
    :return:
    '''
    H = np.eye(x.shape[0])
    s = np.array([100,100])
    iter = 0
    steps = [x.reshape((-1))]
    while True:
        if np.linalg.norm(gf(x),ord=2)<=epsilon:
            break
        d = -np.dot(H,gf(x))
        l = wolfepowell(f,gf,0.3,0.7,1.5,0.5,d.copy(),x.copy())
        # l = goldstein(f,gf,1.5,0.5,0.4,d.copy(),x.copy())
        # lf = lambda l:lgf(x.copy(),d.copy(),l)
        # alpha,beta = bisearch(lf,0,200,1e-3)
        # l = (alpha+beta)/2
        if l<1e-18:
            break
        x_n = x+l*d
        y = gf(x_n)-gf(x)
        s = x_n-x
        H = H + (1+ np.dot(y.T,np.dot(H,y))/np.dot(s.T,y))*(np.dot(s,s.T)/np.dot(s.T,y))
        H -= (np.dot(np.dot(s,y.T),H)+np.dot(H,np.dot(y,s.T)))/np.dot(s.T,y)
        x = x_n
        iter += 1
        print(iter,x.reshape((1,-1)),f(x))
        steps.append(x.reshape((-1)))
    return x,steps
def DFP(x,epsilon,gf,lgf,f):
    '''
    :param x: 初始点
    :param epsilon:
    :param gf:
    :return:
    '''
    H = np.eye(x.shape[0])
    s = np.array([100,100])
    iter = 0
    steps = [x.reshape((-1))]
    while True:
        if np.linalg.norm(gf(x),ord=2)<=epsilon:
            break
        d = -np.dot(H,gf(x))
        l = wolfepowell(f,gf,0.3,0.9,1.5,0.5,d.copy(),x.copy())
        if l<1e-18:
            break
        x_n = x+l*d
        y = gf(x_n)-gf(x)
        s = x_n-x
        H = H + np.dot(s,s.T)/np.dot(s.T,y) - np.dot(np.dot(H,y),np.dot(y.T,H))/np.dot(y.T,np.dot(H,y))
        x = x_n
        iter += 1
        steps.append(x.reshape((-1)))
        print(iter,x.reshape((1,-1)),f(x))
    return x,steps
def FRCG(x,epsilon,gf,lgf,f):
    d = None
    iter = 0
    steps = [x.reshape((-1))]
    while True:
        if np.linalg.norm(gf(x),ord=2)<=epsilon:
            break
        if iter == 0:
            d = - gf(x)
        l = wolfepowell(f,gf,0.3,0.7,1.5,0.5,d.copy(),x.copy())
        x_n = x + l * d

        deltaf = gf(x)
        deltaf_n = gf(x_n)
        beta = np.dot(deltaf_n.T,deltaf_n)/np.dot(deltaf.T,deltaf)
        d = -deltaf_n + beta*d

        x = x_n
        iter += 1
        steps.append(x.reshape((-1)))
        print(iter,x.reshape((1,-1)),f(x))
    return x,steps
def test1():
    '''
    用于实验一-无导数法
    '''
    def f1(x):
        return 2*x**2 - x - 1
    def gf1(x):
        return 4*x-1
    print("start f1")
    # f1黄金分割
    print("f1----黄金分割")
    v = goldenMethod(f1,-1,1,0.06)
    print("区间为",v)
    print("自变量",(v[1]+v[0])/2,"函数值",f1((v[1]+v[0])/2))
    # f1 fibonacci法
    print("f1----fibonacci")
    v = fibonacciMethod(f1,-1,1,0.06)
    print("区间为", v)
    print("自变量", (v[1] + v[0]) / 2, "函数值", f1((v[1] + v[0]) / 2))
def test2():
    '''
    用于实验一-有导数法-线搜索法
    '''
    def f(x:np.ndarray):
        return 100*(x[1]-x[0]**2)**2 +(1-x[0])**2
    def gf(x:np.ndarray):
        v = [-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)]
        return np.array(v)
    # Goldstein
    x = np.array([1,1])
    d = np.array([-1,1])
    print("Goldstein")
    l = goldstein(f,gf,1.5,0.5,0.3,x.copy(),d.copy())
    print(l*d+x,f(l*d+x))
    # wolfe-powell
    print("Wolfe-Powell")
    l = wolfepowell(f,gf,0.3,0.7,1.5,0.5,d.copy(),x.copy())
    print(l*d+x,f(l*d+x))
    #二分法
    print("二分法")
    def bf(l):
        return 100*(-l**2+3*l)**2+l**2-4*l+4
    def bgf(l):
        return 200*(-l**2+3*l)*(-2*l+3)+2*l-4
    alpha,beta = bisearch(bf,bgf,0,1,1e-10)
    l = (alpha+beta)/2
    print(l*d+x,f(l*d+x))
def test3():
    '''
    for SGDMomentum
    '''
    def f(x:np.ndarray):
        return x[0]**2+x[1]**2-x[0]*x[1]-10*x[0]-4*x[1]+60
    def gf(x:np.ndarray):
        v = [2*x[0]-x[1]-10,2*x[1]-x[0]-4]
        return np.array(v)
    x = np.array([-1,1])
    y1 = SGDMomentum(gf,f,x,0.01,0.06,0.4,0.5)
    print(y1[0],f(y1[0]))
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 10, 100)
    y = np.linspace(-1, 10, 100)

    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2 - X * Y - 10 * X - 4 * Y + 60

    plt.contour(X, Y, Z, colors='k')
    y1 = np.array(y1[1])

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='SGDMomentum')
    plt.legend()
    plt.savefig("SGDMomentum")
def test4():
    '''
    FRCG、DFP、BFGS
    '''
    def f(x):
        x1 = x[0,0]
        x2 = x[1,0]
        return x1**2+x2**2-x1*x2-10*x1-4*x2+60
    def gf(x):
        x1 = x[0,0]
        x2 = x[1,0]
        return np.array([[2*x1-x2-10],[2*x2-x1-4]])
    def lgf(x,d,l):
        x1 = x[0, 0]
        x2 = x[1, 0]
        d1 = d[0,0]
        d2 = d[1,0]
        return (2*d2-d1)*(x2+l*d2)+(2*d1-d2)*(x1+l*d1)-10*d1-4*d2
    x = np.array([[0],[0]])
    print("DFP-------start")
    y1 = DFP(x.copy(),0.06,gf,lgf,f)
    print("寻找到的最小点为",y1[0].reshape((1,-1)))
    print("最小点的值为",f(y1[0]))

    print("BFGS-------start")
    y2 = BFGS(x.copy(), 0.06, gf, lgf,f)
    print("寻找到的最小点为", y2[0].reshape((1,-1)))
    print("最小点的值为", f(y2[0]))

    print("FRCG-------start")
    y3 = FRCG(x.copy(), 1e-6, gf, lgf,f)
    print("寻找到的最小点为", y3[0].reshape((1,-1)))
    print("最小点的值为", f(y3[0]))

    import matplotlib.pyplot as plt

    x = np.linspace(-1,10,100)
    y = np.linspace(-1,10,100)

    X,Y = np.meshgrid(x,y)
    Z = X**2+Y**2-X*Y-10*X-4*Y+60

    plt.contour(X,Y,Z,colors='k')
    y1 = np.array(y1[1])
    y2 = np.array(y2[1])
    y3 = np.array(y3[1])

    # plt.plot(y1[:,0],y1[:,1],'*-',label = 'DFP')
    # plt.plot(y2[:,0],y2[:,1],'*-',label ='BFGS')
    plt.plot(y3[:,0],y3[:,1],'*-',label = 'FRCG')
    plt.legend()
    plt.savefig('DFP-BFGS-FRCG')
def test5():
    '''
    for Rosenbrock's 函数-SGD
    '''
    def f(x):
        x1,x2 = x[0],x[1]
        return (1-x1)**2 + 100*(x2-x1**2)**2
    def gf(x):
        x1,x2 = x[0],x[1]
        v = [-2*(1-x1)+200*(x2-x1**2)*-2*x1,200*(x2-x1**2)]
        return np.array(v)
    x = np.array([0,-1])
    y1 = SGDMomentum(gf,f,x.copy(),0,0.06,1e-3,0)

    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)

    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2+100*(Y-X**2)**2
    a = plt.contourf(X,Y,Z,cmap=plt.cm.Spectral)
    b = plt.contour(X, Y, Z, colors='k',levels=15)
    plt.colorbar(a)
    y1 = np.array(y1[1])

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='SGDMomentum')
    plt.legend()
    plt.savefig("SGDMomentum-Rosenbrock")
def test6():
    '''
    for Rosenbrock's 函数-FRCG
    '''
    def f(x):
        x1,x2 = x[0],x[1]
        return (1-x1)**2 + 100*(x2-x1**2)**2
    def gf(x):
        x1,x2 = x[0],x[1]
        v = [-2*(1-x1)+200*(x2-x1**2)*-2*x1,200*(x2-x1**2)]
        return np.array(v)
    def lgf(x,d,l):
        x1,x2 = x[0],x[1]
        d1,d2 = d[0],d[1]
        l1 = -2*d1*(1-(x1+l*d1))
        l2 = 200*((x2+l*d2)-(x1+l*d1)**2)*(d2-2*d1*(x1+l*d1))
        return l1+l2
    x = np.array([0,-1])
    y1 = FRCG(x.copy(),1e-3,gf,lgf,f)

    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)

    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2+100*(Y-X**2)**2
    a = plt.contourf(X,Y,Z,cmap=plt.cm.Spectral)
    b = plt.contour(X, Y, Z, colors='k',levels=15)
    plt.colorbar(a)
    y1 = np.array(y1[1])

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='FRCG')
    plt.legend()
    plt.savefig("FRCG-Rosenbrock")
def test7():
    '''
    for Rosenbrock's 函数-DFP
    '''
    def f(x):
        x1,x2 = x[0],x[1]
        return (1-x1)**2 + 100*(x2-x1**2)**2
    def gf(x):
        x1,x2 = x[0],x[1]
        v = [-2*(1-x1)+200*(x2-x1**2)*-2*x1,200*(x2-x1**2)]
        return np.array(v)
    def lgf(x,d,l):
        x1,x2 = x[0],x[1]
        d1,d2 = d[0],d[1]
        l1 = -2*d1*(1-(x1+l*d1))
        l2 = 200*((x2+l*d2)-(x1+l*d1)**2)*(d2-2*d1*(x1+l*d1))
        return l1+l2
    x = np.array([0,-1])
    y1 = DFP(x,1e-3,gf,lgf,f)

    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)

    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2+100*(Y-X**2)**2
    a = plt.contourf(X,Y,Z,cmap=plt.cm.Spectral)
    b = plt.contour(X, Y, Z, colors='k',levels=15)
    plt.colorbar(a)
    y1 = np.array(y1[1])

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='DFP')
    plt.legend()
    plt.savefig("DFP-Rosenbrock")
def test8():
    '''
    for Rosenbrock's 函数-BFGS
    '''
    def f(x):
        x1,x2 = x[0],x[1]
        return (1-x1)**2 + 100*(x2-x1**2)**2
    def gf(x):
        x1,x2 = x[0],x[1]
        v = [-2*(1-x1)+200*(x2-x1**2)*-2*x1,200*(x2-x1**2)]
        return np.array(v)
    def lgf(x,d,l):
        x1,x2 = x[0],x[1]
        d1,d2 = d[0],d[1]
        l1 = -2*d1*(1-(x1+l*d1))
        l2 = 200*((x2+l*d2)-(x1+l*d1)**2)*(d2-2*d1*(x1+l*d1))
        return l1+l2
    x = np.array([0,-1])
    y1 = BFGS(x,1e-3,gf,lgf,f)
    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)

    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2+100*(Y-X**2)**2
    a = plt.contourf(X,Y,Z,cmap=plt.cm.Spectral)
    b = plt.contour(X, Y, Z, colors='k',levels=15)
    plt.colorbar(a)
    y1 = np.array(y1[1])

    plt.plot(y1[:, 0], y1[:, 1], '*-', label='BFGS')
    plt.legend()
    plt.savefig("BFGS-Rosenbrock")
if __name__ == '__main__':
    test8()