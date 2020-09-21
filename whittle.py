import numpy as np
import matplotlib.pyplot as plt
import logging as log 
log.basicConfig(level=log.CRITICAL)

# ## Exp 1
# initializing transition matrix
dim = 4
p0 = np.diag([1,1,1,1])
for i in range(4):
    p0[np.mod(i+1,4),i] = 1
p0 = 0.5*p0
p1 = np.transpose(p0)
P = np.array([p0,p1])
r = np.transpose([[-1,0,0,1],[-1,0,0,1]]) # reward [R0,R1]
# n,m = 100,20  # total arm, active arm

# ## Exp 2
# dim = 5
# p1 = np.zeros((dim,dim))
# p1[:,0]=1
# p0 = p1/10 
# r = np.zeros((dim,2))
# for i in range(dim):
#     p0[i,min(i+1,4)] = 9/10
#     r[i][0] = 0.9**(i+1)

n,m = 100,20
P = np.array([p0,p1])
X = np.random.randint(0,dim,100) # initializing states
w = np.array([-0.5,0.5,1,-1]) #np.zeros(dim) # initializing whittle index np.array([-0.9,-0.73,-0.5,-0.26,-0.01])#
Q =np.array([r for i in range(dim)], dtype=float)  # initializing Q(x,i,u) = R(i,u)
log.info('Reward \n{} \n\n Transition matrix \n{}, \n\n total arm {}, active arm {} \n \n'.format(r,P,n,m))
log.info('Initial whittle index \n{} \n\n Initial Q \n{} \n\n Initial state \n{}'.format(w,Q,X))

class pm:
    P,r,w,X,n,m,Q,dim = P,r,w,X,n,m,Q,dim
    
class Whittle():
    def __init__(self):
        self.P, self.r, self.w,self.X,self.n,self.m,self.Q,self.dim = pm.P,pm.r,pm.w,pm.X,pm.n,pm.m,pm.Q,pm.dim
        self.t, self.e, self.dim= [1,1], 0.1,self.P[0].shape[0]
        self.v = np.ones((self.dim,2)) # local clock, v(i,u)
        self.A = np.random.randint(0,1,self.n) # list of action for arms
        self.Q_shape = self.Q.shape  # delta for Q 
        self.X_ = self.X # previous state, bookeeping
        self.choose_matrix = np.tril(np.ones((self.dim,self.dim)))-0.5*np.identity(self.dim)
    def action(self):
        self.A = np.random.randint(0,1,self.n) # list of action for arms
        self.toss = np.random.rand()
        self.index = np.arange(self.n)
        if self.toss < self.e:
            log.debug('\n Uniform exploration, toss {}, episilon {} \n'.format(self.toss, self.e))
            np.random.shuffle(self.index)
        else:
            log.debug('\n Exploitation, toss {}, episilon {} \n'.format(self.toss, self.e))
            self.W = [(self.w[x],i) for x,i in zip(self.X,range(self.n))] # [(w0, 0),(w1,1)..]
            self.W = np.array(self.W,dtype=[('whittle',float),('index',int)])
            self.W = np.sort(self.W, order=('whittle','index'))
            self.index = self.W['index']   #sorted whittle index
        self.A[self.index[-self.m:]] = 1
        log.info('(index,whittle,action) \n {}'.format(list(zip(self.W['index'],self.W['whittle'],self.A))))
        
    
    def evolve(self):
        self.X_ = self.X  # storing previous state
        self.X = np.array([np.random.choice(list(range(dim)),p=self.P[a][x]) for a,x in zip(self.A,self.X)]) # next state
        self.R = np.array([self.r[x_,a] for x_,a in zip(self.X_,self.A)]) # reward list
        log.info('\n\n (previous state,next state,action,reward) \n{}'.format(list(zip(self.X_,self.X,self.A,self.R))))
        
        
    def update_w(self):
        log.debug('\n Q(x_,x,a) value \n {}\n'.format(self.Q))
        self.dw = np.diag(self.Q[:,:,1]) - np.diag(self.Q[:,:,0])  # Q^x(x,1)-Q^x(x,0)
        self.w = self.w + 0.01*self.dw  # w --> w + y(t)[Q^x(x,1)-Q^x(x,0)]
        log.info('\n whittle value (w,dw) \n {}\n'.format(list(zip(self.w,self.dw))))
        self.t[0] += 1    
    
    def update_Q(self):
        self.f = np.mean(self.Q,axis=(1,2))
        self.dQ = np.zeros(self.Q_shape,dtype=float)  # initializing delta for Q 
        for x_,x,a,r in zip(self.X_,self.X,self.A,self.R): 
            self.v[x_,a] +=1
            self.dQ[:,x_,a]+=((1-a)*self.w+r+np.max(self.Q[:,x],axis=1)-self.f-self.Q[:,x_,a])/self.v[x_,a]
        self.Q += self.dQ
        
W = Whittle()  
D = []
R = []
def run(n=1000):      
    for i in range(n):
        W.action()
        W.evolve()
        W.update_Q()
        #W.update_w()
        D.append(W.w)
        temp = np.mean(W.R)
        R.append(temp)
        print(W.w,temp)        
            
    # plt.plot(D)
    # plt.title('Whittle index for Exp1')
    # plt.ylabel('whittle values')
    # plt.xlabel('n')
    # plt.savefig('whittle_exp1.png')
    # plt.close()
    

    plt.plot(R)
    plt.title('Average reward for Exp1 with exact indice')
    plt.ylabel('reward')
    plt.xlabel('n')
    plt.savefig('reward_exp1_exac.png')
    plt.close()
    