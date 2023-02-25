"""
Sinkhorn loss function
"""

cuda = torch.device('cuda')
def _squared_distances(x, y) :
    "Returns the matrix of $\|x_i-y_j\|^2$."
    x_col = x.unsqueeze(1) #x.dimshuffle(0, 'x', 1)
    y_lin = y.unsqueeze(0) #y.dimshuffle('x', 0, 1)
    c = torch.sum( torch.abs(x_col - y_lin) , 2)
    return c

def my_sinkhorn(x,y,epsilon,n,niter):
    C=_squared_distances(x, y)
    K=torch.exp(C/epsilon)
    mu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False)
    nu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False)
    actual_nits = 0
    err=0
    
    for i in range(niter):
        mu_1=mu
        mu=(1./n*torch.cuda.FloatTensor(n).fill_(1))/torch.matmul(K,nu)
        nu=(1./n*torch.cuda.FloatTensor(n).fill_(1))/torch.matmul(K,mu)
        err = (mu - mu_1).abs().sum()
        if (err < 1e-1).data.cpu().numpy() :
            
            break
    Gamma=torch.matmul(torch.matmul(mu*torch.eye(n,device=cuda),K),torch.eye(n,device=cuda)*nu)
    cost=torch.sum(Gamma*C)
    return cost
