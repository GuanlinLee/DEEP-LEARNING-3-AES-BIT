import numpy as np

r0=np.loadtxt('shuffle.txt',dtype=np.str,delimiter=",")
#r1=np.loadtxt('shuffle1.txt',dtype=np.str,delimiter=",")

N=8
def inputdata(batchsize,step):
    x=[]
    y=[]
    '''
    for i in range(batchsize//2):
        x0 = ''
        x1 = ''
        for j in range(batchsize):
            x0=x0+r0[N+(batchsize//2*step+i)%(r0.shape[0]-N),0]
        y0=r0[N+(batchsize//2*step+i)%(r0.shape[0]-N),1]

        for j in range(batchsize):
            x1=x1+r1[N+(batchsize//2*step+i)%(r1.shape[0]-N),0]
        y1=r1[N+(batchsize//2*step+i)%(r1.shape[0]-N),1]

        for j in x0:
            x.append(eval(j))
        for j in y0:
            y.append(eval(j))
        for j in x1:
            x.append(eval(j))
        for j in y1:
            y.append(eval(j))
'''
    for i in range(batchsize):
        tempx=r0[N + (batchsize * step +i) % (r0.shape[0] - N), 0]
        tempy=r0[N + (batchsize * step +i) % (r0.shape[0] - N), 1]
        for j in tempx:
            x.append(eval(j))
        for j in tempy:
            y.append((eval(j)))
    x_np=np.array(x).astype(np.float32)
    y_np=np.array(y).astype(np.int32)

    x_np.shape=(batchsize,8)
    y_np.shape=(batchsize)

    return x_np,y_np


def testdata(batchsize,step):
    x = []
    y = []
    '''
    for i in range(batchsize // 2):
        x0 = ''
        x1 = ''
        for j in range(batchsize):
            x0 = x0+r0[ (batchsize // 2 * step + i) , 0]
        y0 = r0[(batchsize // 2 * step + i) , 1]

        for j in range(batchsize):
            x1 = x1+r1[ (batchsize // 2 * step + i) , 0]
        y1 = r1[ (batchsize // 2 * step + i) , 1]

        for j in x0:
            x.append(eval(j))
        for j in y0:
            y.append(eval(j))
        for j in x1:
            x.append(eval(j))
        for j in y1:
            y.append(eval(j))
'''
    for i in range(batchsize):
        tempx = r0[(batchsize * step+i) , 0]
        tempy = r0[(batchsize * step+i), 1]
        for j in tempx:
            x.append((eval(j)))
        for j in tempy:
            y.append(eval(j))
    x_np = np.array(x).astype(np.float32)
    y_np = np.array(y).astype(np.int32)

    x_np.shape = (batchsize, 8)
    y_np.shape = (batchsize)


    return x_np,y_np
