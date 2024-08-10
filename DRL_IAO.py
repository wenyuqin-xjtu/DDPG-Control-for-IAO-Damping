import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import scipy
from scipy import linalg as la 
from scipy.integrate import odeint
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import xlrd
import os
import copy
from torch.nn.parameter import Parameter
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
from operator import itemgetter

# 更改这个变量 自动创建 ./image/1101_1/ 文件夹
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
ORI_NAME = '2023_9_18_7'
for i in range(1000):
    NAME=ORI_NAME+'_'+str(i)
    if not os.path.exists('./image/'+NAME):
        break
np.random.seed(1)
# tf.set_random_seed(1)
select=[1,2,3,5,6,7,9]
####################  hyper parameters  ####################
R_threshold=0.035
state_dim =75
action_dim =28 # 4,5,6,7 local(4*4)+nonlocal

max_t=0.02
timestep=2
t=np.linspace(0,max_t,timestep)

a_b=0.035  # action boundary

maxepisodes = 5000
maxsteps = 500
LR_A = 0.00001    # learning rate for actor
LR_C = 0.000001   # learning rate for critic
GAMMA = 0.95       # reward discount

workbook=xlrd.open_workbook('./data/A.xls')
sheet_name=workbook.sheet_names()[0]
sheet=workbook.sheet_by_index(0)
A=np.zeros((75,75))
for i in range(75):
    for j in range(75):
        A[i,j]=sheet.cell_value(i,j)

workbook=xlrd.open_workbook('./data/B1.xls')
sheet_name=workbook.sheet_names()[0]
sheet=workbook.sheet_by_index(0)
B1=np.zeros((75,9))
for i in range(75):
    for j in range(9):
        B1[i,j]=sheet.cell_value(i,j)

workbook=xlrd.open_workbook('./data/B2.xls')
sheet_name=workbook.sheet_names()[0]
sheet=workbook.sheet_by_index(0)
B2=np.zeros((75,9))
for i in range(75):
    for j in range(9):
        B2[i,j]=sheet.cell_value(i,j)

workbook=xlrd.open_workbook('./data/Q.xls')
sheet_name=workbook.sheet_names()[0]
sheet=workbook.sheet_by_index(0)
Q=np.zeros((75,75))
for i in range(75):
    for j in range(75):
        Q[i,j]=sheet.cell_value(i,j)

Q = tf.convert_to_tensor(Q)

workbook=xlrd.open_workbook('./data/R.xls')
sheet_name=workbook.sheet_names()[0]
sheet=workbook.sheet_by_index(0)
T=np.zeros((9,9))
for i in range(9):
    for j in range(9):
        T[i,j]=sheet.cell_value(i,j)

# T = tf.convert_to_tensor(T)


noise=np.zeros((9))
for o in range(9):
    noise[o]=random.uniform(-0.0001,0.0001)#random noise


REPLACEMENT = [
    dict(name='soft', tau=0.00001),
    dict(name='hard', rep_iter_a=2000, rep_iter_c=1500)
][0]            # you can try different target replacement strategies origin set: 600,500
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

RENDER = False
OUTPUT_GRAPH = True

os.makedirs('./image/'+NAME)
os.makedirs('./result/'+NAME)
os.makedirs('./model/'+NAME)
#################################### ODE ##################################
N=75
def dmove(t,y,AA,BB1,BB2,nn,aa):
    X = np.zeros((N))

    a1 = np.zeros((75))
    for q in range(75):
        for w in range(75):
            a1[q] += AA[q][w]*y[w]

    b1=np.dot(BB1,nn)
    b2=np.dot(BB2,aa)    
    X=a1+b1+b2

    return X

#############################
def mkdir(path_name):
    import os
    isExists = os.path.exists(path_name)
    if not isExists:
        os.makedirs(path_name)
        return True
    else:
        return False


def process_state(state,g):
    if len(state.shape) == 1:
        state = np.expand_dims(state, axis=0)
    s_s = state.shape # [(75,) 
    nstate=np.zeros((s_s[0],75,16)) 
    for j in range(s_s[0]):
        for i in range(7,75):
            nstate[j][i]=state[j][i]*g[int((i+1)/8)]
        for i in range(7):
            nstate[j][i]=state[j][i]*g[0]
    return nstate





def ocs_cal(theta, w):
    osc = 0
    for i in range(theta.shape[0]):
        for j in range(theta.shape[0]):
            osc += (np.power(theta[i]-theta[j], 2))
    for i in range(w.shape[0]):
        osc += (np.power(w[i], 2))
    return osc


def get_X():
    df = pd.read_csv('./data/data1.csv')
    res = df.values.tolist()
    for i in range(len(res)):
        res[i][2] = dict({'weight': res[i][2]})
    res = [tuple(x) for x in res]
    g = nx.Graph()
    g.add_edges_from(res)
    G = g
    node2vec = Node2Vec(G, 
                        dimensions=16,  # 嵌入维度
                        p=4,            # 回家参数
                        q=2,            # 外出参数
                        walk_length=10, # 随机游走最大长度
                        num_walks=800,  # 每个节点作为起始节点生成的随机游走个数
                        workers=4       # 并行线程数
                       )
    
    model = node2vec.fit(window=3,     # Skip-Gram窗口大小
                         min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                         batch_words=4 # 每个线程处理的数据量
                        )
    X = model.wv.vectors
    node_name = np.reshape(g.nodes, (len(g.nodes),1))
    X =np.hstack((node_name,X))
    arr = list(sorted(X, key=itemgetter(0)))
    arr = np.array([list(x) for x in arr])
    result=arr[:,1:]
    return result

node2vec =pd.read_excel('./data/node2vec.xlsx',index_col=0)
node2vec=np.array(node2vec)

data_node2vec = pd.DataFrame(node2vec)

writer = pd.ExcelWriter('./image/'+NAME+'/node2vec.xlsx')
data_node2vec.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()
G=np.zeros((10,16))
for i in range (10):
    G[i]=node2vec[i+29]


###############################  Actor  ####################################

class Actor(nn.Module):
    def __init__(self,action_bound, replacement, name, alpha = LR_A, opt_a = optim.Adam, act_fn = F.relu,
                 chkpt_dir='./model'):
        """

        :param state_dim: 状态维度（输入维度）
        :param action_dim: 动作维度 （输出维度）
        :param action_bound: 动作范围
        :param replacement:
        :param name: 网络名称（保存名称）
        :param net_arch: 网络参数
        :param alpha: Actor网络学习率
        :param opt_a: Actor网络优化器
        :param act_fn: 激活函数
        :param chkpt_dir: 保存路径
        """
        super(Actor, self).__init__()

        self.replacement = replacement
        self.lr = alpha
        self.t_replace_counter = 0
        self.act_fn = act_fn
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_')

        self.conv1 = nn.Linear(75*16,50*16)
        self.conv1.weight.data.normal_(0,0.1) # initialization
        self.conv2 = nn.Linear(50*16,20*16)
        self.conv2.weight.data.normal_(0,0.1) # initialization
        self.conv3.weight.data.normal_(0,0.1) # initialization
        self.conv3 = nn.Linear(20*16, 10*16)
        self.mu = nn.Linear(10*16, action_dim)
        self.conv3.weight.data.normal_(0,0.1) # initialization
        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = opt_a(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_bound = torch.tensor(action_bound, dtype=torch.float).to(self.device)

        self.to(self.device)

    def forward(self,  state):
        bs = state.shape[0]  # 1 为同时处理的图张数（时域）
        s_value = self.conv1(state)
        s_value=torch.relu(s_value)
        s_value = self.conv2(s_value)
        s_value=torch.relu(s_value)
        s_value = self.conv3(s_value)
        s_value=torch.relu(s_value)
        action = torch.tanh(self.mu(s_value))
        # print(action.shape) # [bs, 16]
        # action = torch.mean(s_value)
        # print(action.shape)
        action = torch.mul(action, self.action_bound)
        # print(action.shape)
        return action


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

###############################  Critic  ####################################

class Critic(nn.Module):   # ae(s)=a
    def __init__(self,  replacement, name, 
                 chkpt_dir='./model'):
        super(Critic,self).__init__()
        self.fcs = nn.Linear(75,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(action_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.replacement = replacement
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value
    
    def save_checkpoint(self):# save parameters during training
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims, is_load=True):
        if is_load:
            (capacity, data, pointer) = self.load()
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.p = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)

        # 调试程序，设为按顺序取样
        # indices = np.array([x for x in range(self.p, self.p+n)])
        # self.p = (self.p + n) % self.capacity

        return self.data[indices, :]

    def save(self, filename='Memory_0728'):
        np.savez('./data/' + filename + '.npz', data=self.data, capacity=self.capacity, pointer=self.pointer)

    def load(self, filename='Memory_0728'):
        f = np.load('./data/' + filename + '.npz')
        return (f['capacity'], f['data'], f['pointer'])

###############################  Agent  ####################################

class Agent(object):
    def __init__(self, gamma, tau, AB, name=NAME, chkpt_dir='./model/', Is_predict=False):
        super(Agent, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1, is_load=False)
        action_bound = AB
        self.name = name
        mkdir('./model/'+name)
        mkdir('./result/'+name)
        mkdir('./image/'+name)

        self.actor = Actor( action_bound, REPLACEMENT, name='Actor' + name, chkpt_dir=chkpt_dir+name+'/')
        # state_dim = 15  action_dim = 16 (output)

        # print(self.actor.parameters())

        self.critic = Critic( REPLACEMENT, name='Critic' + name, chkpt_dir=chkpt_dir+name+'/')
        # state_dim = 15  action_dim = 4 (input)

        # print(self.critic.parameters())

        self.target_actor = Actor(action_bound, REPLACEMENT, name='TargetActor' + name, chkpt_dir=chkpt_dir+name+'/')
        self.target_critic = Critic(REPLACEMENT, name='TargetCritic' + name,  chkpt_dir=chkpt_dir+name+'/')
        self.atrain = torch.optim.Adam(self.actor.parameters(),lr=LR_A)
        self.ctrain = torch.optim.Adam(self.critic.parameters(),lr=LR_C)


        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        if Is_predict:
            self.load_models()

####################  main  ######################

    def adjust(self, AB,MAX_EPISODES,MAX_EP_STEPS):
        action_bound = AB
        successtime=0
        series_ang=[0,7,15,23,31,39,47,55,63,71]
        number=0
        var = a_b*0.35  # control exploration

        xx=range(0,MAX_EPISODES)
        length=int(timestep*MAX_EP_STEPS)
        xt=np.zeros((length))
        for lll in range(length):
            xt[lll]=(max_t/timestep)*lll

        yy=np.zeros((MAX_EPISODES))

        p=0
        sr=0
        srr=0
        tt=0

        # dis = tf.dtypes.cast(dis, tf.float64)

        un=np.zeros((9))
        xxxx=np.zeros((MAX_EP_STEPS))
        zs=np.zeros((18,length))
        G_10=np.zeros((2,length))
        a=np.zeros((action_dim))
        u=np.zeros((action_dim))
        s=np.zeros((75))
        s_=np.zeros((75))
        for llll in range(MAX_EP_STEPS):
            xxxx[llll]=max_t*llll
        showx=np.zeros((maxsteps*timestep))
        showg=np.zeros((18,maxsteps*timestep))
        show_Gw=np.zeros((9,maxsteps))
        show_count=np.zeros((maxsteps))
        nshowx=np.zeros((maxsteps*timestep))
        nshowg=np.zeros((75,maxsteps*timestep))
        nshow_Gw=np.zeros((9,maxsteps))
        nshow_count=np.zeros((maxsteps))
        showg=np.zeros((18,maxsteps*timestep))
        show_Gw=np.zeros((9,maxsteps))
        show_count=np.zeros((maxsteps))
        uu=np.zeros((action_dim))
        temp=np.zeros((9,75))
        shown=np.zeros((maxsteps))
        rc = np.zeros((maxsteps*timestep))
    # rn=np.zeros((maxsteps)) ####
        rn = np.zeros((maxsteps * timestep))

        for ttttt in range(maxsteps):
            shown[ttttt]=ttttt+1

        for ttttt in range(maxsteps*timestep):
            showx[ttttt] = (max_t/timestep)*ttttt

        for ttttt in range(maxsteps*timestep):
            nshowx[ttttt] = (max_t/timestep)*ttttt


        for i in range(MAX_EPISODES):
            uuu=0
            ang=0
            anv=0
            wad=0

            p+=1
            er=0
            ep_reward = 0
            s=np.zeros((75))
            s_=np.zeros((75))
            sta=np.zeros((state_dim))
            sta_=np.zeros((state_dim))

            for koo in range(9):
                if koo==0:
                    #s[0]=0.6
                    s[1]=-0.1*2*3.14+random.gauss(0,0.06)
                else:
                    s[8*koo]=-0.1*2*3.14-0.004*koo*3.14+random.gauss(0,0.06)
            # #s[32]=3.14
            s[72]=0.2*2*3.14+random.gauss(0,0.4)

            avs1=0
            avs2=0
            avs1=s[0]
            avs2=s[1]

            for ooo in range(9):
                if abs(s[8*ooo-1])>abs(avs1):
                    avs1=s[8*ooo-1]
                if abs(s[8*ooo])>abs(avs2):
                    avs2=s[8*ooo]


            for j in range(int(maxsteps)):
                # print('j=', j)

                s_o=s
                s = process_state(s,G)
                s=s.reshape(1,75*16)
                s=torch.from_numpy(s).to(torch.float32)
                s = s.unsqueeze(0)
                a = self.choose_action(s)
                # print('****************************************************')
                # print(a)
                # print(a.shape)
                a = np.clip(np.random.normal(a, var), -AB, AB)  # add randomness to action selection for exploration
                # print('****************************************************')
                # print(a)
                # print(a.shape)
                u=a

                cnt=0
                for ls in range(9):
                    if ls!=0 and ((ls+1) in select):
                        temp[ls,8*ls+1-1]=copy.deepcopy(u[4*cnt])
                        temp[ls,8*ls+2-1]=copy.deepcopy(u[4*cnt+1])
                        temp[ls,8*ls+4-1]=copy.deepcopy(u[4*cnt+2])
                        temp[ls,8*ls+7-1]=copy.deepcopy(u[4*cnt+3])    
                        cnt+=1
                    elif ls==0 and ((ls+1) in select) :
                        temp[0,1]=copy.deepcopy(u[0])
                        temp[0,2]=copy.deepcopy(u[1])
                        temp[0,3]=copy.deepcopy(u[2])
                        temp[0,6]=copy.deepcopy(u[3]) 

                uu=-np.dot(temp,s_o)

                result=odeint(dmove,y0=s_o,t=t,tfirst=True, args=(A,B1,B2,noise,uu))
                # print(result)
                # select angular_velocity and phase from the original data
                if i%(100) == 0:
                    for sss in range(9):
                        for zzz in range(timestep):
                            if sss==0:
                                zs[0,timestep*j+zzz]=result[zzz,0]-result[zzz,71]
                            else:
                                zs[sss,timestep*j+zzz]=result[zzz,8*sss-1]-result[zzz,71]

                    for sss in range(9):
                        for zzz in range(timestep):
                            if sss==0:
                                zs[9,timestep*j+zzz]=result[zzz,1]
                            else:
                                zs[9+sss,timestep*j+zzz]=result[zzz,8*sss]

                    for zzz in range(timestep):
                        G_10[1,timestep*j+zzz]=result[zzz,72]

                st=np.transpose(s)
                ut=np.transpose(u)

                cnt_e=0
                sum_imag=0
                sum_dif_real=0
                evals0,evecs0=la.eig(A)
                evals1,evecs1=la.eig(A-np.dot(B2,temp))
                for g in range(75):
                    sum_dif_real += (evals1[g].real-evals0[g].real)**2
                    sum_imag += (evals1[g].imag)**2
                    if evals1[g].real<0:
                        cnt_e+=1
                # print(sum_dif_real,sum_imag,(s[0]-s[71])**2+(s[7]-s[71])**2+(s[15]-s[71])**2+(s[23]-s[71])**2+(s[31]-s[71])**2+(s[39]-s[71])**2+(s[47]-s[71])**2+(s[55]-s[71])**2+(s[63]-s[71])**2)

                if cnt_e==75:
                    # anv=s[1]**2+s[8]**2+s[16]**2+s[24]**2+s[32]**2+s[40]**2+s[48]**2+s[56]**2+s[64]**2+s[72]**2
                    ang=(s_o[0]-s_o[71])**2+(s_o[7]-s_o[71])**2+(s_o[15]-s_o[71])**2+(s_o[23]-s_o[71])**2+(s_o[31]-s_o[71])**2+(s_o[39]-s_o[71])**2+(s_o[47]-s_o[71])**2+(s_o[55]-s_o[71])**2+(s_o[63]-s_o[71])**2
                    # wad=(s[1]-s[72])**2+(s[8]-s[72])**2+(s[16]-s[72])**2+(s[24]-s[72])**2+(s[32]-s[72])**2+(s[40]-s[72])**2+(s[48]-s[72])**2+(s[56]-s[72])**2+(s[64]-s[72])**2
                    # ep_reward=-(6*wad/9+1.2*anv/10+ang/9)/1000
                    ep_reward=-100*(sum_dif_real/200+sum_imag/55000+ang/0.06)

                else:
                    ep_reward=-999999
                #print("wad:",wad)
                #print("anv:",anv)
                #print("ang:",ang)

                tt+=1

                for qq in range(75):
                    s_[qq]=result[-1,qq]


                if self.M.pointer > MEMORY_CAPACITY:
                    var *= 0.95    # decay the action randomness
                    b_M = self.M.sample(BATCH_SIZE)
                    b_s = b_M[:, :state_dim]
                    b_a = b_M[:, state_dim: state_dim + action_dim]
                    b_r = b_M[:, -state_dim - 1: -state_dim]
                    b_s_ = b_M[:, -state_dim:]

                    ######################################## 数据处理：将输入数据转换为graph

                    # g_b_s = process_state(b_s)
                    # g_b_s_ = process_state(b_s_)
                    # g_b_a = process_action(b_a)

                    ########################################

                    self.learn(b_s, b_s_, b_r, b_a)

                self.M.store_transition(s_o, a, ep_reward, s_) # s -> old state // s_ -> new state
                s = s_
                er+=ep_reward

                if ep_reward >=-5:
                    successtime+=1
                    sr+=1

            yy[i]=er/maxsteps # calculate the average reward of each episode to draw the traincurve

            if i == 0:
                # No action Comparing with traditional method, this only run once

                sn=np.zeros((75))
                sn_=np.zeros((75))

                for koo in range (9):
                    if koo==0:
                        #s[0]=0.6
                        sn[1]=-0.1*2*3.14
                    else:
                        sn[8*koo]=-0.1*2*3.14-0.004*koo*3.14
                sn[72]=0.2*2*3.14


                # for koo in range (9):
                #     if koo==0:
                #         sn[0]=0.6
                #         sn[1]=3.14-0.1
                #     else:
                #         sn[8*koo-1]=0.6-0.01*koo
                #         sn[8*koo]=3.14-0.1*koo

                for showj in range(maxsteps):

                    # anvn=sn[1]**2+sn[8]**2+sn[16]**2+sn[24]**2+sn[32]**2+sn[40]**2+sn[48]**2+sn[56]**2+sn[64]**2+sn[72]**2
                    # angn=(sn[0]-sn[71])**2+(sn[7]-sn[71])**2+(sn[15]-sn[71])**2+(sn[23]-sn[71])**2+(sn[31]-sn[71])**2+(sn[39]-sn[71])**2+(sn[47]-sn[71])**2+(sn[55]-sn[71])**2+(sn[63]-sn[71])**2
                    # wadn=(sn[1]-sn[72])**2+(sn[8]-sn[72])**2+(sn[16]-sn[72])**2+(sn[24]-sn[72])**2+(sn[32]-sn[72])**2+(sn[40]-sn[72])**2+(sn[48]-sn[72])**2+(sn[56]-sn[72])**2+(sn[64]-sn[72])**2

                    # ep_rewardn=-(5*wadn/9+1.2*anvn/10+angn/9)/1000
                    sum_imagn=0
                    evals1n,evecs1n=la.eig(A)
                    for g in range(75):
                        sum_imagn += (evals1n[g].imag)**2

                    # ep_rewardn=-(np.dot(np.dot(sn,Q),sn)+2*sum_imagn)
                    # rn[showj]=ep_rewardn

                    resultn=odeint(dmove,y0=sn,t=t,tfirst=True, args=(A,B1,B2,noise,un))


                    for qqqq in range(75):
                        sn_[qqqq]=resultn[-1,qqqq]

                    for ssss in range(75):
                        for zzzz in range(timestep):
                            nshowg[ssss,timestep*showj+zzzz]=resultn[zzzz,ssss]

                    ind1 = [0, 7, 15, 23, 31, 39, 47, 55, 63, 71]
                    ind2 = [71, 71, 71, 71, 71, 71, 71, 71, 71, 71]
                    ind3 = [1, 8, 16, 24, 32, 40, 48, 56, 64, 72]
                    for iiii in range(timestep):
                        rn[timestep*showj+iiii] = ocs_cal(
                            nshowg[ind1, timestep*showj+iiii] -
                            nshowg[ind2, timestep*showj+iiii],
                            nshowg[ind3, timestep*showj+iiii] / 6.28
                    )

                    sn = sn_



                plt.clf()
                plt.plot(nshowx,rn,label='PSS')

                plt.xlabel('step')
                plt.ylabel('value')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/osc_ncomparing')

                datan = pd.DataFrame(rn)
                writer = pd.ExcelWriter('./image/'+self.name+'/osc_ncomparing.xlsx')
                datan.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()



                plt.clf()
                plt.plot(nshowx,nshowg[0,:]-nshowg[71,:],label='G1')
                plt.plot(nshowx,nshowg[7,:]-nshowg[71,:],label='G2')
                plt.plot(nshowx,nshowg[15,:]-nshowg[71,:],label='G3')
                plt.plot(nshowx,nshowg[23,:]-nshowg[71,:],label='G4')
                plt.plot(nshowx,nshowg[31,:]-nshowg[71,:],label='G5')
                plt.plot(nshowx,nshowg[39,:]-nshowg[71,:],label='G6')
                plt.plot(nshowx,nshowg[47,:]-nshowg[71,:],label='G7')
                plt.plot(nshowx,nshowg[55,:]-nshowg[71,:],label='G8')
                plt.plot(nshowx,nshowg[63,:]-nshowg[71,:],label='G9')
                plt.plot(nshowx,nshowg[71,:]-nshowg[71,:],label='G10')

                plt.xlabel('time/s')
                plt.ylabel('Phase_angle/rad')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/ncomparing_phase')

                plt.clf()
                plt.plot(nshowx,nshowg[1,:]/6.28,label='G1')
                plt.plot(nshowx,nshowg[8,:]/6.28,label='G2')
                plt.plot(nshowx,nshowg[16,:]/6.28,label='G3')
                plt.plot(nshowx,nshowg[24,:]/6.28,label='G4')
                plt.plot(nshowx,nshowg[32,:]/6.28,label='G5')
                plt.plot(nshowx,nshowg[40,:]/6.28,label='G6')
                plt.plot(nshowx,nshowg[48,:]/6.28,label='G7')
                plt.plot(nshowx,nshowg[56,:]/6.28,label='G8')
                plt.plot(nshowx,nshowg[64,:]/6.28,label='G9')
                plt.plot(nshowx,nshowg[72,:]/6.28,label='G10')

                plt.xlabel('time/s')
                plt.ylabel('Angular_Velocity/Hz')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/ncomparing_angular_velocity')
                plt.clf()

                datan1 = pd.DataFrame(nshowg)

                writer = pd.ExcelWriter('./image/'+self.name+'/Ncomparing.xlsx')
                datan1.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()
            # if 1:
            if i%(int(maxepisodes/100)) == 0:
                number=number+1
                data = pd.DataFrame(zs)

                writer = pd.ExcelWriter('./image/'+self.name+'/adjustment{}.xlsx'.format(number))
                data.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()


                srr=sr/tt
                sr=0
                tt=0

                # print('now_successrate=',srr)

                plt.clf()

                plt.plot(xt,zs[0,:],label='G1')
                plt.plot(xt,zs[1,:],label='G2')
                plt.plot(xt,zs[2,:],label='G3')
                plt.plot(xt,zs[3,:],label='G4')
                plt.plot(xt,zs[4,:],label='G5')
                plt.plot(xt,zs[5,:],label='G6')
                plt.plot(xt,zs[6,:],label='G7')
                plt.plot(xt,zs[7,:],label='G8')
                plt.plot(xt,zs[8,:],label='G9')
                plt.plot(xt,G_10[0,:],label='G10')


                plt.xlabel('adjust time/ s   (%.2f)' %(avs1))
                plt.ylabel('angle/rad')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/adjustment_angle{}'.format(number))

                plt.clf()

                plt.plot(xt,zs[9,:]/6.28,label='G1')
                plt.plot(xt,zs[10,:]/6.28,label='G2')
                plt.plot(xt,zs[11,:]/6.28,label='G3')
                plt.plot(xt,zs[12,:]/6.28,label='G4')
                plt.plot(xt,zs[13,:]/6.28,label='G5')
                plt.plot(xt,zs[14,:]/6.28,label='G6')
                plt.plot(xt,zs[15,:]/6.28,label='G7')
                plt.plot(xt,zs[16,:]/6.28,label='G8')
                plt.plot(xt,zs[17,:]/6.28,label='G9')
                plt.plot(xt,G_10[1,:]/6.28,label='G10')

                plt.xlabel('adjust time/ s   (%.2f)' %(avs2))
                plt.ylabel('angular velocity/Hz')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/adjustment_angularvelocity{}'.format(number))

                plt.clf()

                plt.plot(xx,yy,label='average reward')
                plt.xlabel('iteration times')
                plt.ylabel('')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/traincurve{}'.format(number))

                plt.clf()



                data66 = pd.DataFrame(yy)

                writer = pd.ExcelWriter('./image/'+self.name+'/traincurve{}.xlsx'.format(number))
                data66.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()

                torch.save(self.actor.state_dict(), './model/'+self.name+'/actor{}.pkl'.format(number))
                torch.save(self.critic.state_dict(), './model/'+self.name+'/critic{}.pkl'.format(number))
                torch.save(self.target_actor.state_dict(), './model/'+self.name+'/target_actor{}.pkl'.format(number))
                torch.save(self.target_critic.state_dict(), './model/'+self.name+'/target_actor{}.pkl'.format(number))

    ###################################### Comparing with traditional method #########################################
            # do this TEST every 200 episodes
            # if 1:
            if i%(int(maxepisodes/100)) == 0:
                show10c=np.zeros((2,maxsteps*timestep))

                for show in range(1):
                    flag=0
                    sc=np.zeros((75))
                    sc_=np.zeros((75))
                    stac=np.zeros((18))

                    # for koo in range (9):
                    #     if koo==0:
                    #         sc[0]=0.6
                    #         sc[1]=3.14-0.1
                    #     else:
                    #         sc[8*koo-1]=0.6-0.01*koo
                    #         sc[8*koo]=3.14-0.1*koo

                    for koo in range (9):
                        if koo==0:
                            #s[0]=0.6
                            sc[1]=-0.1*2*3.14
                        else:
                            sc[8*koo]=-0.1*2*3.14-0.004*koo*3.14
                            # #s[32]=3.14
                    sc[72]=0.2*2*3.14



                    for showj in range(maxsteps):


                        anvc=sc[1]**2+sc[8]**2+sc[16]**2+sc[24]**2+sc[32]**2+sc[40]**2+sc[48]**2+sc[56]**2+sc[64]**2+sc[72]**2
                        # angc=(sc[0]-sc[71])**2+(sc[7]-sc[71])**2+(sc[15]-sc[71])**2+(sc[23]-sc[71])**2+(sc[31]-sc[71])**2+(sc[39]-sc[71])**2+(sc[47]-sc[71])**2+(sc[55]-sc[71])**2+(sc[63]-sc[71])**2
                        # wadc=(sc[1]-sc[72])**2+(sc[8]-sc[72])**2+(sc[16]-sc[72])**2+(sc[24]-sc[72])**2+(sc[32]-sc[72])**2+(sc[40]-sc[72])**2+(sc[48]-sc[72])**2+(sc[56]-sc[72])**2+(sc[64]-sc[72])**2

                        if anvc<R_threshold:
                          flag=1
                        else:
                          flag=0

                        # if  showj>100:
                        #     flag=1
                        # else:
                        #     flag=0
                        # ep_rewardc=-(5*wadc/9+1.2*anvc/10+angc/9)/1000


                        #if (wad<0.1)&(anv<0.1)&(ang<0.1):
                        #    flag=1

                        # if showj>maxsteps/4:
                        #     flag=1


                        if flag==1:
                            ac = np.zeros((action_dim))
                        else:
                            sc_o=sc
                            sc = process_state(sc,G)
                            sc=sc.reshape(1,75*16)
                            sc=torch.from_numpy(sc).to(torch.float32)
                            sc = sc.unsqueeze(0)
                            ac = self.choose_action(sc)
                            # print('****************************************************')
                            # print(ac)
                            #a = np.clip(np.random.normal(a, var), -AB, AB)

                        uc=copy.deepcopy(ac)

                        tempc=np.zeros((9,75))
                        cnt=0
                        for ls in range(9):
                            if ls!=0 and ((ls+1) in select):
                                tempc[ls,8*ls+1-1]=copy.deepcopy(uc[4*cnt])
                                tempc[ls,8*ls+2-1]=copy.deepcopy(uc[4*cnt+1])
                                tempc[ls,8*ls+4-1]=copy.deepcopy(uc[4*cnt+2])
                                tempc[ls,8*ls+7-1]=copy.deepcopy(uc[4*cnt+3])    
                                cnt+=1
                            elif ls==0 and ((ls+1) in select) :
                                tempc[0,1]=copy.deepcopy(uc[0])
                                tempc[0,2]=copy.deepcopy(uc[1])
                                tempc[0,3]=copy.deepcopy(uc[2])
                                tempc[0,6]=copy.deepcopy(uc[3])      
                                cnt+=1 

                        uuc=-np.dot(tempc,sc_o)

                        sum_imagc=0
                        sum_dif_realc=0
                        evals0,evecs0=la.eig(A)
                        evals1c,evecs1c=la.eig(A-np.dot(B2,tempc))
                        for g in range(75):
                            sum_dif_realc += (evals1c[g].real-evals0[g].real)**2
                            sum_imagc += (evals1c[g].imag)**2

                        # ep_rewardc=-(np.dot(np.dot(sc_o,Q),sc_o)+np.dot(np.dot(uuc,T),uuc)+2*sum_dif_realc+2*sum_imagc)
                        # rc[showj]=ep_rewardc
                        resultc=odeint(dmove,y0=sc_o,t=t,tfirst=True, args=(A,B1,B2,noise,uuc))
                        #result,info=tf.contrib.integrate.odeint(ode_fn,s,t,full_output=True)



                        for qqqq in range(75):
                            sc_[qqqq]=resultc[-1,qqqq]

                        for ssss in range(9):
                            if ssss==0:
                                for zzzz in range(timestep):
                                    showg[0,timestep*showj+zzzz]=resultc[zzzz,0]-resultc[zzzz,71]
                                    showg[9,timestep*showj+zzzz]=resultc[zzzz,1]
                            else:
                                for zzzz in range(timestep):
                                    showg[ssss,timestep*showj+zzzz]=resultc[zzzz,8*ssss-1]-resultc[zzzz,71]
                                    showg[9+ssss,timestep*showj+zzzz]=resultc[zzzz,8*ssss]

                        for zzzz in range(timestep):
                            show10c[0,timestep*showj+zzzz]=0
                            show10c[1,timestep*showj+zzzz]=resultc[zzzz,72]
                        sc = sc_

                        for iiii in range(timestep):
                            rc[timestep * showj + iiii] = ocs_cal(
                                np.hstack((showg[0:9,timestep * showj + iiii],
                                            show10c[0,timestep * showj + iiii])),
                                np.hstack((showg[9:18, timestep * showj + iiii] / 6.28,
                                            show10c[1, timestep * showj + iiii] / 6.28))
                        )

                plt.clf()
                plt.plot(showx,rc,label='DRL')

                plt.xlabel('step')
                plt.ylabel('Value')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/osc_comparing{}'.format(number))

                data2 = pd.DataFrame(rc)

                writer = pd.ExcelWriter('./image/'+self.name+'/osc{}.xlsx'.format(number))
                data2.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()


                plt.clf()
                plt.plot(showx,showg[0,:],label='G1')
                plt.plot(showx,showg[1,:],label='G2')
                plt.plot(showx,showg[2,:],label='G3')
                plt.plot(showx,showg[3,:],label='G4')
                plt.plot(showx,showg[4,:],label='G5')
                plt.plot(showx,showg[5,:],label='G6')
                plt.plot(showx,showg[6,:],label='G7')
                plt.plot(showx,showg[7,:],label='G8')
                plt.plot(showx,showg[8,:],label='G9')
                plt.plot(showx,show10c[0,:],label='G10')

                plt.xlabel('time/s')
                plt.ylabel('Phase_angle/rad')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/comparing_phase{}'.format(number))

                plt.clf()
                plt.plot(showx,showg[9,:]/6.28,label='G1')
                plt.plot(showx,showg[10,:]/6.28,label='G2')
                plt.plot(showx,showg[11,:]/6.28,label='G3')
                plt.plot(showx,showg[12,:]/6.28,label='G4')
                plt.plot(showx,showg[13,:]/6.28,label='G5')
                plt.plot(showx,showg[14,:]/6.28,label='G6')
                plt.plot(showx,showg[15,:]/6.28,label='G7')
                plt.plot(showx,showg[16,:]/6.28,label='G8')
                plt.plot(showx,showg[17,:]/6.28,label='G9')
                plt.plot(showx,show10c[1,:]/6.28,label='G10')

                plt.xlabel('time/s')
                plt.ylabel('Angular_Velocity/Hz')
                plt.legend()
                plt.show()
                plt.savefig('./image/'+self.name+'/comparing_angular_velocity{}'.format(number))
                plt.clf()



                data1 = pd.DataFrame(showg)

                writer = pd.ExcelWriter('./image/'+self.name+'/Comparing{}.xlsx'.format(number))
                data1.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()


                data2 = pd.DataFrame(show10c)

                writer = pd.ExcelWriter('./image/'+self.name+'/Comparing10{}.xlsx'.format(number))
                data2.to_excel(writer, 'page_1', float_format='%.5f')
                writer.save()
                writer.close()

    def learn(self, s, s_, r, a):
        # step 1
        # print(11)
        s_oo=s
        s_oo_=s_
        s=process_state(s,G)
        s=s.reshape(BATCH_SIZE,75*16)
        s_=process_state(s_,G)
        s_=s_.reshape(BATCH_SIZE,75*16)
        a = torch.tensor(a, dtype=torch.float).to(self.critic.device) # action
        r = torch.tensor(r, dtype=torch.float).to(self.critic.device) # reward
        s=torch.tensor(s, dtype=torch.float).to(self.critic.device) # state
        s_ = torch.tensor(s_, dtype=torch.float).to(self.critic.device) # state
        s_oo=torch.tensor(s_oo, dtype=torch.float).to(self.critic.device) # state
        s_oo_ = torch.tensor(s_oo_, dtype=torch.float).to(self.critic.device) # state

        # step 2
        self.critic.eval()  # evaluation mode is used, not training mode
        self.actor.train()
        mu = self.actor.forward(s)
        # mu = torch.tensor(process_action(mu.detach().numpy()),
        #                   dtype=torch.float).to(self.critic.device) # 将Actor网络输出的action值mu也处理为图数据
        # q=self.critic.forward(self.b_g, s, mu)
        q=self.critic.forward(s_oo, mu)
        actor_loss = -torch.mean(q)  # PG algorithm

        self.atrain.zero_grad()
        actor_loss.backward()
        self.atrain.step()
        al = actor_loss.item()
        # print('actor_loss=',actor_loss)
        # for parameters in self.actor.parameters():
        #     print(parameters)

        # step 3
        self.target_actor.eval()
        self.target_critic.eval()
        a_ = self.target_actor(s_) # next_action
        # a_ = torch.tensor(process_action(a_.detach().numpy()),
        #                   dtype=torch.float).to(self.critic.device)  # 将next_action值a_也处理为图数据
        # c_value_ = self.target_critic(self.b_g, s_, a_) # target_value
        c_value_ = self.target_critic(s_oo_, a_) # target_value
        # target=r+GAMMA*c_value_
        target = []  # calculate the target Q function
        for j in range(BATCH_SIZE):
            target.append(r[j] + self.gamma * c_value_[j]) # expected_value
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(BATCH_SIZE, 1)  # reshape the target Q with the same shape as batch


        # step 4
        self.critic.train()  # training mode
        self.actor.eval()
        # c_value = self.critic(self.b_g, s, a) # value
        c_value = self.critic(s_oo, a) # value
        critic_loss = F.mse_loss(c_value, target)  # TD error of TD algorithm # value_loss

        # step 5
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()
        cl = critic_loss.item()
        # print('critic_loss=',critic_loss)
        # for parameters in self.critic.parameters():
        #     print(parameters)

        # step 6
        self.update_network_parameters()

        return al, cl

    def update_network_parameters(self, tau=None):  # cover the parameters of target nets with
        # parameters of evaluation nets
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # def save_models(self):
    #     self.actor.save_checkpoint()
    #     self.target_actor.save_checkpoint()
    #     self.critic.save_checkpoint()
    #     self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def choose_action(self, state):
        self.actor.eval()# evaluation mode is used, not training mode
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)# state to GPU
        mu = self.actor.forward(state).to(self.actor.device)# real action to GPU
        mu = torch.reshape(mu, ((-1,)))
        return mu.detach().cpu().numpy()

#abb=ab*np.ones((action_dim),dtype=np.int)
#abb=np.array([12,12,12,10,12,12,11.03,9.51,12])
#abb=np.array([1.5*ab,ab/3,1.2*ab,0,0,0,0,0,0,0])
abb=np.array([a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b,a_b])
agent = Agent(gamma=GAMMA, tau=0.00001, AB=abb, name=NAME)
agent.adjust(abb,maxepisodes,maxsteps)
