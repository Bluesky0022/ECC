
# coding: utf-8

# In[945]:

import tensorflow as tf
import importlib
import matplotlib.pyplot as plt
import numpy as np
import random 
from tensorflow.python.framework import ops
import os
cwd = os.getcwd()
import sys
sys.path.insert(0,cwd)
import bp_decoder_module
importlib.reload(bp_decoder_module)
from bp_decoder_module import polar_encode


# # Define NN model

# # Parameters




# In[946]:

K = 8                       # number of information bits
m=4
N =2**m                      # code length
train_SNR_Eb = 1            # training-Eb/No

nodes_per_layer=[128,64,32]      # each list entry defines the number of nodes in a layer

train_SNR_Es = train_SNR_Eb + 10*np.log10(K/N)
train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))


# # Data Generation


# reset the graph for new run
ops.reset_default_graph()

# Create graph session 
sess = tf.Session()



# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

x_data=tf.placeholder(shape=[None,N],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,K],dtype=tf.float32)

class NN_decoder:

    def fully_connected_relu_act(self,input_layer,weights,bias):
        layer=tf.add(tf.matmul(input_layer,weights),bias)
        return(tf.nn.relu(layer))

    def fully_connected_sigmoid_act(self,input_layer,weights,bias):
        layer=tf.add(tf.matmul(input_layer,weights),bias)
        return(tf.nn.sigmoid(layer))

    def init_weight(self,shape, st_dev):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=st_dev))
        return(weight)
    
    def init_bias(self,shape, st_dev):
        bias = tf.Variable(tf.truncated_normal(shape, stddev=st_dev))
        return(bias)


#%%
Neural_net=NN_decoder()
dev_var=0.01

w1=init_weight(shape=[N,nodes_per_layer[0]],st_dev=dev_var)
b1=init_bias(shape=[nodes_per_layer[0]],st_dev=dev_var )
layer1=fully_connected_relu_act(x_data,w1,b1)

w2=init_weight(shape=[nodes_per_layer[0],nodes_per_layer[1]],st_dev=dev_var)
b2=init_bias(shape=[nodes_per_layer[1]],st_dev=dev_var)

layer2=fully_connected_relu_act(layer1,w2,b2)

w3=init_weight(shape=[nodes_per_layer[1],nodes_per_layer[2]],st_dev=dev_var)
b3=init_bias(shape=[nodes_per_layer[2]],st_dev=dev_var)

layer3=fully_connected_relu_act(layer2,w3,b3)

w4=init_weight(shape=[nodes_per_layer[2],k],st_dev=dev_var)
b4=init_bias(shape=[k],st_dev=dev_var)

#output=fully_connected_sigmoid_act(layer2,w3,b3)
output1=tf.add(tf.matmul(layer3,w4),b4)
output=tf.round(tf.nn.sigmoid(output1))
#output=tf.nn.sigmoid(output1)

# In[950]:

#loss_vec=[]
#for i in range(10):
#    rand_index=np.random.choice()


# In[951]:


correct_prediction = tf.equal(tf.cast(output,tf.int32), tf.cast(y_target,tf.int32))
error_ratio =1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 


# In[ ]:




# In[952]:

dec_vec=tf.round(output)
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=output1))
#loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=y_target,logits=output1)
#loss=tf.losses.mean_squared_error(y_target,output)

#loss=np.mean(np.square(dec_vec-y_target))
my_opt=tf.train.AdamOptimizer(0.005)
train_step=my_opt.minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init=tf.global_variables_initializer()
sess.run(init)

#%%
design_snr_dB=0
class_encod=polar_encode()
A = class_encod.polar_design_awgn(N, K, design_snr_dB)  # logical vector indicating the nonfrozen bit locations 

d = np.zeros((2**K,K),dtype=bool)
for i in range(1,2**K):
    d[i]= class_encod.inc_bool(d[i-1])
x = np.zeros((2**K, N),dtype=bool)
u = np.zeros((2**K, N),dtype=bool)
u[:,A] = d

for i in range(0,2**K):
#    x[i] = polar_transform_iter(u[i])
    x[i]=class_encod.polar_encod_nonsys(m,N,u[i])

# # Train Neural Network

# In[957]:

n_samples=50;
y_train = np.zeros((n_samples*2**K, N),dtype=float)
d_train = np.zeros((n_samples*2**K, K),dtype=int)
x_train = np.zeros((n_samples*2**K, N),dtype=bool)


# In[ ]:




# In[958]:


# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)


# In[959]:
filename='BER_train.txt'
#filename=dir_out+filename
file = open(filename,'w')



train_loss=[]
BER_train=[]
for iter in range(1000):
    cnt=0;
    for i in range(0,2**K):
        for ii in range(0,n_samples):
            d[i].astype(int)
            d_train[cnt] = d[i]
            s_train=-2*x[i]+1 #x is codeword
            x_train[cnt]=x[i]
            y_train[cnt]=s_train+train_sigma*np.random.standard_normal(s_train.shape)
            y_train[cnt]=2*y_train[cnt]/np.float32(train_sigma**2)
            cnt=cnt+1
    
    #tmp=np.greater(y_train,0)
    #print(tmp)
    #ber_raw=(np.mean(np.equal(tmp,x_train)))
    #print(ber_raw)
    #d_train=d_train.astype(int32)
    #y_train = np.nan_to_num(normalize_cols(y_train))
    #y_train=np.nan_to_num(y_train)
    sess.run(train_step, feed_dict={x_data: y_train, y_target: d_train})
    temp_loss = sess.run(loss, feed_dict={x_data: y_train, y_target: d_train})
    
#    dec_info=sess.run(output,feed_dict={x_data: y_train, y_target: d_train})
    
    #print("dec_info:")
    #print(dec_info)
    #print("d_train:")
    #print(d_train)
    ber_tmp=sess.run(error_ratio, feed_dict={x_data: y_train, y_target: d_train})
    BER_train.append(ber_tmp)
    
    train_loss.append(temp_loss)
    if (iter+1) % 25 == 0:
        print('Generation: ' + str(iter+1) + '. Loss = ' + str(temp_loss)+' BER_train='+str(ber_tmp))
        file.write('Generation: ' + str(iter+1) +' BER_train='+str(ber_tmp))

file.close()        


# # Test NN

# In[960]:

code = 'polar'


# # Load MAP

# In[961]:




# In[972]:

test_batch = 1000 
num_msg=100000
y_test = np.zeros((test_batch, N),dtype=float)
d_test = np.zeros((test_batch, K),dtype=bool)

EbN0_vec=np.arange(2,8,0.5)
nSNR=len(EbN0_vec)
  
BER_vec=np.zeros(nSNR)
BLER_vec=np.zeros(nSNR)
for i in range(0,nSNR):
    EsN0=EbN0_vec[i]+10*np.log10(K/N)
    sigma=np.sqrt(1.0/(2*10**(EsN0/10)))
    nbiterr=0
    nblockerr=0
    nblock=0
    for ii in range(0,np.round(num_msg/test_batch).astype(int)):
        
        # Source
        #np.random.seed(0)
        d_test = np.random.randint(0,2,size=(test_batch,K)) 

        # Encoder
        if code == 'polar':
            x_test = np.zeros((test_batch, N),dtype=bool)
            u_test = np.zeros((test_batch, N),dtype=bool)
            u_test[:,A] = d_test

            for iii in range(0,test_batch):
                x_test[iii] = class_encod.polar_encod_nonsys(m,N,u_test[iii])

       
        # Modulator (BPSK)
        s_test = -2*x_test + 1
        #print('d_test='+str(x_test))
        
        
        # Channel (AWGN)
        y_test = s_test + sigma*np.random.standard_normal(s_test.shape)       
        y_test = 2*y_test/(sigma**2)
        #y_test = np.nan_to_num(normalize_cols(y_test))
        #sess.run(output)
        #temp_loss_test = sess.run(loss, feed_dict={x_data: y_test, y_target: d_test})
        dec_info=sess.run(output, feed_dict={x_data:  y_test, y_target: d_test})
        
        nbiterr_batch=np.sum(np.not_equal(dec_info.astype(int),d_test),axis=1)
        nblockerr_batch=sum(nbiterr_batch>0)
        nblockerr+=nblockerr_batch
        #print(nblock_err)
        
        
        #print(nerr_batch.shape)
        #if nerr_batch>0
        nbiterr+=np.sum(nbiterr_batch)
        nblock+=test_batch
        
        if nblockerr>200:
            break
    #print(nblockerr)
    #print(nblock)
    bler=nblockerr/nblock
    #print(bler)
    BLER_vec[i]=bler
    #print(BLER_vec)
    ber=nbiterr/(nblock*K)   
    BER_vec[i]=ber
        
print(BER_vec)


# In[974]:

#file = open(dir_out+'BLER_polar.txt','w')
file = open('BLER_polar.txt','w')
file.write(" ".join(str(elem) for elem in BER_vec))
file.close()
#%%
print(BER_vec)
# In[968]:

# result_map = np.loadtxt('map/{}/results_{}_map_{}_{}.txt'.format(code,code,N,k), delimiter=', ')
# sigmas_map = result_map[:,0]
# nb_bits_map = result_map[:,1]
# nb_errors_map = result_map[:,2]


# # Plot Bit-Error-Rate

# In[ ]:




# In[969]:




# In[970]:

# plt.plot(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N),BER_vec)
# legend.append('NN') 

# plt.plot(10*np.log10(1/(2*sigmas_map**2)) - 10*np.log10(k/N), nb_errors_map/nb_bits_map)
# legend.append('MAP') 

# plt.legend(legend, loc=3)
# plt.yscale('log')
# plt.xlabel('$E_b/N_0$ in dB')
# plt.ylabel('BER')    
# plt.grid(True)
# plt.show()


# In[971]:




# In[ ]:




# In[ ]:




# In[ ]:



