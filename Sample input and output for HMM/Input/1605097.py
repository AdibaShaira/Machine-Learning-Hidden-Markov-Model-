import numpy as np
import itertools
import math 
from scipy.linalg import eig
from scipy.stats import norm
f = open("data.txt", "r")
observation =[]
for line in f:
    observation.append(line.strip())
print(len(observation)) 
transition_mat = np.matrix([
    [0.7, 0.3],\
    [0.1 ,0.9],\
])

S, U = eig(transition_mat.T)
stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
stationary = stationary / np.sum(stationary)
print(stationary)
transition=[]
with open('parameters.txt', 'r') as f:
    for line in itertools.islice(f, 1, 3):
        transition.append(line.split())
transition[0][0]=float(transition[0][0])
transition[0][1]=float(transition[0][1])
transition[1][0]=float(transition[1][0])
transition[1][1]=float(transition[1][1])
mean_sd=[]
end_probs = [1, 1]
with open('parameters.txt', 'r') as f:
    for line in itertools.islice(f, 3, 5):
        mean_sd.append(line.split())
mean_sd[0][0]=int(mean_sd[0][0])
mean_sd[0][1]=int(mean_sd[0][1])
mean_sd[1][0]=math.sqrt(int(mean_sd[1][0]))
mean_sd[1][1]=math.sqrt(int(mean_sd[1][1]))
states=("El Nino","La Nina")
print(mean_sd)
def viterbi(states,observation,transition,mean_sd,stationary):
    V = [{}]
    state_len=len(states)
    
    for st in range(state_len):
        if st==0:
            st_v="El Nino"
        elif st==1:
            st_v="La Nina"
        emission=norm.pdf(float(observation[0]),mean_sd[0][st],mean_sd[1][st]) 
        V[0][st_v]={"probability":math.log(stationary[st])+math.log(emission),"prev_state":None}
        
    for t in range(1, len(observation)):
        V.append({})
        for st in range(state_len):
            if st==0:
              st_v="El Nino"
            elif st==1:
              st_v="La Nina"
            max_transmission=V[t - 1][states[0]]["probability"] + math.log(transition[0][st])
            prev_state_selected=states[0]
            for prev_st in range(1,state_len):
                if(prev_st==1):
                    prev_st_v="La Nina"
                if prev_st==0:
                     prev_st_v="El Nino"
                #print("Checking...",V[t-1][prev_st_v]["probability"])
                #print("Checking trans....",transition[prev_st][st])
                tr_prob = V[t - 1][prev_st_v]["probability"]+math.log(transition[prev_st][st])
                if tr_prob > max_transmission:
                    max_transmission = tr_prob
                    prev_state_selected = prev_st_v
            emission=norm.pdf(float(observation[t]),mean_sd[0][st],mean_sd[1][st]) 
            max_prob = (max_transmission + math.log( emission))
            V[t][st_v] ={"probability": max_prob, "prev_state": prev_state_selected}
    opt = []
    max_prob = 0.0
    best_st = None
    # for i in range(len(V)):
    #      print("len.......hala.",V[i].items())
    for st, data in V[-1].items():
        
        if (data["probability"]*-1) > max_prob:
            print("dhukse....")
            max_prob = data["probability"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    for t in range(len(V) - 2, -1, -1):
        #print("Final....",previous)
        opt.insert(0, V[t + 1][previous]["prev_state"])
        previous = V[t + 1][previous]["prev_state"]
    textfile = open("output_without_baum.txt", "w")
    for i in range(len(opt)):
        textfile.write("\""+ opt[i]+"\"" + "\n")
    textfile.close()   
       
def viterbi_after_baum(states,observation,transition,mean_sd,stationary):
    V = [{}]
    state_len=len(states)
    
    for st in range(state_len):
        if st==0:
            st_v="El Nino"
        elif st==1:
            st_v="La Nina"
        emission=norm.pdf(float(observation[0]),mean_sd[0][st],mean_sd[1][st]) 
        V[0][st_v]={"probability":math.log(stationary[st])+math.log(emission),"prev_state":None}
        
    for t in range(1, len(observation)):
        V.append({})
        for st in range(state_len):
            if st==0:
              st_v="El Nino"
            elif st==1:
              st_v="La Nina"
            max_transmission=V[t - 1][states[0]]["probability"] + math.log(transition[0][st])
            prev_state_selected=states[0]
            for prev_st in range(1,state_len):
                if(prev_st==1):
                    prev_st_v="La Nina"
                if prev_st==0:
                     prev_st_v="El Nino"
                #print("Checking...",V[t-1][prev_st_v]["probability"])
                #print("Checking trans....",transition[prev_st][st])
                tr_prob = V[t - 1][prev_st_v]["probability"]+math.log(transition[prev_st][st])
                if tr_prob > max_transmission:
                    max_transmission = tr_prob
                    prev_state_selected = prev_st_v
            emission=norm.pdf(float(observation[t]),mean_sd[0][st],mean_sd[1][st]) 
            max_prob = (max_transmission + math.log( emission))
            V[t][st_v] ={"probability": max_prob, "prev_state": prev_state_selected}
    opt = []
    max_prob = 0.0
    best_st = None
    # for i in range(len(V)):
    #      print("len.......hala.",V[i].items())
    for st, data in V[-1].items():
        
        if (data["probability"]*-1) > max_prob:
            print("dhukse....")
            max_prob = data["probability"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    for t in range(len(V) - 2, -1, -1):
        #print("Final....",previous)
        opt.insert(0, V[t + 1][previous]["prev_state"])
        previous = V[t + 1][previous]["prev_state"]
    textfile = open("output_after_baum.txt", "w")
    for i in range(len(opt)):
        textfile.write("\""+ opt[i]+"\"" + "\n")
    textfile.close()
def forwardprobabilty(states,observation,transition,mean_sd,stationary):
    alpha = np.zeros((len(states), len(observation)))
    
    for st in range(len(states)): 
        emission=norm.pdf(float(observation[0]),mean_sd[0][st],mean_sd[1][st]) 
        alpha[st][0]=stationary[st]*emission
    sum_state=alpha[0,0]+alpha[1,0]
    alpha[0,0]=alpha[0,0]/sum_state
    alpha[1,0]=alpha[1,0]/sum_state
    for t in range(1, len(observation)):
        for st in range(len(states)):
            emission=norm.pdf(float(observation[t]),mean_sd[0][st],mean_sd[1][st])
            values=[alpha[k][t - 1] * emission* transition[k][st] for k in
                          range(len(states))]
            
            alpha[st, t] = sum(values)
        sum_state=alpha[0,t]+alpha[1,t]
        alpha[0,t]=alpha[0,t]/sum_state
        alpha[1,t]=alpha[1,t]/sum_state
        #print("alpha..",alpha[0,t],alpha[1,t])
    
    alpha_sink = np.multiply(alpha[:, -1], end_probs)
    alpha_sink_value = sum(alpha_sink)
    return alpha,alpha_sink_value
def backwardprobabilty(states,observation,transition,mean_sd,stationary):
    beta=np.zeros((len(states), len(observation)))
    for t in range(1,len(observation)+1):
        for st in range(len(states)):
             if (-t == -1):
                beta[st, -t] = end_probs[st]
             else:
                
                values = [beta[k][-t+1] *norm.pdf(float(observation[-t+1]),mean_sd[0][k],mean_sd[1][k])  * transition[st][k] for k in range(len(states))]
                
                beta[st, -t] = sum(values)
        sum_beta=beta[0,-t]+beta[1,-t]
        beta[0,-t]=beta[0,-t]/sum_beta
        beta[1,-t]=beta[1,-t]/sum_beta
    
    return beta
def gamma(forward,backward,forward_sink):
    gamma_probab = np.zeros((len(states), len(observation)))
    for i in range(len(observation)):
        for j in range(len(states)):
            gamma_probab[j, i] = (forward[j, i] * backward[j, i]) / forward_sink
        sum_gamma=gamma_probab[0,i]+gamma_probab[1,i]
        gamma_probab[0,i]=gamma_probab[0,i]/sum_gamma
        gamma_probab[1,i]=gamma_probab[1,i]/sum_gamma
    return gamma_probab
def epsilon(forward,backward,forward_sink):
    epsilon_probab = np.zeros((len(states), len(observation)-1, len(states)))

    for i in range(len(observation)-1):
        for j in range(len(states)):
            for k in range(len(states)):
                emission=norm.pdf(float(observation[i+1]),mean_sd[0][k],mean_sd[1][k])
                epsilon_probab[j,i,k] = ( forward[j,i] * backward[k,i+1] * transition[j][k] * emission) / forward_sink
        sum_epsilon=epsilon_probab[0,i,0]+epsilon_probab[0,i,1]+epsilon_probab[1,i,0]+epsilon_probab[1,i,1]
        epsilon_probab[0,i,0]=epsilon_probab[0,i,0]/sum_epsilon
        epsilon_probab[0,i,1]=epsilon_probab[0,i,1]/sum_epsilon
        epsilon_probab[1,i,0]=epsilon_probab[1,i,0]/sum_epsilon
        epsilon_probab[1,i,1]=epsilon_probab[1,i,1]/sum_epsilon

                                                  
    return epsilon_probab
def baum_welch(states,observation,transition,mean_sd,stationary):   
    for iteration in range(10):
        forward,forward_sink=forwardprobabilty(states,observation,transition,mean_sd,stationary)
        backward=backwardprobabilty(states,observation,transition,mean_sd,stationary)
        gamma_val=gamma(forward,backward,forward_sink)
        epsilon_val=epsilon(forward,backward,forward_sink)
        a = np.zeros((len(states), len(states)))
        b = np.zeros((len(states), len(observation)))
        for j in range(len(states)):
          for i in range(len(states)):
            for t in range(len(observation)-1):
                a[j,i] = a[j,i] + epsilon_val[j,t,i]

            denomenator_a = [epsilon_val[j, obs, st] for obs in range(len(observation) - 1) for st in range(len(states))]
            denomenator_a = sum(denomenator_a)

            if (denomenator_a == 0):
                a[j,i] = 0
            else:
                a[j,i] = a[j,i]/denomenator_a
        
        transition=a
        mean_divide_zero=0
        mean_zero=0
        mean_one=0
        mean_divide_one=0
        var_divide_one=0
        var_divide_zero=0
        var_zero=0
        var_one=0
        for i in range(len(observation)):
            mean_divide_zero=mean_divide_zero+gamma_val[0,i]
            mean_zero=mean_zero+(float(observation[i])*gamma_val[0,i])
            mean_divide_one=mean_divide_one+gamma_val[1,i]
            mean_one=mean_one+(float(observation[i])*gamma_val[1,i])
        mean_sd[0][0]=mean_zero/mean_divide_zero
        mean_sd[0][1]=mean_one/mean_divide_one
        for i in range(len(observation)):
            var_divide_zero=var_divide_zero+gamma_val[0,i]
            var_zero=var_zero+(gamma_val[0,i]*(float(observation[i])-mean_sd[0][0])*(float(observation[i])-mean_sd[0][0]))
            var_divide_one=var_divide_one+gamma_val[1,i]
            var_one=var_one+(gamma_val[1,i]*(float(observation[i])-mean_sd[0][1])*(float(observation[i])-mean_sd[0][1]))
        
        mean_sd[1][0]=math.sqrt(var_zero/var_divide_zero)
        mean_sd[1][1]=math.sqrt(var_one/var_divide_one)
    transition_mat = np.matrix([
    [transition[0][0], transition[0][1]],\
    [transition[1][0] ,transition[1][1]],\
      ])

    S, U = eig(transition_mat.T)
    stationary_new = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary_new= stationary_new / np.sum(stationary_new)
    textfile = open("learned_parameter.txt", "w")
    textfile.write("2"+"\n")
    textfile.write(str(transition[0][0]))
    textfile.write(" "+str(transition[0][1]))
    textfile.write("\n")
    textfile.write(str(transition[1][0]))
    textfile.write(" "+str(transition[1][1]))
    textfile.write("\n")
    textfile.write(str(mean_sd[0][0]))
    textfile.write(" "+str(mean_sd[0][1]))
    textfile.write("\n")
    textfile.write(str(mean_sd[1][0]*mean_sd[1][0]))
    textfile.write(" "+str(mean_sd[1][1]*mean_sd[1][1]))
    textfile.write("\n")
    textfile.write(str(stationary_new[0]))
    textfile.write(" "+str(stationary_new[1]))
    textfile.close()
    return transition,mean_sd  
viterbi(states,observation,transition,mean_sd,stationary)
transition,mean_sd=baum_welch(states,observation,transition,mean_sd,stationary)
viterbi_after_baum(states,observation,transition,mean_sd,stationary)

