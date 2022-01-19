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
        print("data prob...",data["probability"])
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
       
    
 
viterbi(states,observation,transition,mean_sd,stationary)