#%%
import pandas as pd
import numpy as np
import math
import seaborn as sns

import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import*
plt.style.use('seaborn-whitegrid')
total_vol = 140*140*100 #total volume
no_atoms = 63600  #total number of atoms

density_liquid = [0.74,0.68]    #density of liquid[liquid_1,liquid_2]
density_vapour = [0.012,0.03]   #density of vapour[vapour_1,vapour_2]

data_1 = np.genfromtxt('data_1.csv',delimiter=',',skip_header=1)    #extracted data from Data_1.csv and eliminated it's first  row 
data_1 = data_1[:,1:]


data_2 = np.genfromtxt('data_2.csv',delimiter=',',skip_header=1)    #extracted data from Data_2.csv and eliminated it's first  row 
data_2 = data_2[:,1:]

data_3 = np.genfromtxt('density.csv',delimiter=',',skip_header=1)    #extracted data from Data_1.csv and eliminated it's first  row 
data_3 = data_3[:,1:]

data_4 = np.genfromtxt('density.csv',delimiter=',',skip_header=1)    #extracted data from Data_1.csv and eliminated it's first  row 
data_4 = data_4[:,1:]
tags = np.loadtxt("data_1.csv",usecols=[0],dtype=str,delimiter=",",skiprows=1)
for i,v in enumerate(tags):
    tags[i] = v.replace(" ","") #extracted lattice type

lattice_1 = np.loadtxt("data_1.csv",usecols=[0],dtype=str,delimiter=",",skiprows=1)
for i,v in enumerate(lattice_1):
    lattice_1[i] = v.replace(" ","") #extracted lattice type

lattice_2 = np.loadtxt("data_2.csv",usecols=[0],dtype=str,delimiter=",",skiprows=1)
for i,v in enumerate(lattice_2):
    lattice_2[i] = v.replace(" ","") #extracted lattice type

lattice_3 = np.loadtxt("density_3.csv",usecols=[0],dtype=str,delimiter=",",skiprows=1)
for i,v in enumerate(lattice_3):
    lattice_3[i] = v.replace(" ","") #extracted lattice type

lattice_4 = np.loadtxt("density.csv",usecols=[0],dtype=str,delimiter=",",skiprows=1)
for i,v in enumerate(lattice_4):
    lattice_4[i] = v.replace("","")
#%%

def rdrop(cos_theta,v_drop):
    r_drop = pow(3*v_drop/(np.pi*(1+cos_theta.nominal_value)**2 * (2-cos_theta.nominal_value)),1/3)
    return r_drop
def ASL(theta, cos_theta, density_vapour,density_liquid):   #defined function for computing area of the solid-liquid interface
    v_drop = (no_atoms-total_vol*density_vapour)\
             /(density_liquid-density_vapour)

    r_drop = pow(3*v_drop/(np.pi*(1+cos_theta.nominal_value)**2 * (2-cos_theta.nominal_value)),1/3)

    rsl = r_drop*math.sin(math.radians(theta.nominal_value))

    return np.pi*rsl**2 

def ALV(theta,cos_theta, density_vapour,density_liquid):    #defined function for computing area of the liquid-vapour interface
    v_drop = (no_atoms-total_vol*density_vapour)\
             /(density_liquid-density_vapour)    # computes volume of drop

    r_drop = pow(3*v_drop/(np.pi*(1+cos_theta)**2 * (2-cos_theta)),1/3)  #computes radius of drop
    
    rsl = r_drop*math.sin(math.radians(theta.nominal_value))    #computes radius of solid-liquid interface
    
    hcent = (r_drop**2-rsl**2)**(0.5) + r_drop
    return   np.pi*(hcent**2+rsl**2)

def GAMMA_LV(data): #defined function for computing interfacial energy of the liquid-vapour interface
      return -data[:,1]/data[:,2]

#%%
asl_1=[]
asl_2=[]
alv_1=[]
alv_2=[]
gamma_eff_1=[]
gamma_eff_2=[]
s_eff=[]
a=[]
ae=[]
b=[]
be=[]
for i in range(len(data_1[:,1])):
    sl_1 = ASL(theta=ufloat(data_1[i,4],data_1[i,5]), cos_theta=ufloat(data_1[i,2],data_1[i,3]),density_vapour=ufloat(density_vapour[0],0),density_liquid=ufloat(density_liquid[0],0))
    asl_1.append(sl_1)
    sl_2 = ASL(theta=ufloat(data_2[i,4],data_2[i,5]), cos_theta=ufloat(data_2[i,2],data_2[i,3]),density_vapour=ufloat(density_vapour[1],0),density_liquid=ufloat(density_liquid[1],0))
    asl_2.append(sl_2)
    lv_1 = ALV(theta=ufloat(data_1[i,4],data_1[i,5]), cos_theta=ufloat(data_1[i,2],data_1[i,3]),density_vapour=ufloat(density_vapour[0],0),density_liquid=ufloat(density_liquid[0],0))
    alv_1.append(lv_1)
    lv_2 = ALV(theta=ufloat(data_2[i,4],data_2[i,5]), cos_theta=ufloat(data_2[i,2],data_2[i,3]),density_vapour=ufloat(density_vapour[1],0),density_liquid=ufloat(density_liquid[1],0))
    alv_2.append(lv_2)
    g_eff_1 = ufloat(data_1[i,1],data_1[i,6])
    gamma_eff_1.append(g_eff_1)
    a.append(g_eff_1.nominal_value)
    ae.append(g_eff_1.s)
    g_eff_2 = ufloat(data_2[i,1],data_2[i,6])
    gamma_eff_2.append(g_eff_2)
    ss_eff = (gamma_eff_2[i]-gamma_eff_1[i])/data_2[i,0]-data_1[i,0]
    s_eff.append(ss_eff)
    b.append(ss_eff.nominal_value)
    be.append(ss_eff.s)
gamma_lv_1 = GAMMA_LV(data_1)
gamma_lv_2 = GAMMA_LV(data_2)

gibbs_1 = np.multiply(gamma_eff_1,asl_1) + np.multiply(gamma_lv_1,alv_1)
gibbs_2 = np.multiply(gamma_eff_2,asl_2) + np.multiply(gamma_lv_2,alv_2)

final_file = open("final.dat","w")  #creates .dat file for storing data
#%%
gibbs_real=[]
gibbs_real_abs=[]
gibbs=[]
gibbs_abs=[]
gibbs_real_inv=[]
gibbs_real_inv_abs=[]
gibbs_inv=[]
gibbs_inv_abs=[]
#REAL AND HYPOTHETICAL PLOT
for i in range(len(data_1[:,1])):
    for j in range(len(data_1[:,1])):
        m=gibbs_1[j]-gibbs_1[i]
        md=gibbs_2[j]-gibbs_2[i]
        c=m.nominal_value
        d=md.nominal_value
        if d==0:
            gibbs.append(np.nan)
            gibbs_abs.append(np.nan)
            gibbs_real.append(np.nan)
            gibbs_real_abs.append(np.nan)
        else:
            gibbs.append(c/d)
            gibbs_abs.append(abs(c/d))
#CHANGE IN TEMPERATURE
        if c==0:
            gibbs_inv.append(np.nan)
            gibbs_inv_abs.append(np.nan)
            gibbs_real_inv_abs.append(np.nan)
            gibbs_real_inv.append(np.nan)
        else:
            gibbs_inv_abs.append(abs(d/c))
            gibbs_inv.append(d/c)        

    
print('gibbs-real',gibbs_real_abs)
print('gibbs_real_inv',gibbs_real_inv_abs)
print('gibbs_inv',gibbs_inv_abs)
print('gibbs',gibbs_abs)
#%%
#INDEX ARRAY
index=[]
for i in range(len(data_1[:,1])):
    for j in range(len(data_2[:,1])):
        ind=str(tags[i])+"+"+str(tags[j])
        index.append(ind)

# def insertionSort(arr,labels): 
#     for k in range(len(arr)):
#         for i in range(1, len(arr)): 
#             key = arr[k][i]
#             label = labels[k][i]
#             j = i-1
#             while j >=0 and key > arr[k][j] : 
#                     arr[k][j+1] = arr[k][j]
#                     labels[k][j+1] = labels[k][j]  
#                     j -= 1
#             arr[k][j+1] = key
#             labels[k][j+1] = label
#     return arr,labels

def insertionSort(arr,labels): 
    for i in range(1, len(arr)): 
        key = arr[i]
        label = labels[i]
        j = i-1
        while j >=0 and key > arr[j] : 
                arr[j+1] = arr[j]
                labels[j+1] = labels[j]  
                j -= 1
        arr[j+1] = key
        labels[j+1] = label
    return arr,labels

#%%

#HYPOTHETICAL ABSOLUT HEATMAP
map_1 = np.array(gibbs_abs)
index=np.array(index)
map_1 = np.nan_to_num(map_1)
map_1,index = insertionSort(map_1,index.flatten())

map_1 = map_1.reshape(9,9)
index = index.reshape(9,9)
mask = map_1>10 # this mask will not show values less than 10
mask = mask.reshape(9,9)
sns.set(font_scale=0.35)
g = sns.heatmap(map_1,cmap='gist_rainbow_r',annot_kws={"rotation":30},annot=index,fmt='',vmax=4)
g.set_facecolor('xkcd:black')
plt.title('FREE ENERGY FOR ALL SURFACES(sorted in increasing order of obtaining surface couples \n plot on absolute scale )',fontsize=8)
#%%
index.flatten()
map_2 = (np.array(gibbs_inv_abs))
index=np.array(index)
map_2 = np.nan_to_num(map_2)
map_2,index = insertionSort(map_2,index.flatten())
map_2 = map_2.reshape(9,9)
index = index.reshape(9,9)
mask=mask.reshape(9,9)
sns.heatmap(map_2,annot=index,annot_kws={"rotation":30},fmt='', cmap='gist_rainbow_r',mask=mask_2)
g.set_facecolor('xkcd:black')
plt.title('CHANGE IN TEMPERATURE FOR ALL SURFACES(sorted in increasing order of tendency to obtain surface couple) \n plot is formed on absolute scale',fontsize=8)

#%%
index.flatten()
map_3 = (np.array(gibbs_inv))
index=np.array(index)
map_3 = np.nan_to_num(map_3)
map_3,index = insertionSort(map_3,index.flatten())
map_3=map_3.reshape(9,9)
index=index.reshape(9,9)
mask_3=map_3<0.1
mask=mask.reshape(9,9)
sns.heatmap(map_3,annot=index,annot_kws={"rotation":30},fmt='',cmap='gist_rainbow_r',mask=mask_3)
g.set_facecolor('xkcd:black')
plt.title('CHANGE IN TEMPERATURE FOR ALL SURFACES(sorted in increasing order of tendency to obtain surface couples',fontsize=8)

#%%
index.flatten()
map_4 = (np.array(gibbs))
map_4 = np.nan_to_num(map_4)
map_4,index = insertionSort(map_4,index.flatten())
map_4 = map_4.reshape(9,9)
index=index.reshape(9,9)
mask_4=map_4<0.1
mask=mask.reshape(9,9)
sns.heatmap(map_4,annot=index,annot_kws={"rotation":30},fmt='', cmap='gist_rainbow_r',mask=mask_3)
g.set_facecolor('xkcd:black')
plt.title('FREE ENERGY CHANGE FOR ALL SURFACES(sorted in increasing order of tendency to obtain surface couples',fontsize=8)


#%%
#REAL SURFACE DENSITY 
index.flatten()
sd_a1=[]
sd_a2=[]
sd_diff=[]
for i in range(len(data_3[:,1])):
    a1=data_3[i,0]/data_3[i,1]        
    a2=data_3[i,0]/data_3[i,3]    # second layer surface density
    sd_a1.append(a1)
    sd_a2.append(a2)
for i in range(len(data_3[:,1])):
    for j in range(len(data_3[:,1])):
        sd_diff.append(sd_a1[i]-sd_a2[j])
map_5 = (np.array(sd_diff))
map_5=np.nan_to_num(map_5)
map_5,index=insertionSort(map_5,index.flatten())
map_5=map_5.reshape(9,9)
index=index.reshape(9,9)
sns.set(font_scale=0.25)
sns.heatmap(map_5,annot=True,annot_kws={"rotation":30},fmt='', cmap='gist_rainbow_r')
plt.title('SURFACE DENSITY',fontsize=8)

#%%
#REAL VOLUME DENSITY
index = index.flatten()
vd_a1=[]
vd_a2=[]
vd_diff=[]
for i in range(len(data_4[:,1])):
    for i in range(len(data_4[:,1])):
        a3=data_4[i,0]/(data_4[i,1]*data_4[i,3])
        a4=data_4[i,0]/(data_4[i,1]*data_4[i,6])
        vd_diff.append(a4-a3)
        vd_a1.append(a3)
        vd_a2.append(a4)
map_6 = (np.array(vd_diff))
map_6 = np.nan_to_num(map_6)
map_6, index = insertionSort(map_6,index)
map_6  =map_6.reshape(9,9)
index =index.reshape(9,9)
sns.set(font_scale=0.45)
sns.heatmap(map_6,annot=index,vmin=10**(-5),vmax=10**(-4),annot_kws={"rotation":33},fmt='', cmap='gist_rainbow_r')
plt.title('VOLUME DENSITY',fontsize=8)




# %%


# %%
