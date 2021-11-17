import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import matplotlib.pyplot as plt


def fsar(df,alp0,alp_lim):
    """
    Function to calculate the parabolic SAR indicator
    df: DataFrame. Data to which the ADX will be calculated
    alp0,alp_lim: Floats. alp0 and alp_lim give the initial and limit acceleration parameter, respectively.
    """    

    def ept(h1,h2,l1,l2,out):
        """Function to calculate the extreme point
        h1,h2: Floats. Market high at time t1 and t2 respectively (t2 > t1) 
        l1,l2: Floats. Market low at time t1 and t2 respectively (t2 > t1)
        out: 1 or -1. Strategy opinion at t2
        """
        if  out == 1:
            return max(h1,h2)
        elif out == -1:
            return min(l1,l2)
    
    def alpha(h0,h1,h2,l0,l1,l2,out1,out2,alp):
        """Function to calculate the acceleration parameters
        h0,h1,h2: Floats. Market high at time t0, t1 and t2 respectively (t2 > t1 > t0)
        l0,l1,l2: Floats. Market low at time t0, t1 and t2 respectively (t2 > t1 > t0)
        out1,out2: 1 or -1. Strategy opinion at t1 and t2 respectively (t2 > t1)
        alp: Float. previous value of the acceleration parameter
        """
        if out1 != out2:
            return alp0
        elif ept(h0,h1,l0,l1,out1) != ept(h1,h2,l1,l2,out2):
            return min(alp_lim,alp + alp0)
        else:
            return min(alp_lim,alp)
    
    def sar(tt,h0,h1,h2,l0,l1,l2,out1,out2,psar,alp):
        """Function to calculate the parabolic SAR
        tt: Integer. Time (array index) at which the SAR will be calculate
        h0,h1,h2: Floats. Market high at time t0, t1 and t2 respectively (t2 > t1 > t0)
        l0,l1,l2: Floats. Market low at time t0, t1 and t2 respectively (t2 > t1 > t0)
        out1,out2: 1 or -1. Strategy opinion at t1 and t2 respectively (t2 > t1) 
        psar: Float. Previous SAR value. Calculated at tt-1      
        alp: Float. Previous acceleration parameter. Calculated at tt-1        
        """
        if (out1 != out2 and out2 == 1):
            return min( (df['Low'][tt]), ept(h1,h2,l1,l2,out1) )
        elif (out1 != out2 and out2 == -1):            
            return max( df['High'][tt], ept(h1,h2,l1,l2,out1) )
        else: 
            return psar +  alpha(h0,h1,h2,l0,l1,l2,out1,out2,alp) * ( ept(h1,h2,l1,l2,out1) - psar)  

    # Initial parameters
    out0, nn  = 1, df.shape[0]
    v_sar = [ (df['Low'])[0] + alp0*( (df['High'])[0] + (df['Low'])[0] ) ]
    v_out, v_alp = [out0], [alp0]
    v_out.append( ( (df['High'])[1] - (df['High'])[0]  ) / abs( (df['High'])[1] - (df['High'])[0]  ) )
    v_alp.append(alp0)

    if v_out[1] == 1:
        v_sar.append( (df['High'])[1] )
    elif v_out[1] == -1:
        v_sar.append( (df['Low'])[1] )  
    
    # Running over all the times tt > 1
    for tt in range(2,nn):
        h0, h1, h2 = df['High'][tt-2], df['High'][tt-1], df['High'][tt]
        l0, l1, l2 = df['Low'][tt-2], df['Low'][tt-1], df['Low'][tt]
        psar, alp = v_sar[tt-1], v_alp[tt-1]
        v_sar.append( sar(tt,h0,h1,h2,l0,l1,l2,v_out[tt-2],v_out[tt-1],psar,alp) )
        v_alp.append(alpha(h0,h1,h2,l0,l1,l2,v_out[tt-2],v_out[tt-1],alp))
        if v_sar[tt] < l2:
            v_out.append(1)
        elif v_sar[tt] > h2:
            v_out.append(-1)
        else:
            v_out.append(-v_out[tt-1])       
    return  v_sar, v_out

def frsi(df,ped):
    """
    Function to calculate the Relative Strength Index (RSI)
    df: DataFrame. Data to which the ADX will be calculated
    ped: Integer. Period where ADX will be calculated
    """    
    gain, loss = [], []
    for i in range(nn-1):
        diff = df['Close'][i+1] - df['Close'][i]
        if diff >= 0:
            gain.append(diff)
            loss.append(0)
        else:
            gain.append(0)
            loss.append(-diff)
    aveg = [np.mean(gain[0:ped])]
    avel = [np.mean(loss[0:ped])]
    rs = [np.mean(gain[0:ped])/np.mean(loss[0:ped])]
    rsi = [100 - 100 / (1+ rs[0])]    
            
    for i in range(1,nn-ped):
        aveg.append((aveg[i-1]*(ped-1) + gain[i+ped-1])/ped)
        avel.append((avel[i-1]*(ped-1) + loss[i+ped-1])/ped)
        rs.append(aveg[i]/avel[i])    
        rsi.append(100 - 100 / (1+ rs[i]))    
    return rsi    

def fadx(df,ped):
    """
    Function to calculate the Average Directional Index ADX
    df: DataFrame. Data to which the ADX will be calculated
    ped: Integer. Period where ADX will be calculated
    """
    nn  = df.shape[0]
    tr = [df['High'][0] - df['Low'][0]]
    vdm = df['Close'][0] - (df['High'][0] + df['Low'][0])/2
    if vdm >= 0:
        dmp, dmn = [vdm], [0]
    else:
        dmp, dmn = [0], [-vdm]        
    for i in range(1,nn):
        tr.append( max(df['High'][i],df['Close'][i]) - min(df['Low'][i],df['Close'][i]) )
        vdm1, vdm2 = df['High'][i] - df['High'][i-1], df['Low'][i-1] - df['Low'][i]
        if  vdm1 >= vdm2:
            dmp.append(vdm1); dmn.append(0)
        elif vdm2 > vdm1:
            dmp.append(0); dmn.append(vdm2) 

    atr, dmp_, dmn_ = [np.mean(tr[:ped])], [np.mean(dmp[:ped])], [np.mean(dmn[:ped])]
    dip, din = [dmp_[0]/atr[0]], [dmn_[0]/atr[0]]
    dx = [abs(dip[0] - din[0])/abs(dip[0] + din[0])]
    for i in range(1,nn-ped):
        atr.append((atr[i-1]*(ped-1) + tr[i+ped-1])/ped)
        dmp_.append((dmp_[i-1]*(ped-1) + dmp[i+ped-1])/ped )
        dmn_.append((dmn_[i-1]*(ped-1) + dmn[i+ped-1])/ped )
        dip.append(dmp_[i]*100/atr[i]) 
        din.append(dmn_[i]*100/atr[i])
        dx.append( abs(dip[i]-din[i])*100/abs(dip[i]+din[i]) )
    adx = [np.mean(dx[:ped])]
    nn_ =  len(dx)
    for i in range(1,nn_-ped):
        adx.append((adx[i-1]*(ped-1) + dx[i+ped-1])/ped)    
    return (adx, dip, din)


df = yf.download('TSLA',period = '1d', interval = '1m')
names = ['Open', 'High', 'Low', 'Close', 'Volume']
timev = list(map(lambda x: x.hour+x.minute/60,df.index))   # Pay attention to the size of the data (where it starts and finishs running time)
nt = len(timev)

#######################    Test SAR   ###############################  
  
# SAR, feel = fsar(df,0.02,0.2)    
# plt.plot(timev,df['Close'])
# plt.plot(timev,SAR,'ko', markersize = 1)
# plt.show()


#######################    Test RSI   ###############################  
# ped = 14
# RSI = frsi(df,ped)
        
# fig, ax = plt.subplots(2)
# ax[0].plot(timev,df['Close'])
# ax[1].plot(timev[ped:],RSI)
# ax[0].set_xlim(timev[0],timev[-1])
# ax[1].set_xlim(timev[0],timev[-1])
# plt.show()


#######################    Test fadx   ###############################  
# ped = 14    
# ADX, DIp, DIn = fadx(df,ped)
# fig, ax = plt.subplots(2)
# ax[0].plot(timev,df['Close'],'b')
# ax[1].plot(timev[ped:],DIp,'g--')
# ax[1].plot(timev[ped:],DIn,'r--')
# ax[1].plot(timev[2*ped:],ADX,'k')
# ax[0].set_xlim(timev[0],timev[-1])
# ax[1].set_xlim(timev[0],timev[-1])
# plt.show()
