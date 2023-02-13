#---------------------------------------------------------------------------------------------------------------------------------#
#-HI mass function for HIZOA
#-Author: Khaled Said (The University of Queensland)
#-Method: 2DSWML
#-KS thanks Michael G. Jones for his advice to use numpy.einsum instead of using loops (during the 3GC4 workshop in 2016)
#-numpy.einsum makes this script much faster
#-https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
#---------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
from math import*
from numpy import*
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from scipy import integrate
#------------------------------------------------------------------------------------#
#------------Read your data----------------------------------------------------------#
#------------------------------------------------------------------------------------#
name_0=[];S_0=[];flux_0=[];vLG=[];D_Mpc=[];M_HI=[];w_50=[];Veff_final=[]
#------------------------------------------------------------------------------------#
f0 = open("finalfinal_hizoa.dat","r")
for line in f0:
    if line[0]!='#':
        if float(line.split()[15])/float(line.split()[13]) >= 0.022 and -5.0 <= float(line.split()[9]) <= 5.0:#----using Mean Flux S lim----#
            name_0.append(str(line.split()[1]))
            flux_0.append(float(line.split()[15]))
            w_50.append(np.log10(float(line.split()[13])))
            S_0.append(float(line.split()[15])/float(line.split()[13]))
            v_hel_hizoa_s = float(line.split()[11])
            l_hizoa_s = float(line.split()[8])
            b_hizoa_s = float(line.split()[9])
            v_LG_hizoa_s = v_hel_hizoa_s + 300*(sin(radians(l_hizoa_s)))*(cos(radians(b_hizoa_s)))
            D_hizoa_s = v_LG_hizoa_s/75.0
            MHI_hizoa_s = np.log10(235600*(D_hizoa_s**2)*float(line.split()[15]))
            #            print name_hizoa,log10(MHI_hizoa)
            M_HI.append(MHI_hizoa_s)
            D_Mpc.append(D_hizoa_s)
            vLG.append(v_hel_hizoa_s)
f0.close()
#-------------------------------------------------------------------------------------#
f1 = open("finalfinal_NE.dat","r")
for line in f1:
    if line[0]!='#':
        if float(line.split('|')[8])!=0 and float(line.split('|')[7])/float(line.split('|')[8]) >= 0.022 and -5.0 <= float(line.split('|')[5]) <= 5.0:#----using Mean Flux S lim-----#
            name_0.append(str(line.split('|')[1]))
            flux_0.append(float(line.split('|')[7]))
            w_50.append(np.log10(float(line.split('|')[8])))
            S_0.append(float(line.split('|')[7])/float(line.split('|')[8]))
            v_hel_NEGB_s = float(line.split('|')[6])
            l_NEGB_s = float(line.split('|')[4])
            b_NEGB_s = float(line.split('|')[5])
            v_LG_NEGB_s = v_hel_NEGB_s + 300*(sin(radians(l_NEGB_s)))*(cos(radians(b_NEGB_s)))
            D_NEGB_s = v_LG_NEGB_s/75.0
            MHI_NEGB_s = np.log10(235600*(D_NEGB_s**2.0)*float(line.split('|')[7]))
            #            print name_NEGB,log10(MHI_NEGB)
            M_HI.append(MHI_NEGB_s)
            D_Mpc.append(D_NEGB_s)
            vLG.append(v_LG_NEGB_s)
f1.close()
#------------------------------------------------------------------------------------#
w01 = np.array(w_50)
dist01 = np.array(D_Mpc)
mHI01 = np.array(M_HI)
flux01 = np.array(flux_0)
speak01 = np.array(S_0)
D_lim = np.max(D_Mpc)
slim = 0.022
#area = 1230 #l fraction 216/360 = 0.6 & b fraction 10/180 = 0.05 ---> 0.03x41000 = 1230 deg^2
def f(x):
    return np.sin(x)
resu = integrate.quad(f, np.radians(85),np.radians(95))
area = resu[0]*np.radians(216)*180.**2/np.pi**2
print (area)
print ('HIZOA sources: ',len(mHI01),np.max(D_Mpc))
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#---------------------Some functions to be used--------------------------------------#
def Schechter(logMHI,alpha,m_s,phi_s):
    '''The Schechter function for a logarithmic (base 10) mass axis. 
    Note that the iut "m_s" is the log of M_star.'''

    M = 10.**logMHI
    M_s = 10.**m_s

    return np.log(10.)*phi_s*((M/M_s)**(alpha+1.))*np.exp(-M/M_s)

def HIPASS_distance(m,f,s):
    '''For the BGC, D_{lim,i} can simply be found by multiplying the distance D_i at which 
        the object is detected by (S_i/S_lim)^1/2, where S_i and S_lim are, respectively, 
        the peak flux of object i and the limiting peak flux of the sample, 
        which is 116 mJy (Zwaan et al. 2003)'''

    return np.sqrt(((10.**m)*s)/(235600.*f*slim))

def H(mass,w,xedge,yedge,s):
    '''The function Hijk is defined as the fraction of the bin accessible to source i. See eqs. 11 by Zwaan et al. 2003!'''
    Mjcenter = xedge+delM/2.
    Wkcenter = yedge+delW/2.
    Mjminus = xedge
    Mjplus = xedge+delM
    Wkminus = yedge
    Wkplus = yedge+delW
    Mlimminus = mass+np.log10(slim/s)-w+Wkminus
    Mlimplus = mass+np.log10(slim/s)-w+Wkplus
    if Mjminus >= Mlimplus:
        return 1.0
    elif Mjplus <= Mlimminus:
        return 0.0
    elif Mjplus > Mlimminus and Mjplus <= Mlimplus:
        return (((Mjplus-Mlimminus)**2.)/(2.*delM*delW))
    elif Mjminus <= Mlimminus and Mjplus >= Mlimplus:
        return ((2.*Mjplus - Mlimplus - Mlimminus)/(2.*delM))
    elif Mjminus >= Mlimminus and Mjplus <= Mlimplus:
        return 0.0
    elif Mjminus >= Mlimminus and Mjminus < Mlimplus:
        return 1.-(((Mlimplus-Mjminus)**2.)/(2.*delM*delW))
    else:
        return 0.0


def TWODSWML(m_final,w_final,s_final,f_final,d_final,omega):
    '''CALCULATING THE HIMF! See section 3 in Zwaan et al (2003) first to understand the method and then see section 2 in Zwaan et al (2005) for performing this V_eff'''

    #Calculate Hijk. This is slowest step, but only has to happen once.
    Hijk = np.zeros((len(m_final),len(xedges)-1,len(yedges)-1))
    print (len(m_final),len(xedges),len(yedges))

    for i in range(len(m_final)):
        for j in range(len(xedges)-1):
            for k in range(len(yedges)-1):
                Hijk[i,j,k] = H(m_final[i],w_final[i],xedges[j],yedges[k],s_final[i])
    #print Hijk[i,j,k]


    
    #Initial guess
    big_phi = 0.
    phi = 0.
    phi_new = 0.
    Gjk = 0.
    Veff = 0.
    phi = np.zeros((len(xedges)-1,len(yedges)-1)) + 1./(len(yedges)*len(xedges)*delW*delM)
    Gjk = np.zeros((len(m_final),len(m_final)))

    for i in range(len(m_final)):
        Gjk[i] = np.array(np.greater(HIPASS_distance(m_final,f_final,s_final),d_final[i]),dtype='float')
    #-------------------------------------------------------------------------------------------------------------#
    #----------------------This is the part where we iterate to find stable solutions for theta_jk----------------#
    #-I set the iteration to high number. Stable solutions (convergence) are usually found before 20 iteration with max difference of 1e-3 and before 30 iteration with max diff 1e-5.
    for N_iterations in range(1000):
        
        #Multiply H and phi over all mass and width bins
        big_phi = np.einsum('lmn,mn',Hijk,phi)*delW*delM
        #print big_phi
        
        #Calculate 1/big_phi and remove infinities
        i_big_phi = np.where(np.isfinite(np.divide(1.,big_phi)),np.divide(1.,big_phi),0.)
        #print i_big_phi,big_phi
        
        #Initialise Veff
        Veff = np.zeros(len(big_phi))

        #Calculate Veff
        Veff = np.einsum('j,jk',i_big_phi,Gjk)
        #print Veff
        
        #Initialise phi_new
        phi_new = np.zeros((len(xedges)-1,len(yedges)-1))

        #Calculate the correct bin for each galaxy and add it's 1/Veff to phi_new
        for l in range(len(m_final)):
            j = np.floor((m_final[l]-xedges[0])/delM)
            k = np.floor((w_final[l]-yedges[0])/delW)
            if Veff[l] != 0.:
                phi_new[int(j),int(k)] += 1./Veff[l]

        #check for convergence and then set phi = phi_new
        err_max = np.max(abs(phi-phi_new/(np.sum(phi_new)*delW*delM)))
        print ('iteration = ',N_iterations,'---> absolute error = ',err_max)
        if np.allclose(phi, phi_new/(np.sum(phi_new)*delW*delM), atol=1e-3):
            break
        phi = phi_new/(np.sum(phi_new)*delW*delM)
    phisum = 0.
    for j in range(len(xedges)-1):
        for k in range(len(yedges)-1):
            phisum += phi[j,k]*(xedges[j+1]-xedges[j])*(yedges[k+1]-yedges[k])
    phi_norm = phi/phisum
    #print phi_norm
    norm_cnt = 0.
    for i in range(len(m_final)):
        if big_phi[i] > 0.0:
            norm_cnt += 1./big_phi[i]
            #print norm_cnt,big_phi
    #norm_cnt = 647837.684
    vol = (4.*np.pi/3.)*(np.pi*omega/129600.)*((D_lim)**3.)
    #print vol
    n1 = norm_cnt/vol
    phi_norm = n1*phi_norm
    #print phi_norm,xedges,yedges
    np.savetxt('phi_norm.txt',phi_norm)
    Veff_norm = (Veff)/n1
    np.savetxt('Veff_norm.txt',Veff_norm)
    Veff_final.append(np.log10(1/Veff_norm))
    #Caclulate bin centres
    mj = np.zeros(len(xedges)-1)
    for j in range(len(xedges)-1):
        mj[j] = (xedges[j+1]+xedges[j])/2.
    #Calculate 2D binned mass-width function
    nwj = np.zeros((len(xedges)-1,len(yedges)-1))
    for l in range(len(m_final)):
        j = np.floor((m_final[l]-xedges[0])/delM)
        k = np.floor((w_final[l]-yedges[0])/delW)
        if Veff_norm[l] > 0.:
            nwj[int(j),int(k)] += 1./(Veff_norm[l]*delM*delW)
    #Calculate Poisson errors for mass-width function
    sigjk = np.zeros((len(xedges)-1,len(yedges)-1))
    for i in range(len(m_final)):
        j = np.floor((m_final[i]-xedges[0])/delM)
        k = np.floor((w_final[i]-yedges[0])/delW)
        if Veff_norm[i] > 0.:
            sigjk[int(j),int(k)] += 1./((Veff_norm[i]*delM*delW)**2.)
    sigjk = np.sqrt(sigjk)
    #Calculate HIMF from mass-width function
    nj = np.sum(nwj,axis=1)*delW
    sigmj = np.sqrt(np.sum(sigjk**2.,axis=1)*(delW**2.))
    #Fit HIMF with a Schecther function
    imin = 0
    imax = len(mj)
    for i in range(20):
        if nj[i] == 0.:
            imin = i+1
    for i in range(20,len(mj)):
        if nj[i] == 0.:
            imax = i
            break
    fit,cov_fit = scipy.optimize.curve_fit(Schechter,mj[imin:imax],nj[imin:imax],p0=[-1.3,9.7,1.E-7],sigma=sigmj[imin:imax])
    return nj,sigmj,[fit[0],fit[1],fit[2]],cov_fit


#Useful plotting function
def plot_HIMF(HIMF_final,HIMF_err,fit,cov,leg='HIMF'):
    print ('alpha = '+str(round(fit[0],2))+' +/- '+str(round(np.sqrt(cov[0,0]),2)))
    print ('M* = '+str(round(fit[1],2))+' +/- '+str(round(np.sqrt(cov[1,1]),2)))
    print ('phi = '+str(round(fit[2],4))+' +/- '+str(round(np.sqrt(cov[2,2]),4)))
    
    imin = 0
    imax = len(xedges)-1

    for i in range(20):
        if HIMF_err[i] >= HIMF_final[i]:
            HIMF_err[i] = 0.999*HIMF_final[i]
        if HIMF_final[i] == 0.:
            imin = i+1
    for i in range(20,len(xedges)-1):
        if HIMF_err[i] >= HIMF_final[i]:
            HIMF_err[i] = 0.999*HIMF_final[i]
        if HIMF_final[i] == 0.:
            imax = i
            break
    
#fig, ax  = plt.subplots(figsize=(10,7))#,dpi=300)
    plt.figure(1,figsize=(9,8))
#plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),-1.33,9.96,4.8E-3),'--',c='red',lw=1.,label="Martin et al. 2010")
#plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),-1.3,9.79,8.6E-3),'--',c='grey',lw=1.,label="Zwaan et al. 2003")
    plt.errorbar(xedges[imin:imax]+delM/2.,HIMF_final[imin:imax],yerr=(HIMF_err[imin:imax]),lw=0,elinewidth=2,c='black')#,label="Poisson errors")
    #plt.errorbar(xedges[imin:imax]+delM/2.,HIMF_final[imin:imax],yerr=HIMF_err[imin:imax],fmt=None,ecolor='black')
    plt.scatter(xedges[imin:imax]+delM/2.,HIMF_final[imin:imax],color='black',s=20,marker='o')#,label="2DSWML HIZOA HIMF")
    plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),fit[0],fit[1],fit[2]),ls='-',lw=1.,c='black',label="2DSWML HIZOA HIMF")
    plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),-1.3,9.79,0.0086),ls=':',lw=2.,c='black',label="Zwaan et al (2003)")
    plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),-1.33,9.96,0.0048),ls='--',lw=1.,c='black',label="Martin et al (2010)")
    plt.plot(np.arange(6.,11.,0.01),Schechter(np.arange(6.,11.,0.01),-1.25,9.94,0.005),ls='-.',lw=1.,c='black',label="Jones et al (2018)")
    
    plt.yscale('log')
    plt.xlabel(r'log $M_{HI}\/[M_{\odot}]$', fontname = 'Times New Roman',size=20)
    plt.ylabel(r'log $\Phi\/$[Mpc$^{-3}$ dex$^{-1}]$', fontname = 'Times New Roman',size=20)
    plt.ylim(1.E-6,1.)
    plt.xlim(6.,11.)
    plt.yticks([1.E-5,1.E-4,1.E-3,1.E-2,1.E-1,1.E0],[-5,-4,-3,-2,-1,0])
    legend = plt.legend(loc=1, fontsize=15)#, fancybox=True)#, framealpha=0.1)#,frameon=False)
    #legend.get_frame().set_facecolor(color = '0.97')
#legend.get_frame().set_linewidth(0.0)
#legend.get_frame().set_edgecolor('b')
#legend.set_rasterized(True)
    #plt.legend(loc=3,fontsize='small')
    plt.grid(color='0.65',linestyle=':')
    plt.text(6.2,0.00001,r'$\alpha\/=\/'+'{0:.2f}'.format(fit[0])+' \pm$ '+str(round(np.sqrt(cov[0,0]),2)), fontname = 'Times New Roman',size=20)
    plt.text(6.2,0.000005,r'$\Phi^{\star}\/=\/'+''+str(round(fit[2],4))+' \pm$ '+str(round(np.sqrt(cov[2,2]),4))+' Mpc$^{-3}$', fontname = 'Times New Roman',size=20)
    plt.text(6.2,0.0000025,r'log ($M_{HI}^{\star}/M_{\odot})\/=\/'+str(round(fit[1],2))+' \pm$ '+str(round(np.sqrt(cov[1,1]),2)), fontname = 'Times New Roman',size=20)
    plt.savefig('HIZOA_2dswml_HIMF.pdf')#,rasterized=True,dpi=300)#,orientation='landscape')
    plt.show()


    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    f = plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axScatter.grid(b=True, which='both', color='0.65',linestyle=':')
    axHistx = plt.axes(rect_histx)
    axHistx.grid(b=True, which='both', color='0.65',linestyle=':')
    axHisty = plt.axes(rect_histy)
    axHisty.grid(b=True, which='both', color='0.65',linestyle=':')

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
#axScatter.scatter(mHI01,w01,c=Veff_final,edgecolor='none',cmap='PuBuGn')
    cax2 = f.add_axes([0.12, 0.32, 0.02, 0.4])
    ccc = np.array(Veff_final)
    d02= axScatter.scatter(mHI01,w01,c=np.ndarray.flatten(ccc),edgecolor='none',cmap='gray')#cmap='YlGnBu' or cmap=plt.cm.seismic
    cbar2 = f.colorbar(d02,cax=cax2)
    cbar2.set_label(r'log ($1/V_{eff})\/\/[\mathrm{Mpc^{-3}}]$', fontname = 'Times New Roman')#, size=20)
    axScatter.set_xlabel(r'log $M_{HI}\/[M_{\odot}]$', fontname = 'Times New Roman',size=20)
    axScatter.set_ylabel(r'log $W_{50}\/[\mathrm{km/s}]$', fontname = 'Times New Roman',size=20)

    axHistx.hist(mHI01,bins=np.arange(6.0,11.0,0.2),color='0.85',edgecolor='none')#,alpha=0.1)
    axHistx.yaxis.set_ticks(np.arange(0, 201, 50))
    axHistx.set_ylabel(r'$N$', fontname = 'Times New Roman',size=20)
    axHisty.hist(w01,bins=np.arange(1.4,3.0,0.1),color='0.85',edgecolor='none', orientation='horizontal')
    axHisty.xaxis.set_ticks(np.arange(0, 201, 50))
    axHisty.set_xlabel(r'$N$', fontname = 'Times New Roman',size=20)

    plt.savefig('2Dhist.pdf')
    plt.show()



#######################################
#Make bins
delM = 0.2
delW = 0.1
xedges, yedges = np.arange(6.0,11.0,0.2), np.arange(1.2,3.0,0.1)
#----------------------------------------------------------------------------------#


HIMF, HIMF_err, [alpha,m_s,phi_s], fit_cov = TWODSWML(mHI01,w01,speak01,flux01,dist01,area)
print ('alpha = '+str(round(alpha,2))+' +/- '+str(round(np.sqrt(fit_cov[0,0]),2)))
print ('M* = '+str(round(m_s,2))+' +/- '+str(round(np.sqrt(fit_cov[1,1]),2)))
print ('phi = '+str(round(phi_s,4))+' +/- '+str(round(np.sqrt(fit_cov[2,2]),4)))
print (alpha+1.0)
print (10**m_s)
print (phi_s)


plot_HIMF(HIMF,HIMF_err,[alpha,m_s,phi_s],fit_cov)

print (len(xedges),xedges)


