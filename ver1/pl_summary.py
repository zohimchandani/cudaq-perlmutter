#!/usr/bin/env python3
""" 
 Kibble-Zurek scaling analysis

Meony plot

Assumes data were regressed to 0-readout noise using ZNE

YAML INPUT:   ./kzOct23Final//sum_confB_eps10_0.04.yaml  
OUTPUT: same

Output: K-Z slope and wall density variance plots
    

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from time import  time
import sys,os
from pprint import pprint
import matplotlib.pyplot as plt

'''
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_miscIO import read_yaml
from FitterReadErrZNE import FitterReadErrZNE
from toolbox.ModelLinear import ModelLinear
from toolbox.ModelQuadratic import ModelQuadratic
from toolbox.Util_ahs import wall_count_mitig_milan_202407, wall_variance_milan_202407
from toolbox.UAwsQuEra_job import flatten_ranked_hexpatt_array

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec

# fitting tau_Q
from toolbox.Util_Fitter  import  fit_data_model
from toolbox.ModelPower import ModelPower
'''
import yaml


import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--inpPath",default='/pscratch/sd/b/balewski/tmp_cudaq/',help="all input data")
    parser.add_argument("--outPath",default='out/',help="all outputs")
    parser.add_argument('-i',"--inpName",  default='sum_confB_eps10_0.05',help='[.yaml] name of input   experiment ')
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-O", "--observable",  default='wvar',choices=['wnum','wvar'] ,help=" select observable")

    args = parser.parse_args()
    
    args.showPlots=''.join(args.showPlots)
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)

    return args

#............................
#............................
#............................
class Plotter():
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        self.styD={'wall_num':['blue','lightblue','o'], 'wall_var':['red','lightsalmon','^']}
        self.styE={'hi':['green','*',2],'mi':['blue','o',1],'lo':['sienna','x',0]}
        
#...!...!....................
    def XXcreate_multi_canvas_layout(self,figId): 
        """
        Create a multi-canvas layout with specified dimensions and axes.
        Returns:
        tuple: A tuple containing the figure and axes (axl, axr).
        """
        figId=self.smart_append(figId)
        kwargs={'num':figId,'facecolor':'white', 'figsize':(5, 8)}

        # Create a figure with specified dimensions
        fig = self.plt.figure(**kwargs)

        # Define the grid layout
        nrow,ncol=4,1
        gs = GridSpec(nrow,ncol, figure=fig)

        # Create axes
        axt = fig.add_subplot(gs[:3, 0])  # top
        axb = fig.add_subplot(gs[3:,0])  # bottom

        # Hide x-axis labels for axr1 and axr2 to avoid label overlapping
        #self.plt.setp(axr1.get_xticklabels(), visible=False)

        return fig, (axt, axb)

#...!...!....................
    def XX_axt_decoration(self,axt,md):
        axt.set_xscale('log')
        axt.set_yscale('log')
        axt.grid()
        znm=md['zne_mit']
        txt='mit readErr  p1m0: %.2f  p0m1: %.2f'%(znm['prep1meas0'],znm['prep0meas1'])
        axt.text(0.05,0.75 ,txt,transform=axt.transAxes,color='m',fontsize=10)
       
        axt.set_xticks([2, 5, 10, 20, 50])
        axt.get_xaxis().set_major_formatter(ScalarFormatter())  

        axt.set_yticks([ 1,2,4,6, 8, 10, 15, 20,30])
        axt.get_yaxis().set_major_formatter(ScalarFormatter())  
        axt.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
        
#...!...!....................
    def XX_plot_syst_error(self,axb,md, bigD,obsL=['wall_num','wall_var']):
        tauQ=bigD["tauQ_MHz/us"]
        for obs in obsL:
            dCol,dCol2,fmt=self.styD[obs]
            valHi=bigD[obs+'_hi'][:,0]
            valLo=bigD[obs+'_lo'][:,0]
            delHalf=np.abs(valHi-valLo)/2
            axb.plot(tauQ,delHalf,color=dCol,label=obs)
            
        axb.set_xscale('log')
        
        axb.grid()
        axb.set(title='syst error    '+md['short_name'])
        axb.legend(loc='lower center')
        axb.set_xticks([2, 5, 10, 20, 50])
        axb.get_xaxis().set_major_formatter(ScalarFormatter())
        axb.set_ylim(0,)
        
#...!...!....................
    def circ_timewall_density_and_variance(self,md,bigD,figId=2,axs=None,systFill=True):
        if axs==None:
            fig, (axt, axb)=self.create_multi_canvas_layout(figId)
        else:
            axt,axb=axs
            
        xLab=r"$\tau_Q ~(MHz/\mu s)$"
        tLab='ZNE corrected K-Z data    %s'%(md['short_name'])

        tauQ=bigD["tauQ_MHz/us"]
        for obs in ['wall_num','wall_var']:
            valeV=bigD[obs+'_mi']
            dCol,dCol2,fmt=self.styD[obs]
            axt.errorbar(tauQ, valeV[:,0] , yerr= valeV[:,1],fmt=fmt,color=dCol,markersize=3,linewidth=1.5,label=obs+'+stat')
            valHi=bigD[obs+'_hi'][:,0]
            valLo=bigD[obs+'_lo'][:,0]
            if systFill: # Fill the area between valLo and valHi
                axt.fill_between(tauQ, valLo, valHi, color=dCol2, alpha=0.5, rasterized=True,label=obs+' syst')
    
        axt.set(ylabel='',xlabel=xLab,title=tLab)
        axt.legend(loc='upper left',title='Observables')

        self._axt_decoration(axt,md['mi'])        
        self._plot_syst_error(axb, md,bigD)

    #...!...!....................
    def wall_number_fit(self,md, bigD,figId=1,axs=None):
        if axs==None:
            fig, (axt, axb)=self.create_multi_canvas_layout(figId)
        else:
            axt,axb=axs
            
        obs='wall_num'
        yLab="avr wall density number"
        xLab=r"$\tau_Q ~(MHz/\mu s)$"
        tLab='ZNE corrected K-Z fit    %s'%(md['short_name'])

        hlmL=['lo','mi','hi']
        for key  in hlmL:            
            fmd=md[key]['fit_result'] #  fit data + fit func
            tauQ=bigD["tauQ_MHz/us"]        
            valeV=bigD[obs+'_'+key]
            assert valeV.shape[0] == tauQ.shape[0]

            [dCol,fmt,yOff]=self.styE[key]
            axt.errorbar(tauQ, valeV[:,0] , yerr= valeV[:,1],fmt=fmt,color=dCol,markersize=3,linewidth=1.5,label='eps10_'+key)

            ftag='lmfit_%s_%s'%(obs,key)
            dataX=bigD[ftag+'_x'] # MHz/us
            dataP=bigD[ftag+'_y']  # <n>  PE
            fitY= bigD[ftag+'_f']   # <n> : avr wall density
            
            axt.plot(dataX,fitY,'-',color=dCol)
            #pprint(fmd)
            # ... beautification
            f_mu=fmd['fitPar']['MU']
            redchi=fmd['fitQA']['redchi'][0]
            txt1=r'%s:  $\mu =  %.3f\pm  %.3f$'%(key,f_mu[0], f_mu[1])
            txt1+=' chi2/ndf=%.1f'%(redchi)
            axt.text(0.05,0.7-yOff/25,txt1,transform=axt.transAxes,color=dCol,fontsize=10)
        axt.set(ylabel='',xlabel=xLab,title=tLab)
        lTit=r'%s fit: $a \cdot (\tau_Q)^{\mu}$'%obs
        axt.legend(loc='upper left',title=lTit)
        self._axt_decoration(axt,md['mi'])
        if 'densLR' in md:   axt.set_ylim(tuple(md['densLR']))

        self._plot_syst_error(axb, md,bigD,['wall_num'])

#...!...!....................
def extract_data(dataD):

    tableD=dataD['table']
    tagL=sorted(tableD)

    tauV=[]; wnumV=[]; wvarV=[]
    for tag in tagL:
        rec=tableD[tag]
        evolTime=rec['evol_time_us']
        ramp2T=sum(rec['rabi_ramp_time_us'])
        a,b=rec['detune_delta_MHz']
        detuneTime=evolTime-ramp2T  # (us)
        tauQ=(b-a)/detuneTime  # (MHz/us)

        tauV.append(tauQ)
        wnumV.append(rec['wnum:lin']['mit'])
        wvarV.append(rec['wvar:quad']['mit'])
        
    return {'tauQ_MHz/us':np.array(tauV), 'wall_num':np.array(wnumV), 'wall_var':np.array(wvarV)}


#...!...!..................
def plot_circ_time(dataD,figN='aaa.png'):
    fig, ax = plt.subplots(figsize=(7, 9))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    md=dataD['meta']
    tableD=dataD['table']
    tagL0=list(tableD)
    tagL=sorted(tagL0, key=lambda x: int(x[1:]))
    print('tt',tagL)
    for i,tag in enumerate(tagL):
        rec=tableD[tag]
        xV=[]; yV=[]
        for nq in rec:
            tV=rec[nq]
            n=len(tV)
            xV+=[nq]*n
            yV+=tV
        ax.plot(xV,yV, linestyle='-',marker=markers[i],label=tag)
        
    ax.legend(title='num GPUs')
    tit='Perlmutter, 2024_08, random %d CX, %d shots, target=%s'%(md['nom_num_cx'], md['num_shot'],md['cudaq_target'])
    ax.set(xlabel='num qubits in circuit',ylabel='circ time (sec)',title=tit)
    ax.grid(True)

    # .... figure is done ...
    fig.savefig(figN, format='png')
    print('saved:',figN)
    # Show the plot
    plt.show()

#...!...!....................
def readOne(inpF,dataD,verb=1,nom_ncx=3000):
    #print('iii',inpF)
    assert os.path.exists(inpF)
    ymlFd = open(inpF, 'r')
    D=yaml.load( ymlFd, Loader=yaml.CLoader)
    ymlFd.close()
    #pprint(D)
    
    tag1='G%d'%D['num_rank']
    tag2=D['num_qubit']
    
    if tag1 not in dataD:
        dataD[tag1]={}
        md={ xx:D[xx] for xx in [ 'cudaq_target', 'num_shot'] }
        md['nom_num_cx']=nom_ncx
        dataD['meta']=md
        

    if tag2 not in dataD[tag1] : dataD[tag1][tag2]=[]
    circT=D['circ_time']/D['num_cx']*nom_ncx
    dataD[tag1][tag2].append(circT)

#...!...!....................
def find_yaml_files(directory_path, vetoL=None):
    """
    Scans the specified directory for all files with a .h5 extension,
    rejecting files whose names contain any of the specified veto strings.

    Args:
    directory_path (str): The path to the directory to scan.
    vetoL (list): A list of strings. Files containing any of these strings in their names will be rejected.

    Returns:
    list: A list of paths to the .yaml files found in the directory, excluding vetoed files.
    """
    if vetoL is None:
        vetoL = []

    h5_files = []
    #print('ff earch path: ',directory_path)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.yaml') and not any(veto in file for veto in vetoL):
                h5_files.append(os.path.join(root, file))
    h5_files.sort()
    return h5_files



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    np.set_printoptions(precision=3, suppress=True)


    jobL=[ 29490833, 29490834,29490836,29490837,29490838, 29490839 , 29639979, 29639980, 29639981
]
    fileL=[]
    for jobId in jobL:
        path2=os.path.join(args.inpPath,str(jobId))
        fileL+=find_yaml_files( path2)

    nInp=len(fileL)
    assert nInp>0
    print('found %d input files, e.g.: '%(nInp),fileL[0])
    
    motherD={}
    for i,fileN in enumerate(fileL):
        #print(i,fileN)
        readOne(fileN,motherD,i==0)        
        #if i>6: break
    #... all data are in
    #pprint(motherD)


    # repack  data
    md=motherD.pop('meta')
    dataD={'table':motherD, 'meta':md}
    
    print('\nM: dump %d mit data:'%(len(motherD)))
    pprint(dataD)
    print('M:done')

    if 0:
        #...... WRITE  JOB META-DATA .........
        outF='kzsum_%s_%s.yaml'%(expConf,args.sufixPath)
        outFF=os.path.join(args.outPath,outF)
        write_yaml(dataD,outFF)

    plot_circ_time(dataD)
   
