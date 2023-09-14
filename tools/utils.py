
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def gridplot(datas:list,data_names:list,true_para:list=None,para_mins:list=None,para_maxs:list=None,para_names:list=None,figsize:tuple=(10,10)):
    # plt.rcParams.update({'font.size': figsize[0]*1.5})
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams.update({'label.fontsize': figsize[0]*1.5})
    # plt.rcParams.update({'xtick.labelsize': figsize[0]*1})
    # plt.rcParams.update({'ytick.labelsize': figsize[0]*1})
    cmaps=['Blues','Reds','BuGn','Greens','viridis',"YlOrBr"]
    colors=['tab:blue','tab:red','tab:green']
    alphas=[1,0.8,1]
    patches=[]
    N=len(datas)
    SMALL_SIZE = figsize[0]//2
    MEDIUM_SIZE = figsize[0]
    BIGGER_SIZE = figsize[0]*1.5

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig=plt.figure(figsize=figsize,facecolor=(1,1,1))
    v_line_fac=1.1
    para_space_fac=0.5
    tick_number=5
    tick_round=2

    N_dim=datas[0].shape[1]
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    

    for i in range(N_dim):
        for j in range(i+1):

            if para_mins is not None and para_maxs is not None:
                x_min=para_mins[j]
                x_max=para_maxs[j]
            else:
                x_min=np.min(data[:,j])
                x_max=np.max(data[:,j])
                x_min, x_max = x_min-(x_max-x_min)*para_space_fac,x_max+(x_max-x_min)*para_space_fac
            
            plt.subplot(N_dim,N_dim,i*N_dim+j+1)
            if j==i:
                height=0
                for idx,data in enumerate(datas):
                    ax = sns.kdeplot(data[:,i],color=colors[idx],common_norm=False)
                    
                    height = np.maximum(np.max(ax.lines[idx].get_ydata()),height)
    
                
                if true_para is not None:
                    plt.vlines(true_para[i], ymin=0, ymax=height*v_line_fac,colors='black',linestyles='dashed')

                plt.xlim(x_min,x_max)
                plt.ylim(0,height*v_line_fac)
                plt.yticks([], [])
                plt.ylabel("")
            else:
                for idx,data in enumerate(datas):
                    ax = sns.kdeplot(x=data[:,j],y=data[:,i],levels=[0.05,0.32,1],common_norm=False,cmap=cmaps[idx],shade=True,grid_size=50,alpha=alphas[idx])
                    ax = sns.kdeplot(x=data[:,j],y=data[:,i],levels=[0.05,0.32,1],common_norm=False,color=colors[idx],shade=False,grid_size=50,alpha=0.6)
                    patches.append(mpatches.Patch(color=colors[idx],label=data_names[idx]))
#                 plt.legend()
                if para_mins is not None and para_maxs is not None:
                    y_min=para_mins[i]
                    y_max=para_maxs[i]
                else:
                    y_min=np.min(data[:,i])
                    y_max=np.max(data[:,i])
                    y_min, y_max = y_min-(y_max-y_min)*para_space_fac,y_max+(y_max-y_min)*para_space_fac
                if true_para is not None:
                    plt.vlines(true_para[j], ymin=y_min, ymax=y_max,colors='black',linestyles='dashed')
                    plt.hlines(true_para[i], xmin=x_min, xmax=x_max,colors='black',linestyles='dashed')
                    plt.scatter(true_para[j],true_para[i],s=figsize[0],c='black',marker='s')
                plt.xlim(x_min,x_max)
                plt.ylim(y_min,y_max)

                if not (j==0):
                    plt.yticks([], [])
                else:

#                     plt.yticks(np.linspace(y_min,y_max,tick_number)[1:-1])
                    plt.yticks(np.linspace(round(y_min,tick_round-1),round(y_max,tick_round-1),tick_number)[1:-1])
                    if para_names is None:
                        plt.ylabel(f"$param_{i}$")
                    else:
                        plt.ylabel(f"${para_names[i]}$")

            if not (i==N_dim-1):
                plt.xticks([], [])
            else:
                plt.xticks(np.linspace(round(x_min,tick_round-1),round(x_max,tick_round-1),tick_number)[1:-1])
                if para_names is None:
                    plt.xlabel(f"$param_{j}$")
                else:
                    plt.xlabel(f"${para_names[j]}$")
    fig.legend(handles=patches,loc=(0.83,0.88))
    return fig

# def gridplot(data,true_para:list=None,para_mins:list=None,para_maxs:list=None,para_names:list=None,figsize:tuple=(10,10)):
#     # plt.rcParams.update({'font.size': figsize[0]*1.5})
#     # plt.rcParams["font.family"] = "Times New Roman"
#     # plt.rcParams.update({'label.fontsize': figsize[0]*1.5})
#     # plt.rcParams.update({'xtick.labelsize': figsize[0]*1})
#     # plt.rcParams.update({'ytick.labelsize': figsize[0]*1})

#     SMALL_SIZE = figsize[0]//2
#     MEDIUM_SIZE = figsize[0]
#     BIGGER_SIZE = figsize[0]*1.5

#     plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     plt.figure(figsize=figsize)
#     v_line_fac=1.1
#     para_space_fac=0.5
#     tick_number=5

#     N_dim=data.shape[1]
#     plt.subplots_adjust(wspace=0.0,hspace=0.0)

#     for i in range(N_dim):
#         for j in range(i+1):

#             if para_mins is not None and para_maxs is not None:
#                 x_min=para_mins[j]
#                 x_max=para_maxs[j]
#             else:
#                 x_min=np.min(data[:,j])
#                 x_max=np.max(data[:,j])
#                 x_min, x_max = x_min-(x_max-x_min)*para_space_fac,x_max+(x_max-x_min)*para_space_fac
            
#             plt.subplot(N_dim,N_dim,i*N_dim+j+1)
#             if j==i:
#                 ax = sns.kdeplot(data[:,i])
#                 height = np.max(ax.lines[0].get_ydata())
                
#                 if true_para is not None:
#                     plt.vlines(true_para[i], ymin=0, ymax=height*v_line_fac,colors='r',linestyles='dashed')

#                 plt.xlim(x_min,x_max)
#                 plt.ylim(0,height*v_line_fac)
#                 plt.yticks([], [])
#                 plt.ylabel("")
#             else:
#                 ax = sns.kdeplot(x=data[:,j],y=data[:,i],levels=[0.05,0.32,1],shade=True,grid_size=50)
#                 if para_mins is not None and para_maxs is not None:
#                     y_min=para_mins[i]
#                     y_max=para_maxs[i]
#                 else:
#                     y_min=np.min(data[:,i])
#                     y_max=np.max(data[:,i])
#                     y_min, y_max = y_min-(y_max-y_min)*para_space_fac,y_max+(y_max-y_min)*para_space_fac
#                 if true_para is not None:
#                     plt.vlines(true_para[j], ymin=y_min, ymax=y_max,colors='r',linestyles='dashed')
#                     plt.hlines(true_para[i], xmin=x_min, xmax=x_max,colors='r',linestyles='dashed')
#                     plt.scatter(true_para[j],true_para[i],s=figsize[0],c='r',marker='s')
#                 plt.xlim(x_min,x_max)
#                 plt.ylim(y_min,y_max)

#                 if not (j==0):
#                     plt.yticks([], [])
#                 else:

#                     plt.yticks(np.linspace(y_min,y_max,tick_number)[1:-1])

#                     if para_names is None:
#                         plt.ylabel(f"$param_{i}$")
#                     else:
#                         plt.ylabel(f"${para_names[i]}$")

#             if not (i==N_dim-1):
#                 plt.xticks([], [])
#             else:
#                 plt.xticks(np.linspace(x_min,x_max,tick_number)[1:-1])
#                 if para_names is None:
#                     plt.xlabel(f"$param_{j}$")
#                 else:
#                     plt.xlabel(f"${para_names[j]}$")
