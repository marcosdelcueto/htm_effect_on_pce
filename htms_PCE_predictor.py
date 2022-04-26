#!/usr/bin/env python3
#################################################################################
import sys
import ast
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib import ticker, gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import differential_evolution

def main(inp):
    '''
    Main function assigning conditions and calling subroutines to train/test models

    Parameters
    ----------
    inp: str
        string with name of file with input options

    Returns
    -------
    rms: float
        root-mean square error value
    best_hyperparams: dict
        dictionary with values of hyperparameters
    '''
    # Read input options
    (inp_csv_file, ML, Neighbors, CV, kfold, 
     optimize_hyperparams, gamma_fp, gamma_mol, gamma_type, gamma_arch, gamma_dft, gamma_dope, alpha, 
     gamma_fp_lim, gamma_mol_lim, gamma_type_lim, gamma_arch_lim, gamma_dft_lim, gamma_dope_lim, alpha_lim, 
     xcols_fp, xcols_mol, xcols_type, xcols_arch, xcols_dft, xcols_dope, ycols, NCPU, diff_popsize, diff_tol, test_proportion, 
     print_plots) = read_initial_values(inp)
    # Read dataframe from csv
    df = pd.read_csv(inp_csv_file)
    # Create X,y from dataframe
    X, y = preprocess(df,xcols_fp,xcols_mol,xcols_type, xcols_arch, xcols_dft, xcols_dope, ycols)
    # if there is a test set, divide PCE range in 7 intervals and select random points to match population of each interval
    if test_proportion > 0.0:
        lower_bound_PCE=math.floor(min(y))
        upper_bound_PCE=math.ceil(max(y))
        PCE_intervals=7
        bins = np.linspace(lower_bound_PCE, upper_bound_PCE, PCE_intervals)
        y_binned = np.digitize(y, bins)
        X_cv, X_test, y_cv, y_test = train_test_split(X,y,test_size=test_proportion,shuffle=True,random_state=2022,stratify=y_binned)
        # set up labels
        labels_No=df['No.']
        _, _, labels_cv, labels_test = train_test_split(X,labels_No,test_size=test_proportion,shuffle=True,random_state=2022,stratify=y_binned)
        labels_cv=labels_cv.values
        labels_test=labels_test.values
    # if there is no test set, use all as training
    elif test_proportion ==0.0:
        X_cv = X
        y_cv = y
        X_test = None
        y_test = None
        labels_cv   = df['No.'].values
        labels_test = None
    else:
        print('ERROR: test_proportion should be zero or positive, but is:', test_proportion)
        sys.exit()
    # Set bounds for hyperparams optimization
    bounds = []
    if gamma_fp > 0:   bounds = bounds + [gamma_fp_lim]
    if gamma_mol > 0:  bounds = bounds + [gamma_mol_lim]
    if gamma_type > 0: bounds = bounds + [gamma_type_lim]
    if gamma_arch > 0: bounds = bounds + [gamma_arch_lim]
    if gamma_dft > 0:  bounds = bounds + [gamma_dft_lim]
    if gamma_dope > 0: bounds = bounds + [gamma_dope_lim]
    # kNN
    if ML == 'kNN':
        all_hyperparams = {'gamma_fp':gamma_fp, 'gamma_mol':gamma_mol, 'gamma_type':gamma_type, 'gamma_arch':gamma_arch, 'gamma_dft':gamma_dft, 'gamma_dope':gamma_dope, 'neighbor_value':Neighbors[0]}
    # KRR
    elif ML == 'KRR':
        all_hyperparams = {'gamma_fp':gamma_fp, 'gamma_mol':gamma_mol, 'gamma_type':gamma_type, 'gamma_arch':gamma_arch, 'gamma_dft':gamma_dft, 'gamma_dope':gamma_dope, 'alpha':alpha}
        bounds = bounds + [alpha_lim]
    # If not optimizing parameters
    if optimize_hyperparams == False:
        final_call=True
        best_hyperparams = dict(all_hyperparams)
        rms = func_ML(None,X_cv,X_test,y_cv,y_test,ML,all_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type, xcols_arch,xcols_dft, xcols_dope, ycols,final_call, print_plots,labels_cv,labels_test)
    # If optimizing parameters
    elif optimize_hyperparams == True:
        final_call=False
        # kNN
        if ML == 'kNN':
            for k in Neighbors:
                print('Start k=',k)
                all_hyperparams['neighbor_value']=k
                if (gamma_fp>0 and gamma_mol==0 and gamma_type==0 and gamma_arch==0 and gamma_dft==0 and gamma_dope==0) or (gamma_fp==0 and gamma_mol>0 and gamma_type==0 and gamma_arch==0 and gamma_dft==0 and gamma_dope==0) or (gamma_fp==0 and gamma_mol==0 and gamma_type>0 and gamma_arch==0 and gamma_dft==0 and gamma_dope==0) or (gamma_fp==0 and gamma_mol==0 and gamma_type==0 and gamma_arch>0 and gamma_dft==0 and gamma_dope==0) or (gamma_fp==0 and gamma_mol==0 and gamma_type==0 and gamma_arch==0 and gamma_dft>0 and gamma_dope==0) or (gamma_fp==0 and gamma_mol==0 and gamma_type==0 and gamma_arch==0 and gamma_dft==0 and gamma_dope>0):
                    rms = func_ML(None,X_cv,X_test,y_cv,y_test,ML,all_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type, xcols_arch,xcols_dft,xcols_dope, ycols,final_call, print_plots,labels_cv,labels_test)
                    opt_hyperparams = dict(all_hyperparams)
                else:
                    mini_args = (X_cv,X_test,y_cv,y_test, ML, all_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type, xcols_arch,xcols_dft,xcols_dope,ycols,final_call, print_plots,labels_cv,labels_test)
                    solver = differential_evolution(func_ML,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol,polish=False,workers=NCPU,updating='deferred')
                    opt_hyperparams = solver.x
                    rms = solver.fun
                if k == Neighbors[0]:
                    best_k = k
                    all_hyperparams['neighbor_value']=best_k
                    best_rms = rms
                    #########################################################################################
                    best_hyperparams = dict(all_hyperparams)
                    counter_input=0
                    for i in ['gamma_fp','gamma_mol','gamma_type','gamma_arch','gamma_dft','gamma_dope']:
                        if best_hyperparams[i] > 0:
                            best_hyperparams[i]=opt_hyperparams[counter_input]
                            counter_input=counter_input+1
                    #########################################################################################
                elif rms < best_rms:
                    best_k = k
                    all_hyperparams['neighbor_value']=best_k
                    best_hyperparams['neighbor_value']=best_k
                    best_rms = rms
                    counter_input=0
                    for i in ['gamma_fp','gamma_mol','gamma_type','gamma_arch','gamma_dft','gamma_dope']:
                        if best_hyperparams[i] > 0:
                            best_hyperparams[i]=opt_hyperparams[counter_input]
                            counter_input=counter_input+1
                    print(best_hyperparams)
        # KRR
        elif ML == 'KRR':
            ### call differential evolution algorithm ###
            initial_hyperparams=dict(all_hyperparams)
            mini_args = (X_cv,X_test,y_cv,y_test, ML, all_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type, xcols_arch,xcols_dft,xcols_dope,ycols,final_call, print_plots,labels_cv,labels_test)
            solver = differential_evolution(func_ML,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol,polish=False,workers=NCPU,updating='deferred')
            opt_hyperparams = solver.x
            best_rms = solver.fun
            best_hyperparams = dict(initial_hyperparams)
            counter_input=0
            for i in ['gamma_fp','gamma_mol','gamma_type','gamma_arch','gamma_dft','gamma_dope','alpha']:
                if best_hyperparams[i] > 0:
                    print('overwrite', i, 'with:', opt_hyperparams[counter_input])
                    best_hyperparams[i]=opt_hyperparams[counter_input]
                    counter_input=counter_input+1
        # Once optimized hyperparams are found, do final calculation using those values
        final_call=True
        print('###############################################')
        print('Doing final call with optimized hyperparameters')
        print('###############################################')
        rms = func_ML(None,X_cv,X_test,y_cv,y_test,ML,best_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type, xcols_arch,xcols_dft,xcols_dope,ycols,final_call, print_plots,labels_cv,labels_test)
    return rms, best_hyperparams

def read_initial_values(inp):
    '''
    Function that reads input values from file

    Parameters
    ----------
    inp: str
        string with name of file with input options

    Returns
    -------
    All input values
    '''
    # open input file
    input_file_name = inp
    f_in = open('%s' %input_file_name,'r')
    f1 = f_in.readlines()
    # initialize arrays
    input_info = []
    var_name = []
    var_value = []
    # read info before comments. Ignore commented lines and blank lines
    for line in f1:
        if not line.startswith("#") and line.strip():
            input_info.append(line.split('#',1)[0].strip())
    # read names and values of variables
    for i in range(len(input_info)):
        var_name.append(input_info[i].split('=')[0].strip())
        var_value.append(input_info[i].split('=')[1].strip())
    # close input file
    f_in.close()

    inp_csv_file = ast.literal_eval(var_value[var_name.index('inp_csv_file')])
    ML = ast.literal_eval(var_value[var_name.index('ML')])
    Neighbors = ast.literal_eval(var_value[var_name.index('Neighbors')])
    CV = ast.literal_eval(var_value[var_name.index('CV')])
    kfold = ast.literal_eval(var_value[var_name.index('kfold')])
    optimize_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_hyperparams')])
    gamma_fp = ast.literal_eval(var_value[var_name.index('gamma_fp')])
    gamma_mol = ast.literal_eval(var_value[var_name.index('gamma_mol')])
    gamma_type = ast.literal_eval(var_value[var_name.index('gamma_type')])
    gamma_arch = ast.literal_eval(var_value[var_name.index('gamma_arch')])
    gamma_dft = ast.literal_eval(var_value[var_name.index('gamma_dft')])
    gamma_dope = ast.literal_eval(var_value[var_name.index('gamma_dope')])
    alpha = ast.literal_eval(var_value[var_name.index('alpha')])
    gamma_fp_lim = ast.literal_eval(var_value[var_name.index('gamma_fp_lim')])
    gamma_mol_lim = ast.literal_eval(var_value[var_name.index('gamma_mol_lim')])
    gamma_type_lim = ast.literal_eval(var_value[var_name.index('gamma_type_lim')])
    gamma_arch_lim = ast.literal_eval(var_value[var_name.index('gamma_arch_lim')])
    gamma_dft_lim = ast.literal_eval(var_value[var_name.index('gamma_dft_lim')])
    gamma_dope_lim = ast.literal_eval(var_value[var_name.index('gamma_dope_lim')])
    alpha_lim = ast.literal_eval(var_value[var_name.index('alpha_lim')])
    xcols_fp = ast.literal_eval(var_value[var_name.index('xcols_fp')])
    xcols_mol = ast.literal_eval(var_value[var_name.index('xcols_mol')])
    xcols_type = ast.literal_eval(var_value[var_name.index('xcols_type')])
    xcols_arch = ast.literal_eval(var_value[var_name.index('xcols_arch')])
    xcols_dft = ast.literal_eval(var_value[var_name.index('xcols_dft')])
    xcols_dope = ast.literal_eval(var_value[var_name.index('xcols_dope')])
    ycols = ast.literal_eval(var_value[var_name.index('ycols')])
    NCPU = ast.literal_eval(var_value[var_name.index('NCPU')])
    diff_popsize = ast.literal_eval(var_value[var_name.index('diff_popsize')])
    diff_tol = ast.literal_eval(var_value[var_name.index('diff_tol')])
    test_proportion = ast.literal_eval(var_value[var_name.index('test_proportion')])
    print_plots = ast.literal_eval(var_value[var_name.index('print_plots')])
    f_in.close()

    return (inp_csv_file, ML, Neighbors, CV, kfold, optimize_hyperparams, gamma_fp, gamma_mol, 
            gamma_type, gamma_arch, gamma_dft, gamma_dope, alpha, gamma_fp_lim, gamma_mol_lim, gamma_type_lim, 
            gamma_arch_lim, gamma_dft_lim, gamma_dope_lim, alpha_lim, xcols_fp, xcols_mol, xcols_type, xcols_arch, 
            xcols_dft, xcols_dope, ycols, NCPU, diff_popsize, diff_tol, test_proportion, print_plots)

def preprocess(df,xcols_fp,xcols_mol,xcols_type, xcols_arch, xcols_dft, xcols_dope, ycols):
    '''
    Function that reads initial dataframe and returns data in desired format for ML: X, y

    Parameters
    ----------
    df: pd.dataframe
        dataframe with htm database
    xcols_fp: list
        list with fp descriptors
    xcols_mol: list
        list with molecular descriptors
    xcols_type: list
        list with types of perovskites
    xcols_arch: list
        list with architectures
    xcols_dft: list
        list with DFT descriptors
    xcols_dope: list
        list with dopants
    ycols: list
        list with target property label

    Returns
    -------
    X: np.array 
        array containing descriptors specified in xcols
    y: np.array 
        array containing target property specified in ycols
    '''     
    df['PCE (%)'] = df['PCE (%)'].str.replace('\'', '')
    df['PCE (%)'] = pd.to_numeric(df['PCE (%)'])
    #############################
    xcols = []
    xcols.append(xcols_fp)
    xcols.append(xcols_mol)
    xcols.append(xcols_type)
    xcols.append(xcols_arch)
    xcols.append(xcols_dft)
    xcols.append(xcols_dope)
    xcols = [item for sublist in xcols for item in sublist]
    #############################
    X=df[xcols].values
    y=df[ycols].values
    for i in range(len(X)):
        X1=X[i][0][1:-1].split(',')
        fp = [] 
        for j in range(len(X1)):
            fp.append(int(float(X1[j])))
        if len(fp) != 2048:
            print('ERROR: length of SMILES number %i does not have correct length' %i)
            sys.exit() 
        X[i][0]=fp
    y = [item for sublist in y for item in sublist]
    y = np.array(y)

    X_fp  = []
    X_mol = [[] for j in range(len(X))]
    X_type = [[] for j in range(len(X))]
    X_arch = [[] for j in range(len(X))]
    X_dft = [[] for j in range(len(X))]
    X_dope = [[] for j in range(len(X))]
    for i in range(len(X)):
        X_fp.append(X[i][0])
        for j in range(1,len(xcols_mol)+1):
            X_mol[i].append(X[i][j])
        for j in range(len(xcols_mol)+1,len(xcols_mol)+len(xcols_type)+1):
            X_type[i].append(X[i][j])
        for j in range(len(xcols_mol)+len(xcols_type)+1,len(xcols_mol)+len(xcols_type)+len(xcols_arch)+1):
            X_arch[i].append(X[i][j])
        for j in range(len(xcols_mol)+len(xcols_type)+len(xcols_arch)+1,len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+1):
            X_dft[i].append(X[i][j])
        for j in range(len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+1,len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+len(xcols_dope)+1):
            X_dope[i].append(X[i][j])
    xscaler = StandardScaler()
    X_mol = xscaler.fit_transform(X_mol) # scale molecular descriptors
    X_dft = xscaler.fit_transform(X_dft) # scale DFT descriptors
    X = np.c_[ X_fp,X_mol,X_type,X_arch,X_dft,X_dope]

    return X, y



def func_ML(hyperparams,X_cv,X_test,y_cv,y_test,ML,all_hyperparams,CV,kfold,xcols_fp,xcols_mol,xcols_type,xcols_arch,xcols_dft,xcols_dope,ycols,final_call,print_plots,labels_cv,labels_test):
    '''
    Creates sklearn ML objects and calls other functions to predict values and calculate rms

    Parameters
    ----------
    hyperparams: np.array
        array with values of hyperparameters during optimization
    X_cv: np.array
        array containing descriptors for cv set
    y_cv: np.array
        array containing target property for cv set
    X_test: np.array
        array containing descriptors for test set
    y_test: np.array
        array containing target property for test set
    ML: str
        label indicating ML model used
    CV: str
        label indicating type of cross-validation
    kfold: int
        number of folds in case we're using a k-fold cross-validation
    xcols_fp: list
        list with fp descriptors
    xcols_mol: list
        list with molecular descriptors
    xcols_type: list
        list with types of perovskites
    xcols_arch: list
        list with architectures
    xcols_dft: list
        list with DFT descriptors
    xcols_dope: list
        list with dopants
    ycols: list
        list with target property label
    final_call: bool
        whether this is the final call or not
    print_plots: bool
        whether to print plots or not
    labels_cv: np.array
        array with labels of data in CV set
    labels_test:np.array
        array with labels of data in test set

    Returns
    -------
    rms: float
        value of root-mean squared error
    '''
    ########## Build ML objects ##########
    # Update values of all_hyperparams depending on what values are given by diff_evol
    if hyperparams is not None:
        counter_input=0
        for i in ['gamma_fp','gamma_mol','gamma_type','gamma_arch','gamma_dft','gamma_dope']:
            if all_hyperparams[i] > 0:
                all_hyperparams[i]=hyperparams[counter_input]
                counter_input=counter_input+1
        if ML=='KRR':
            all_hyperparams['alpha']=hyperparams[counter_input]
            counter_input=counter_input+1
    # Set gamma values
    gamma_fp = all_hyperparams['gamma_fp']
    if gamma_fp < 0: gamma_fp = -gamma_fp
    gamma_mol = all_hyperparams['gamma_mol']
    if gamma_mol < 0: gamma_mol = -gamma_mol
    gamma_type = all_hyperparams['gamma_type']
    if gamma_type < 0: gamma_type = -gamma_type
    gamma_arch = all_hyperparams['gamma_arch']
    if gamma_arch < 0: gamma_arch = -gamma_arch
    gamma_dft = all_hyperparams['gamma_dft']
    if gamma_dft < 0: gamma_dft = -gamma_dft
    gamma_dope = all_hyperparams['gamma_dope']
    if gamma_dope < 0: gamma_dope = -gamma_dope
    # Negative value used to indicate that it is not optimized: return to positive value
    if ML=='KRR':
        alpha = all_hyperparams['alpha']
        if alpha < 0: alpha = -alpha

    if ML=='kNN':
        neighbor_value =  all_hyperparams['neighbor_value']
        ML_algorithm = KNeighborsRegressor(n_neighbors=neighbor_value, weights='distance', metric=total_dist,algorithm='brute',metric_params={'gamma_fp':gamma_fp,'gamma_mol':gamma_mol,'gamma_type':gamma_type,'gamma_arch':gamma_arch, 'gamma_dft':gamma_dft,'gamma_dope':gamma_dope, 'xcols_fp':xcols_fp,'xcols_mol':xcols_mol,'xcols_type':xcols_type,'xcols_arch':xcols_arch,'xcols_dft':xcols_dft,'xcols_dope':xcols_dope})
    elif ML=='KRR':
        ML_algorithm = KernelRidge(alpha=alpha, kernel=build_KRR_kernel(),kernel_params={'gamma_fp':gamma_fp,'gamma_mol':gamma_mol,'gamma_type':gamma_type,'gamma_arch':gamma_arch,'gamma_dft':gamma_dft,'gamma_dope':gamma_dope,'xcols_fp':xcols_fp,'xcols_mol':xcols_mol,'xcols_type':xcols_type,'xcols_arch':xcols_arch,'xcols_dft':xcols_dft,'xcols_dope':xcols_dope})
    # If NOT final call
    if final_call==False:
        ########## Predict values with k-fold / LOO cross-validation ##########
        y_cv_real, y_cv_predicted = kf_loo_cv(X_cv,y_cv,ML_algorithm,CV,kfold)
        ################## Predict test values #################
        if y_test is None:
            y_test_real = None
            y_test_predicted = None
        else:
            y_test_real, y_test_predicted = test_set(X_cv,X_test,y_cv,y_test,ML_algorithm)
        # Get metrics
        rms = get_pred_errors(y_cv_real,y_cv_predicted,y_test_real,y_test_predicted,ML_algorithm,ML)
    # If final call
    elif final_call==True:
        y_test_real = None
        y_test_predicted = None
        labels_plot=['Train']
        y_cv_real, y_cv_predicted = kf_loo_cv(X_cv,y_cv,ML_algorithm,CV,kfold)
        print('CV predictions')
        for i in range(len(y_cv_real)):
            print('PSC No.: %4i. Real PCE: %6.2f. Predicted PCE: %6.2f' %(labels_cv[i],y_cv_real[i],y_cv_predicted[i],))
        # If there is test
        if y_test is not None:
            labels_plot.append('Test')
            y_test_real, y_test_predicted = test_set(X_cv,X_test,y_cv,y_test,ML_algorithm)
            print('Test predictions')
            for i in range(len(y_test_real)):
                print('PSC No.: %4i. Real PCE: %6.2f. Predicted PCE: %6.2f' %(labels_test[i],y_test_real[i],y_test_predicted[i]))
        print("")
        ########## Plot predictions ##########
        if print_plots == True: do_plots(y_cv_real, y_cv_predicted, y_test_real, y_test_predicted,'Figure.png',labels_plot)
        ########## Calculate rmse of predicted values ##########
        rms = get_pred_errors(y_cv_real,y_cv_predicted,y_test_real,y_test_predicted,ML_algorithm,ML)
    return rms


def do_plots(x, y, x2, y2, plot_name,labels_plot):
    '''
    Function to do plots

    Parameters
    ----------
    x: np.array
        array with experimental PCE of train set
    y: np.array
        array with predicted PCE of train set
    x2: np.array
        array with experimental PCE of test set
    y2: np.array
        array with predicted PCE of test set
    plot_name: str
        name for png with plot
    labels_plot: list
        list containing the labels for plot
    '''
    # set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    ax.tick_params(axis='both', which='major', direction='in', labelsize=8, pad=2, length=5)
    # calculate metrics of train set
    r,_ = pearsonr(x, y)
    rho,_ = spearmanr(x, y)
    rmse  = math.sqrt(mean_squared_error(x,y))
    # set up max and min values of plot
    ma = np.max([x.max(), y.max()]) + 0.5
    mi = np.min([x.min(),y.min()]) - 0.5
    # options for plot_target_predictions
    ax.set_xlabel(r"Experimental PCE (%)", size=12, labelpad=2)
    ax.set_ylabel(r'Predicted PCE  (%)', size=12, labelpad=2)
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    ax.set_aspect('equal')
    ax.annotate(u'%5s: $r$ = %.2f. $rmse$ = %.1f%s' % (labels_plot[0],r,rmse,'%'), xy=(0.03,0.93), xycoords='axes fraction', size=12, color="C0")
    # plot train data
    ax.plot(np.arange(mi, ma + 0.5, 0.5), np.arange(mi, ma + 0.5, 0.5), color="k", ls="--")
    ax.scatter(x, y, color="C0")
    # plot test set data, if any
    if x2 is not None:
        r2,_ = pearsonr(x2, y2)
        rmse2  = math.sqrt(mean_squared_error(x2,y2))
        ax.scatter(x2, y2, color="C1")
        ax.annotate(u'%5s: $r$ = %.2f. $rmse$ = %.1f%s' % (labels_plot[1],r2,rmse2,'%'), xy=(0.03,0.87), xycoords='axes fraction', size=12, color="C1")
    # set up tick options
    xtickmaj = ticker.MaxNLocator(10)
    xtickmin = ticker.AutoMinorLocator(5)
    ytickmaj = ticker.MaxNLocator(10)
    ytickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=20, pad=10, length=2)
    # save plot into corresponding file
    plt.savefig(plot_name,dpi=600,bbox_inches='tight')
    plt.close()
    return


def kf_loo_cv(X,y,ML_algorithm,CV,kfold):
    '''
    Calculates the predicted values for a dataset {X,y}

    Parameters
    ----------
    X: np.array
        array containing descriptors
    y: np.array
        array containing target property
    ML_algorithm: sklearn object
        object containing the ML model structure
    CV: str
        label indicating type of cross-validation
    kfold: int
        number of folds in case we're using a k-fold cross-validation

    Returns
    -------
    y_real: np.array
        vector with actual values of target property
    y_predicted: np.array
        vector with predicted values of target property
    '''

    # Assign cross-validation: k-fold or LOO (k-fold random state fixed now for reproducibility)
    if CV=='kf':
        cv = KFold(n_splits=kfold,shuffle=True,random_state=2020)
    elif CV=='loo':
        cv = LeaveOneOut()
    # calculate predicted values
    y_predicted = cross_val_predict(ML_algorithm, X, y, cv=cv)
    y_real = y
    y_real = np.array(y_real)
    y_predicted = np.array(y_predicted)
    return y_real, y_predicted



def test_set(X_cv,X_test,y_cv,y_test,ML_algorithm):
    '''
    Calculates the predicted values for a dataset {X,y}

    Parameters
    ----------
    X_cv: np.array
        array containing descriptors of CV set
    X_test: np.array
        array containing descriptors of test set
    y_cv: np.array
        array containing target property of CV set
    y_test: np.array
        array containing target property of test set
    ML_algorithm: sklearn object
        object containing the ML model structure

    Returns
    -------
    y_real: np.array
        vector with actual values of target property
    y_predicted: np.array
        vector with predicted values of target property
    '''
    # calculate predicted values
    y_predicted = ML_algorithm.fit(X_cv, y_cv).predict(X_test)
    y_real = np.array(y_test)
    y_predicted = np.array(y_predicted)
    return y_real, y_predicted


def get_pred_errors(y_cv_real,y_cv_predicted,y_test_real,y_test_predicted,ML_algorithm,ML):
    '''
    Uses the real and predicted target property values to calculate error metrics

    Parameters
    ----------
    y_cv_real: np.array
        vector with actual values of target property for cv set
    y_cv_predicted: np.array
        vector with predicted values of target property for cv set
    y_test_real: np.array
        vector with actual values of target property for test set
    y_test_predicted: np.array
        vector with predicted values of target property for test set
    ML_algorithm: sklearn object
        object containing the ML model structure
    ML: str
        label indicating ML model used

    Returns
    -------
    rms_cv: float
        value of root-mean squared error for cv set
    '''
    rms_weights_order=0
    # Calculate metrics validation set
    if y_cv_real is None:
        r_cv    = 0.0
        rho_cv  = 0.0
        rms_cv    = 0.0
        errors_cv = 0.0
        median_error_cv = 0.0
    else:
        r_cv,_   = pearsonr(y_cv_real, y_cv_predicted)
        rho_cv,_ = spearmanr(y_cv_real, y_cv_predicted)
        weights = np.power(y_cv_real,rms_weights_order) / np.linalg.norm(np.square(y_cv_real))
        rms_cv  = math.sqrt(mean_squared_error(y_cv_real, y_cv_predicted, sample_weight=weights))
        errors_cv = abs(y_cv_real - y_cv_predicted)
        median_error_cv = np.median(errors_cv)
    # Calculate metrics test set
    if y_test_real is None:
        r_test            = 0.0
        rho_test          = 0.0
        rms_test          = 0.0
        errors_test       = 0.0
        median_error_test = 0.0
    else:
        r_test,_   = pearsonr(y_test_real, y_test_predicted)
        rho_test,_ = spearmanr(y_test_real, y_test_predicted)
        weights = np.power(y_test_real,rms_weights_order) / np.linalg.norm(np.square(y_test_real))
        rms_test  = math.sqrt(mean_squared_error(y_test_real, y_test_predicted, sample_weight=weights))
        errors_test = abs(y_test_real - y_test_predicted)
        median_error_test = np.median(errors_test)
    if ML=='kNN':
        k = ML_algorithm.get_params()['n_neighbors']
        gamma_fp =   ML_algorithm.get_params()['metric_params']['gamma_fp']
        gamma_mol =  ML_algorithm.get_params()['metric_params']['gamma_mol']
        gamma_type = ML_algorithm.get_params()['metric_params']['gamma_type']
        gamma_arch = ML_algorithm.get_params()['metric_params']['gamma_arch']
        gamma_dft = ML_algorithm.get_params()['metric_params']['gamma_dft']
        gamma_dope = ML_algorithm.get_params()['metric_params']['gamma_dope']
        print('k: %i, gamma_fp: %.6f, gamma_mol: %.6f, gamma_type: %.6f, gamma_arch: %.6f, gamma_dft: %.6f, gamma_dope: %.6f' %(k,gamma_fp,gamma_mol,gamma_type,gamma_arch,gamma_dft,gamma_dope))
        print('r_cv:   %.2f, rho_cv:   %.2f, rmse_cv:   %.2f, median_error_cv:   %.2f' %(r_cv,rho_cv,rms_cv,median_error_cv))
        print('r_test: %.2f, rho_test: %.2f, rmse_test: %.2f, median_error_test: %.2f' %(r_test,rho_test,rms_test,median_error_test))
        sys.stdout.flush()
    if ML=='KRR':
        alpha    = ML_algorithm.get_params()['alpha']
        gamma_fp = ML_algorithm.get_params()['kernel_params']['gamma_fp']
        gamma_mol = ML_algorithm.get_params()['kernel_params']['gamma_mol']
        gamma_type = ML_algorithm.get_params()['kernel_params']['gamma_type']
        gamma_arch = ML_algorithm.get_params()['kernel_params']['gamma_arch']
        gamma_dft = ML_algorithm.get_params()['kernel_params']['gamma_dft']
        gamma_dope = ML_algorithm.get_params()['kernel_params']['gamma_dope']
        print('gamma_fp: %.6f, gamma_mol: %.6f, gamma_type: %.6f, gamma_arch: %.6f, gamma_dft: %.6f, gamma_dope: %.6f,alpha: %.6f' %(gamma_fp,gamma_mol,gamma_type,gamma_arch,gamma_dft,gamma_dope,alpha))
        print('r_cv  : %.2f, rho_cv:   %.2f, rmse_cv:   %.2f, median_error_cv:   %.2f' %(r_cv,rho_cv,rms_cv,median_error_cv))
        print('r_test: %.2f, rho_test: %.2f, rmse_test: %.2f, median_error_test: %.2f' %(r_test,rho_test,rms_test,median_error_test))
        sys.stdout.flush()
    return rms_cv


def total_dist(x1,x2,gamma_fp,gamma_mol,gamma_type,gamma_arch,gamma_dft,gamma_dope,xcols_fp,xcols_mol,xcols_type,xcols_arch,xcols_dft,xcols_dope):
    '''
    Function to calculate tanimoto distance between two fingerprint vectors

    Parameters
    ----------
    x1: np.array
        data point
    x2: np.array
        data point
    gamma_fp: float
        value of hyperparameter gamma_fp
    gamma_mol: float
        value of hyperparameter gamma_mol
    gamma_type: float
        value of hyperparameter gamma_type
    gamma_arch: float
        value of hyperparameter gamma_arch
    gamma_dft: float
        value of hyperparameter gamma_dft
    gamma_dope: float
        value of hyperparameter gamma_dope
    xcols_fp: list
        list with fp descriptors
    xcols_mol: list
        list with molecular descriptors
    xcols_type: list
        list with types of perovskites
    xcols_arch: list
        list with architectures
    xcols_dft: list
        list with DFT descriptors
    xcols_dope: list
        list with dopants

    Returns
    -------
    D: float
        Total distance
    '''
    # sanity check
    if x1.ndim != 1 or x2.ndim != 1:
        print('ERROR: Custom metric was expecting 1D vectors!')
        sys.exit()
    # Fingerprint Distance
    x1_fp  = x1[0:2048]
    x2_fp  = x2[0:2048]
    D_fp = tanimoto_dist(x1_fp,x2_fp)
    # Molecular Descriptors Distance
    x1_mol = x1[2048:2048+len(xcols_mol)]
    x2_mol = x2[2048:2048+len(xcols_mol)]
    D_mol = euclid_dist(x1_mol, x2_mol)
    # Perovskite Type Distance
    x1_type = x1[2048+len(xcols_mol):2048+len(xcols_mol)+len(xcols_type)]
    x2_type = x2[2048+len(xcols_mol):2048+len(xcols_mol)+len(xcols_type)]
    D_type = euclid_dist(x1_type, x2_type)
    # Perovskite Type Distance
    x1_arch = x1[2048+len(xcols_mol)+len(xcols_type):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)]
    x2_arch = x2[2048+len(xcols_mol)+len(xcols_type):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)]
    D_arch = euclid_dist(x1_arch, x2_arch)
    # DFT Descriptors Distance
    x1_dft = x1[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)]
    x2_dft = x2[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)]
    D_dft = euclid_dist(x1_dft, x2_dft)
    # Dopant Distance
    x1_dope = x1[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+len(xcols_dope)]
    x2_dope = x2[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+len(xcols_dope)]
    D_dope = euclid_dist(x1_dope, x2_dope)
    # Total Distance
    D = gamma_fp * D_fp + gamma_mol * D_mol + gamma_type * D_type + gamma_arch * D_arch + gamma_dft * D_dft + gamma_dope * D_dope
    print('TEST distances:', D_fp, D_mol, D_type, D_arch, D_dft, D_dope, '. Total:',D)
    return D


def tanimoto_dist(x1,x2):
    '''
    Function to calculate tanimoto distance between two fingerprint vectors

    Parameters
    ----------
    x1: np.array
        data point
    x2: np.array
        data point

    Returns
    -------
    D: float
        Distance using Tanimoto similarity index
    '''
    T = ( np.dot(np.transpose(x1),x2) ) / ( np.dot(np.transpose(x1),x1) + np.dot(np.transpose(x2),x2) - np.dot(np.transpose(x1),x2) )
    D = 1-T
    return D


def euclid_dist(x1,x2):
    '''
    Calculates Euclidean distance

    Parameters
    ----------
    x1: np.array
        data point
    x2: np.array
        data point

    Returns
    -------
    D: float
        Euclidean distance
    '''
    # Calculate Euclidean Distance
    D = np.linalg.norm(x1-x2)
    return D


def build_KRR_kernel():
    '''
    Returns
    -------
    KRR_kernel: callable
        function to compute the Tanimoto kernel matrix terms
    '''
    def KRR_kernel(x1, x2, gamma_fp,gamma_mol,gamma_type,gamma_arch,gamma_dft,gamma_dope,xcols_fp,xcols_mol,xcols_type,xcols_arch,xcols_dft,xcols_dope):
        '''
        Function to compute a Tanimoto kernel matrix term(KRR)

        Parameters
        ----------
        x1: np.array
            data point
        x2: np.array
            data point
        gamma_fp: float
            gamma_fp hyperparameter
        gamma_mol: float
            gamma_mol hyperparameter
        gamma_type: float
            value of hyperparameter gamma_type
        gamma_arch: float
            value of hyperparameter gamma_arch
        gamma_dft: float
            value of hyperparameter gamma_dft
        gamma_dope: float
            value of hyperparameter gamma_dope
        xcols_fp: list
            list with fp descriptors
        xcols_mol: list
            list with molecular descriptors
        xcols_type: list
            list with types of perovskites
        xcols_arch: list
            list with architectures
        xcols_dft: list
            list with DFT descriptors
        xcols_dope: list
            list with dopants

        Returns
        -------
        K: np.float.
            Kernel matrix element.
        '''
        # sanity check
        if x1.ndim != 1 or x2.ndim != 1:
            print('ERROR: KRR kernel was expecting 1D vectors!')
            sys.exit()
        # Fingerprint Distance
        x1_fp  = x1[0:2048]
        x2_fp  = x2[0:2048]
        # Molecular Descriptors Distance
        x1_mol = x1[2048:2048+len(xcols_mol)]
        x2_mol = x2[2048:2048+len(xcols_mol)]
        # Perovskite Type Distance
        x1_type = x1[2048+len(xcols_mol):2048+len(xcols_mol)+len(xcols_type)]
        x2_type = x2[2048+len(xcols_mol):2048+len(xcols_mol)+len(xcols_type)]
        # Perovskite Type Distance
        x1_arch = x1[2048+len(xcols_mol)+len(xcols_type):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)]
        x2_arch = x2[2048+len(xcols_mol)+len(xcols_type):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)]
        # DFT Descriptors Distance
        x1_dft = x1[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)]
        x2_dft = x2[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)]
        # Dopant Distance
        x1_dope = x1[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+len(xcols_dope)]
        x2_dope = x2[2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft):2048+len(xcols_mol)+len(xcols_type)+len(xcols_arch)+len(xcols_dft)+len(xcols_dope)]
        # initialize kernel values
        K_fp   = 1.0
        K_mol  = 1.0
        K_type = 1.0
        K_arch = 1.0
        K_dft  = 1.0
        K_dope = 1.0
        # calculate Tanimoto kernel
        if gamma_fp != 0:
            D_fp = tanimoto_dist(x1_fp,x2_fp)
            K_fp = np.exp(-gamma_fp * (D_fp)**2)
        # calculate Molecular kernel
        if gamma_mol != 0:
            D_mol = euclid_dist(x1_mol,x2_mol)
            K_mol = np.exp(-gamma_mol * (D_mol)**2)
        # calculate Type kernel
        if gamma_type != 0:
            D_type = euclid_dist(x1_type,x2_type)
            K_type = np.exp(-gamma_type * (D_type)**2)
        # calculate Arch kernel
        if gamma_arch != 0:
            D_arch = euclid_dist(x1_arch,x2_arch)
            K_arch = np.exp(-gamma_arch * (D_arch)**2)
        # calculate DFT kernel
        if gamma_dft != 0:
            D_dft = euclid_dist(x1_dft,x2_dft)
            K_dft = np.exp(-gamma_dft * (D_dft)**2)
        # calculate Dope kernel
        if gamma_dope != 0:
            D_dope = euclid_dist(x1_dope,x2_dope)
            K_dope = np.exp(-gamma_dope * (D_dope)**2)
        # total kernel
        K = K_fp * K_mol * K_type * K_arch * K_dft * K_dope
        return K
    return KRR_kernel


#################################################################################
if __name__ == '__main__':
    inp = 'input_htms_PCE_predictor.inp'
    if len(sys.argv)==2:
        print('Overriding input by argument')
        inp = sys.argv[1]
    elif  len(sys.argv)>2:
        print('ERROR: no more than one argument accepted')
        sys.exit()
    start = time()
    main(inp)
    finish = time()-start
    print('Process took %0.4f seconds' %finish)
#################################################################################
