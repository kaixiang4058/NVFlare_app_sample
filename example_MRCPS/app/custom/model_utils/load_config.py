import argparse, pprint
import os
import yaml

#model config load
def load_modelcfg(cfgpath):
    """
    Load config from traincfg.yaml 
    """
    custom_dir ="" # modify
    with open(cfgpath, 'r') as fp:
        traincfg = yaml.load(fp, Loader=yaml.FullLoader)
 
    traincfg['modelname'] = 'MRCPS'

    if 'cps' in traincfg['sslset']['type'] \
        and 'branch2' not in traincfg:
        traincfg['branch2'] = traincfg['branch1']

    #load branch model yaml (add info in model.config)
    with open(custom_dir+traincfg['branch1'], 'r') as fp: #modify
        modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
    modelcfg['model.classes'] = len(traincfg['classes'])                            # class num
    modelcfg['model_seed'] = traincfg['expset']['model_seed']                       # model random seed
    if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:   # multiscale size
        modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
    traincfg['branch1'] = modelcfg                                                  #set branch modelcfg 
    traincfg['modelnum'] = 1

    if 'branch2' in traincfg:
        with open(custom_dir+traincfg['branch2'], 'r') as fp: #modify
            modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
        modelcfg['model.classes'] = len(traincfg['classes'])
        modelcfg['model_seed'] = 2 * traincfg['expset']['model_seed']
        if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:
            modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
        traincfg['branch2'] = modelcfg
        traincfg['modelnum'] = 2
        
    if 'branch3' in traincfg:
        with open(custom_dir+traincfg['branch3'], 'r') as fp: #modify
            modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
        modelcfg['model.classes'] = len(traincfg['classes'])
        modelcfg['model_seed'] = traincfg['expset']['model_seed']
        if 'model.lrscale' in modelcfg and traincfg['expset']['lrratio'] is not None:
            modelcfg['model.lrscale'] = traincfg['expset']['lrratio']
        traincfg['branch3'] = modelcfg
        traincfg['modelnum'] = 3
    

    # expname
    note = traincfg['expname']
    exp_seed = traincfg['expset']['exp_seed']
    epochs = traincfg['expset']['epochs']
    traincfg['expname'] = traincfg['modelname'] + traincfg['sslset']['type'] + '_' \
                          + f"_sd{exp_seed}_e{epochs}"
    
    if 'sda' in traincfg['traindl'] and traincfg['traindl']['sda'] == True:
        traincfg['expname'] += '_s'

    if note != "":
        traincfg['expname'] += f"-{note}"

    # root path setting
    fold = traincfg['expset']['fold']
    if 'fversion' in traincfg['expset'].keys():
        fold = f"{fold}_v{traincfg['expset']['fversion']}"
    # traincfg['rootset']['pklroot_train'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_train'])
    # traincfg['rootset']['pklroot_test'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_test'])
    # traincfg['rootset']['savepath'] =  os.path.join('./result', f'fold{fold}', f'sd{exp_seed}',
    #                                                 traincfg['sslset']['type'], traincfg['modelname'], traincfg['expname'])
    
    # traincfg['rootset']['datalist'] = \
        # os.path.join(custom_dir, traincfg['rootset']['datalist'], f"fold_{fold}.json") # modify

    # if traincfg['sslset']['type'] == "mrcpsmix" and \
    #      traincfg['labelWSI'] == traincfg['totalWSI']:
    #     traincfg['sslset']['type'] += "_f"

    return traincfg



#model config load
def load_datacfg(cfgpath):
    """
    Load config from traincfg.yaml 
    """
    custom_dir ="" # modify
    with open(cfgpath, 'r') as fp:
        traincfg = yaml.load(fp, Loader=yaml.FullLoader)
 
    labelWSI = traincfg['expset']['labelWSI']
    unlabelWSI = traincfg['expset']['totalWSI'] - labelWSI

    traincfg['modelname'] = 'MRCPS'


    # expname
    note = traincfg['expname']
    exp_seed = traincfg['expset']['exp_seed']
    epochs = traincfg['expset']['epochs']
    traincfg['expname'] = traincfg['modelname'] +'_'+ traincfg['sslset']['type'] +'_'  \
                          + f"_sd{exp_seed}_e{epochs}"
    
    if 'sda' in traincfg['traindl'] and traincfg['traindl']['sda'] == True:
        traincfg['expname'] += '_s'

    if note != "":
        traincfg['expname'] += f"-{note}"

    # root path setting
    fold = traincfg['expset']['fold']
    if 'fversion' in traincfg['expset'].keys():
        fold = f"{fold}_v{traincfg['expset']['fversion']}"
    # traincfg['rootset']['pklroot_train'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_train'])
    # traincfg['rootset']['pklroot_test'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_test'])
    traincfg['rootset']['savepath'] = os.path.join('./result', f'fold{fold}', f'sd{exp_seed}',
                                                   traincfg['sslset']['type'], traincfg['modelname'], traincfg['expname'])
    
    # traincfg['rootset']['datalist'] = \
        # os.path.join(custom_dir, traincfg['rootset']['datalist'], f"fold_{fold}.json") # modify

    # if traincfg['sslset']['type'] == "mrcpsmix" and \
    #      traincfg['labelWSI'] == traincfg['totalWSI']:
    #     traincfg['sslset']['type'] += "_f"

    return traincfg

