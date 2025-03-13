import os
import yaml

def load_traincfg(args):
    """
    Load config from traincfg.yaml 
    """
    custom_dir ="" # modify
    with open(args.cfgpath, 'r') as fp:
        traincfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    if args.project is not None:
        traincfg['project'] = args.project
    if args.expname is not None:
        traincfg['expname'] = args.expname
    if args.kfold is not None:
        traincfg['expset']['fold'] = args.kfold
    if args.fversion is not None:
        traincfg['expset']['fversion'] = args.fversion
    # if args.dataroot is not None:
    #     traincfg['rootset']['dataroot'] = os.path.join('/work/u7085556', args.dataroot)
    if args.epochs is not None:
        traincfg['expset']['epochs'] = args.epochs
    if args.lr is not None:
        traincfg['optim']['lr'] = args.lr
    if args.lr_end is not None:
        traincfg['sched']['lr_end'] = args.lr_end
    if args.labelWSI is not None:
        traincfg['expset']['labelWSI'] = args.labelWSI 
    if args.totalWSI is not None:
        traincfg['expset']['totalWSI'] = args.totalWSI
    if args.exp_seed is not None:
        traincfg['expset']['exp_seed'] = args.exp_seed
        traincfg['expset']['env_seed'] = args.exp_seed
        traincfg['expset']['model_seed'] = args.exp_seed
    if args.sda is not None:
        traincfg['traindl']['sda'] = args.sda
    if args.batchsize is not None:
        traincfg['traindl']['batchsize'] = args.batchsize
    if args.tifpage is not None:
        traincfg['traindl']['tifpage'] = args.tifpage
        traincfg['testdl']['tifpage'] = args.tifpage
 
    labelWSI = traincfg['expset']['labelWSI']
    unlabelWSI = traincfg['expset']['totalWSI'] - labelWSI

    if "SemiSegPathology" in traincfg['rootset']['dataroot']:
        if labelWSI == 1:
            traincfg['expset']['KVGHlabel'] = 1
            traincfg['expset']['KVGHunlabel'] = traincfg['expset']['totalWSI'] // 5 * 3 - 1
            traincfg['expset']['NCKUlabel'] = 0
            traincfg['expset']['NCKUunlabel'] = traincfg['expset']['totalWSI'] // 5 * 2
        elif labelWSI == 3:
            traincfg['expset']['KVGHlabel'] = 2
            traincfg['expset']['KVGHunlabel'] = traincfg['expset']['totalWSI'] // 5 * 3 - 2
            traincfg['expset']['NCKUlabel'] = 1
            traincfg['expset']['NCKUunlabel'] = traincfg['expset']['totalWSI'] // 5 * 2 - 1
        else:
            traincfg['expset']['KVGHlabel'] = labelWSI // 5 * 3
            traincfg['expset']['KVGHunlabel'] = unlabelWSI // 5 * 3
            traincfg['expset']['NCKUlabel'] = labelWSI // 5 * 2
            traincfg['expset']['NCKUunlabel'] = unlabelWSI // 5 * 2
        
    # Check sslset flag
    if 'branch2' in traincfg:
        if traincfg['sslset']['type'] == 'sup' or traincfg['sslset']['type'] == 'mix':
            raise ValueError("Supervised or Mix semi mode should only have one branch."
                    "Detect branch2.")

    if 'cps' in traincfg['sslset']['type'] \
        and 'branch2' not in traincfg:
        traincfg['branch2'] = traincfg['branch1']

    # modelname setting
    if 'branch2' not in traincfg or traincfg['branch1'] == traincfg['branch2']:
        modelname = traincfg['branch1'].split('/')[-1].split('.')[0]
    else:
        modelname = traincfg['branch1'].split('/')[-1].split('.')[0]
        modelname += traincfg['branch2'].split('/')[-1].split('.')[0]
    traincfg['modelname'] = modelname

    if 'branch2' in traincfg:
        with open(custom_dir+traincfg['branch2'], 'r') as fp: #modify
            modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
        modelcfg['model.classes'] = len(traincfg['classes'])
        modelcfg['model_seed'] = 2 * traincfg['expset']['model_seed']
        if 'model.lrscale' in modelcfg and args.lrratio is not None:
            modelcfg['model.lrscale'] = args.lrratio
        traincfg['branch2'] = modelcfg
        traincfg['modelnum'] = 2
    else:
        traincfg['modelnum'] = 1

    if 'branch3' in traincfg:
        with open(custom_dir+traincfg['branch3'], 'r') as fp: #modify
            modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
        modelcfg['model.classes'] = len(traincfg['classes'])
        modelcfg['model_seed'] = traincfg['expset']['model_seed']
        if 'model.lrscale' in modelcfg and args.lrratio is not None:
            modelcfg['model.lrscale'] = args.lrratio
        traincfg['branch3'] = modelcfg
        traincfg['modelnum'] = 3
    
    with open(custom_dir+traincfg['branch1'], 'r') as fp: #modify
        modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
    modelcfg['model.classes'] = len(traincfg['classes'])
    modelcfg['model_seed'] = traincfg['expset']['model_seed']
    if 'model.lrscale' in modelcfg and args.lrratio is not None:
        modelcfg['model.lrscale'] = args.lrratio
        traincfg['expset']['lrratio'] = args.lrratio
    traincfg['branch1'] = modelcfg

    # expname
    note = traincfg['expname']
    exp_seed = traincfg['expset']['exp_seed']
    epochs = traincfg['expset']['epochs']
    traincfg['expname'] = traincfg['sslset']['type'] + '_' + modelname \
                          + f"_l{labelWSI}_u{unlabelWSI}_sd{exp_seed}_e{epochs}"

    if 'sda' in traincfg['traindl'] and traincfg['traindl']['sda'] == True:
        traincfg['expname'] += '_s'

    if note != "":
        traincfg['expname'] += f"-{note}"

    # root path setting
    fold = traincfg['expset']['fold']
    if args.fversion is not None:
        fold = f"{fold}_v{args.fversion}"
    # traincfg['rootset']['pklroot_train'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_train'])
    # traincfg['rootset']['pklroot_test'] = \
    #     os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_test'])
    traincfg['rootset']['savepath'] = \
        os.path.join('./result', f'fold{fold}', f'sd{exp_seed}', traincfg['sslset']['type'], modelname, traincfg['expname'])
    
    # traincfg['rootset']['datalist'] = \
        # os.path.join(custom_dir, traincfg['rootset']['datalist'], f"fold_{fold}.json") # modify

    if traincfg['sslset']['type'] == "mrcpsmix" and \
        args.labelWSI == args.totalWSI:
        traincfg['sslset']['type'] += "_f"

    return traincfg

def searchnewname(root, expname_base):
    expname = expname_base
    num = 0
    while(os.path.isdir(os.path.join(root, expname))):
        num += 1
        expname = f'{expname_base}-{num}'
    return expname


def flatten_json(json):
    if type(json) == dict:
        for k, v in list(json.items()):
            if type(v) == dict:
                flatten_json(v)
                json.pop(k)
                for k2, v2 in v.items():
                    json[k+"."+k2] = v2


def unflatten_json(json):
    if type(json) == dict:
        for k in sorted(json.keys(), reverse=True):
            if "." in k:
                key_parts = k.split(".")
                json1 = json
                for i in range(0, len(key_parts)-1):
                    k1 = key_parts[i]
                    if k1 in json1:
                        json1 = json1[k1]
                        if type(json1) != dict:
                            conflicting_key = ".".join(key_parts[0:i+1])
                            raise Exception('Key "{}" conflicts with key "{}"'.format(
                                k, conflicting_key))
                    else:
                        json2 = dict()
                        json1[k1] = json2
                        json1 = json2
                if type(json1) == dict:
                    v = json.pop(k)
                    json1[key_parts[-1]] = v

# def load_traincfg_old(cfgpath = './config/traincfg.yaml'):
#     """
#     Load config from traincfg.yaml 
#     """
#     with open(cfgpath, 'r') as fp:
#         traincfg = yaml.load(fp, Loader=yaml.FullLoader)
    
#     labelWSI = traincfg['expset']['labelWSI']
#     unlabelWSI = traincfg['expset']['totalWSI'] - labelWSI
#     traincfg['expset']['KVGHlabel'] = labelWSI // 5 * 3
#     traincfg['expset']['KVGHunlabel'] = unlabelWSI // 5 * 3
#     traincfg['expset']['NCKUlabel'] = labelWSI // 5 * 2
#     traincfg['expset']['NCKUunlabel'] = unlabelWSI // 5 * 2

#     # Check sslset flag
#     if 'branch2' in traincfg:
#         if traincfg['sslset']['type'] == 'sup' or traincfg['sslset']['type'] == 'mix':
#             raise ValueError("Supervised or Mix semi mode should only have one branch."
#                     "Detect branch2.")

#     if 'cps' in traincfg['sslset']['type'] \
#         and 'branch2' not in traincfg:
#         traincfg['branch2'] = traincfg['branch1']

#     # expname setting
#     if 'branch2' not in traincfg or traincfg['branch1'] == traincfg['branch2']:
#         modelname = traincfg['branch1'].split('/')[-1].split('.')[0]
#     else:
#         modelname = "Mix"

#     # branch modelcfg setting
#     for idx in range(1, 5):
#         key = f"branch{idx}"
#         if key in traincfg:
#             with open(traincfg[key], 'r') as fp:
#                 modelcfg = yaml.load(fp, Loader=yaml.FullLoader)
#             modelcfg['model.classes'] = len(traincfg['classes'])
#             modelcfg['model_seed'] = idx * 24
#             traincfg[key] = modelcfg
#         else:
#             traincfg['modelnum'] = idx - 1
#             break

#     note = traincfg['expname']
#     fold = traincfg['expset']['fold']
#     traincfg['expname'] = traincfg['sslset']['type'] + '_' + modelname \
#                           + f"_l{labelWSI}_u{unlabelWSI}_f{fold}"
#     if note != "":
#         traincfg['expname'] += f"-{note}"

#     # root path setting
#     traincfg['rootset']['pklroot_train'] = \
#         os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_train'])
#     traincfg['rootset']['pklroot_test'] = \
#         os.path.join(traincfg['rootset']['dataroot'], traincfg['rootset']['pklroot_test'])
#     traincfg['rootset']['savepath'] = \
#         os.path.join('./result', f'fold{fold}', traincfg['expname'])
#     traincfg['rootset']['datalist'] = \
#         os.path.join(traincfg['rootset']['datalist'], f"fold_{fold}.json")

#     return traincfg