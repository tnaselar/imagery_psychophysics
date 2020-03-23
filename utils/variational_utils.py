import numpy as np
import pandas as pd
import copy
from os.path import join
from PIL.Image import open as open_image
from PIL import Image

##data types
floatX = 'float32'
intX = 'int32'

#Big dumb function for importing data from first experiment. Will remove in later iterations.
def open_imagery_probe_data(*args):
    '''
    open_imagery_probe_data() returns a pandas dataframe with lots of info
    
    or
    
    open_imagery_probe_data(subject, state, targetImage) accesses the dataframe and gets the stuff you want
    
    
    
    '''
    ##which repo?
    drive = '/home/tnaselar/FAST'

    ##base directory
    base = 'imagery_psychophysics/multi_poly_probes'

    ##pandas dataframe with all the experimental conditions and data
    data_place = 'data'
    data_file = 'multi_poly_probe_data_5_subjects.pkl'

    ##open experimental data: this is a pandas dataframe
    experiment = pd.read_pickle(join(drive, base, data_place, data_file))
    if not args:
        return experiment
    else:
        subject = args[0]
        state = args[1]
        targetImageName = args[2]
        ##target images
        image_place = 'target_images'
        mask_place = 'masks/processed'

        target_image_file = targetImageName+'_letterbox.png'
        mask_image_file = targetImageName+'_mask.png'

        ##window files
        window_place = 'probes'
        window_file = targetImageName+'_letterbox_img__probe_dict.pkl'

        ##open target image/object map: useful as a guide
        targetImage = open_image(join(drive, base, image_place, target_image_file),mode='r').convert('L')

        ##open
        targetObjectMap = open_image(join(drive, base, mask_place, mask_image_file),mode='r').convert('L')


        ##get the responses you want
        ##responses of select subject / target image / cognitive state
        resp = experiment[(experiment['image']==targetImageName) * (experiment['subj']==subject) * (experiment['state']==state)].response.values

        ##run a nan check--who knew there would be nans in this data?
        nanIdx = ~np.isnan(resp)
        resp = resp[nanIdx]
        if any(~nanIdx):
            print 'killed some nans'
        
        ##give it dimensions
        resp = resp[np.newaxis,:]
        
        ##corresponding window indices
        windowIdx = experiment[(experiment['image']==targetImageName) * (experiment['subj']==subject) * (experiment['state']==state)].probe.values

        ##open the windows. creates a dictionary with "index/mask" keys. this usually takes a while
        windows = pd.read_pickle(join(drive, base, window_place, window_file)) ##N x D1Prime x D2Prime
        N = len(windows['index'])
        window_shape = windows['mask'][0].shape
        W = np.zeros((N,window_shape[0],window_shape[1]),dtype=floatX)


        ##correctly order and reformat
        for ii,w in enumerate(windowIdx):
            str_dx = map(int, w.split('_'))
            dx = windows['index'].index(str_dx)
            W[ii] = windows['mask'][dx].clip(0,1)
            
        ##run another nan check
        W = W[nanIdx]

        return W, resp, experiment, targetObjectMap,targetImage
    


##btw, this is how you get nested attributes
def rgetattribute(obj, listOfAttributes):
    if type(listOfAttributes) is not list:
        listOfAttributes = [listOfAttributes]
    try:
        return getattr(obj, listOfAttributes[-1])
    except:
        return rgetattribute(getattr(obj,  listOfAttributes.pop(0)), listOfAttributes)


##grab the modeling result you want
def get_model_attribute(attributeString,df,shapeOfAttribute=None):
    '''
    get_model_attribute(attributeString,df,shapeOfAttribute=None)
     attributeString can be a list of attributes if the desired attribute is nested.
    '''
    n = len(df)
    if shapeOfAttribute is not None:
        dfshape = [n]+list(shapeOfAttribute)
        newArray = np.full(dfshape,0.)
    else:
        newArray = [0]*n
    for idx,row in df.iterrows():
        copyAttributeString = copy.deepcopy(attributeString)
        newArray[idx] = rgetattribute(df['model'].iloc[idx],copyAttributeString)
    
    return newArray
        
##...if you do something with it that results in nice plottable scalar, add it to a new df
def make_new_df(newColName,newColData,df):
    try:
        newDf = df.drop('model', axis=1)
    except:
        newDf = df
    newDf[newColName] = newColData
    return newDf
