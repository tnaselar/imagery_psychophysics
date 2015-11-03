# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

# <codecell>

subjects = {}

probe_images_path = '/musc.repo/Data/tnaselar/imagery_psychophysics/multi_poly_probes/'

subjects['KL'] = {}
subjects['KL']['sourcefile'] = probe_images_path+'data/KL_2015_May_05_0828.log'

subjects['z1'] = {}
subjects['z1']['sourcefile'] = probe_images_path+'data/z1_2015_May_05_1518.log'


# subjects['CP'] = {}
# subjects['CP']['sourcefile'] = probe_images_path+'data/CP_2014_Nov_14_1513.log'


max_images_per_exp = 4


imagery_marker = '_img_'
perception_marker = '_pcp_'

subject_table = pd.DataFrame(subjects)

number_of_lines_in_chunk = 3

# <codecell>

subject_table

# <codecell>

def starts_a_response_chunk(x):
    if 'New trial' in x:
        return True
    else:
        return False

def get_probe_number(new_line):
    return '_'.join(re.search('_probe\((.*?)\).png', new_line).group(1).split(','))
#     dx = new_line.index('_probe(')
#     if new_line[dx-2].isdigit():
#         return int(new_line[dx-2:dx])
#     else:
#         return int(new_line[dx-1])
    
# def get_repetition(new_line):
#     return int(new_line[new_line.index('rep=')+4])

def get_state(new_line):
    dx = new_line.index('probe(')
    state = new_line[dx-4:dx-1]
    if state:
        return state
    else:
        raise Exception('you are not at the start of a new trial')
#     if 'just-probes' in new_line:
#         return 'img'
#     elif 'probes-with-im' in new_line:
#         return 'pcp'
#     else:
        

def get_time_stamp(new_line):
    return float(new_line[0:new_line.index('\t')])

def get_response(new_line):
    try:
        return int(new_line[new_line.index('Keypress')+10])
    except:
        print('subject did not respond, returning None')
        return None
        
def skip_a_line(all_lines):
    return all_lines.pop(0)

def get_image(new_line):
    return re.search('poly_probes/probes/(.*?_\d\d)', new_line).group(1)
#     dx = new_line.index('finalprobeset')
#     return new_line[dx:(dx+15)]

# <codecell>

f = '/musc.repo/Data/tnaselar/imagery_psychophysics/multi_poly_probes/probes/candle_01_letterbox_img_probe(1,11).png'
snippet = '27.9541	DATA	Keypress: 2'
print get_time_stamp(snippet)
print 'state: %s' %(get_state(f))
print get_response(snippet)
print get_image(f)

# <codecell>

data_dict = {'subj': [], 'image': [], 'probe': [], 'state': [], 'image_on': [], 'resp_on': [], 'response': []}
for subj in subjects.keys():
    print subj    
    all_lines = open(subjects[subj]['sourcefile'], 'r').readlines()
    while all_lines:
        new_line = all_lines.pop(0)
        if starts_a_response_chunk(new_line):
            data_dict['subj'].append(subj)
            new_line = all_lines.pop(0)
            if new_line.find('WARNING') > 0:
                _ = skip_a_line(all_lines)
            new_line = all_lines.pop(0)
            data_dict['probe'].append(get_probe_number(new_line))
            data_dict['state'].append(get_state(new_line))  
            data_dict['image'].append(get_image(new_line))
            data_dict['image_on'].append(get_time_stamp(all_lines.pop(0)))
            new_line = all_lines.pop(0)
            data_dict['resp_on'].append(get_time_stamp(new_line))
            data_dict['response'].append(get_response(new_line))
            


probe_exp = pd.DataFrame(data_dict)


# <codecell>

probe_exp.head()

# <codecell>

probe_exp.dtypes

# <codecell>

probe_exp.describe()

# <codecell>

probe_exp[0:3]

# <codecell>

probe_exp[:3]

# <codecell>

print probe_exp.loc[1]
print type(probe_exp.loc[1])

# <codecell>

probe_exp.loc[:, ['probe', 'response']].head()

# <codecell>

probe_exp.iloc[0]

# <codecell>

probe_exp.iloc[0:2, 1:3]

# <codecell>

probe_exp[probe_exp['probe']=='40_41']

# <codecell>

probe_exp[probe_exp.probe.isin(['40','41','40_41'])]

# <codecell>

KL = probe_exp[probe_exp.subj=='KL']
KL.sort(columns='probe')[0:6]

# <codecell>

probe_exp.save(probe_images_path+'data/z1_KL.pkl')

# <codecell>


