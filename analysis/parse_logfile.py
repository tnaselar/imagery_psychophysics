import numpy as np
##parse the outputs of the imagery psychophysics experiment

#events (dicionary keys)
#paused
#quit
#resumed
#started viewing image i
#stopped viewing image i 
#probe at location (x,y) on
#pressed c - animal
#pressed v - human
#pressed n - natural
#pressed m - manmade
#no press


f = open('/Data/psychophysics/object.imagery.probe/logs/trial_2012-07-22_a/trial_2012-07-22_a.txt', 'r')
data = f.readlines() ##puts it all into a list, 1 item per line

#parse each line, build a dictionary
#paused/quit/resumed -- search for word, store [timestamps...]
#started viewing -- find word "study", store [(timestamp, image)...]
#stopped viewing -- find word "scramble" store [(timestamp, image)...]
#probe on -- find word "probe", store [(timestamp, x,y) ... ]
##pressed animal / human / manmade / natural -- search for 'c'/'v'/'n'/'m', store timestamps 

event_dict = {}
event_types = ['paused', 'resumed', 'quit', 'start_view', 'stop_view', 'probe_on', 'response']
for event in event_types:
  event_dict[event] = []

##TODO: Make sure there is a response accounted for every probe_on, even if response is none.
for events in data:
  state_list = events.strip('\n').split('\t')
  ##determine event type and store
  timestamp = float(state_list[0])
  if state_list[1] == 'paused':
    event_dict['paused'].append(timestamp)
  elif state_list[1] == 'resumed':
    event_dict['resumed'].append(timestamp)
  elif state_list[1] == 'quit':
    event_dict['quit'].append(timestamp)
  elif state_list[2] == 'study':
    event_dict['start_view'].append((timestamp, state_list[1]))
  elif state_list[2] == 'scramble':
    event_dict['stop_view'].append((timestamp, state_list[1]))
  elif state_list[1] == 'probe':
    event_dict['probe_on'].append((timestamp, int(state_list[3]), int(state_list[4])))
  elif state_list[2] == 'c':
    event_dict['response'].append((timestamp, 'animal'))
  elif state_list[2] == 'v':
    event_dict['response'].append((timestamp, 'human'))
  elif state_list[2] == 'n':
    event_dict['response'].append((timestamp, 'natural'))
  elif state_list[2] == 'm':
    event_dict['response'].append((timestamp, 'manmade'))
  elif state_list[2] == 'None':
    event_dict['response'].append((timestamp, 'None'))    
    
##build a simple design matrix
##image, probe, resp
##determine time interval
##determine pauses, quits, and pad the design matrix appropriately = 1+kernel length.

##reaction times
reaction_times = np.array([ii[0] for ii in event_dict['probe_on']]) - np.array([ii[0] for ii in event_dict['response']])









    