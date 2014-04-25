#!/usr/bin/python
"""Grid presentation as described in 'tn_ecog.proj''"""
import sys

def main():

    from psychopy import event, logging, visual, gui, core, data
    from random import randint
    import numpy
    import os    
    import datetime

    #sets whether or not envisioned tile is actually displayed (0=no, 1=yes)
    control_trial = 0
    
    # create log file for our exp
    LOGS_FOLDER = os.path.abspath('./logs')
    if not os.path.isdir(LOGS_FOLDER):
        os.mkdir(LOGS_FOLDER)
    TRIAL_NAME = 'trial_' + str(datetime.datetime.now())
    TRIAL_FOLDER = os.path.abspath(LOGS_FOLDER + '/' + TRIAL_NAME)
    if not os.path.isdir(TRIAL_FOLDER):
        os.mkdir(TRIAL_FOLDER)
    DATA_FILENAME = TRIAL_NAME + '.txt'
    data_file = open(os.path.join(TRIAL_FOLDER, DATA_FILENAME), 'w')
    
    
# load our audio cues
    ##audio_cues = MyAudioCues()

    #present a dialogue to change grid size
    expInfo = {'grid height':10, 'grid width':10}
    """
    dlg = gui.DlgFromDict(expInfo, title='visual imagery receptive fields exp')
    if dlg.OK:
        grid_size = (expInfo['grid height'],expInfo['grid width'])
    else:
        core.quit()#the user hit cancel so exit
"""

    #log user info
    ##data_file.write('----Subject Information----\n')
    subj_name = 'anonymous'
    subj_DOB = 'unstated'
    subj_group = 'Behavioral'
    ##data_file.write(
    ##                      'Subject Name = %s\n' \
    ##                      'D.O.B. = %s\n' \
    ##                      'Group = %s\n' \
    ##                      % (subj_name, subj_DOB, subj_group)
    ##                      )
    ##data_file.write('\n')

    #log experimental parameters
    ##data_file.write('----Experimental Parameters----\n')
    
    min_number_of_patterns = 5
    max_number_of_patterns = 5
    
    # create our window
    window_height_pixels=500 
    window_width_pixels=window_height_pixels*2
    window = visual.Window(size=[window_width_pixels, window_height_pixels], 
                    monitor ='mbp', color='Black', units='pix', fullscr=False)
    window.setMouseVisible(False)
    
    # create our grid
    grid_size = (10,10)
    grid = numpy.zeros(grid_size)
    object_grid = numpy.empty(grid_size, dtype=numpy.object)
    circle_grid = numpy.empty(grid_size, dtype=numpy.object)
    total_number_of_tiles_in_grid = (grid_size[0]*grid_size[1])
    tile_dim = float(window.size[1]) / float(object_grid.shape[0])
    dist_from_gridline_to_tile_center = tile_dim/2.0
    
    #set up timers
    globalClock = core.Clock()
    envisioning_timer = core.Clock()
    pattern_timer = core.Clock()
    timer = core.Clock()
    
    #set number of trials for experiment = each tile in grid acts as envisioned tile twice
    #1 trial = 1 envisioned_tile with random number of patterns shown
    num_of_trials_in_exp = int(2.0*total_number_of_tiles_in_grid) 
    ##num_of_trials_in_exp = (randint(min_number_of_trials, max_number_of_trials)) #random number of trials
    min_number_of_trials = 1
    max_number_of_trials = num_of_trials_in_exp 
    
    min_number_of_circles_in_a_pattern = 1
    max_number_of_circles_in_a_pattern = int((1.0/4.0)*total_number_of_tiles_in_grid)
    min_circle_stim_size = int((1.0/4.0)*tile_dim)
    max_circle_stim_size = int(tile_dim)
    length_of_time_for_grid_reminder = 4.0
    length_of_time_to_envision = 5.0 #seconds
    length_of_time_to_show_patterns = 3.0 #seconds
    ##data_file.write(
    ##                      'grid_size = %s\n' \
    ##                      'max_number_of_trials = %s\n' \
    ##                      'max_number_of_patterns = %s\n' \
    ##                      'total_number_of_tiles_in_grid = %s\n' \
    ##                      'max_number_of_circles_in_a_pattern = %s\n' \
    ##                      'window_height_pixels = %s\n' \
    ##                      'window_width_pixels = %s\n' \
    ##                      'length_of_time_to_envision = %s\n' \
    ##                      'length_of_time_to_show_patterns = %s\n' \
    ##                      % (
    ##                           grid_size, max_number_of_trials, max_number_of_patterns, total_number_of_tiles_in_grid,
    ##                           max_number_of_circles_in_a_pattern, window_height_pixels, window_width_pixels,
    ##                           length_of_time_to_envision, length_of_time_to_show_patterns
    ##                           )
    ##                      )
    ##data_file.write('\n\nTimestamp\tXe\tYe\tPattern\n')
    
    
    # dictionary of array coordinates paired with cartesian visual grid coordinates
    tile_dict = {}
    # dictionary of cartesian visual grid coordinates with pixel coordinates of tile's center
    vis_grid_pix_dict = {}
    #dictionary of rect object paired with visual grid coordinates
    object_dict = {}
    circle_dict = {}
    
    # create our stimuli and display board
    create_grid_view(object_grid, circle_grid, object_dict, circle_dict, tile_dict, vis_grid_pix_dict, window)

##for i in range(0, object_grid.shape[0]):
        ##for j in range(0, object_grid.shape[1]):
            ##print 'object_grid position: %s is located on visual grid at position: %s with center coords: %s' %((i,j), tile_dict[(i,j)] , vis_grid_pix_dict[tile_dict[(i,j)]])

    #draw outer 4 corner grid framework squares and set them to redraw on each window.flip()
    bottomleft = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, (0,0), dist_from_gridline_to_tile_center, window)
    bottomleft.turn_on_framework()
    bottomleftcornerofgrid_coord = ((bottomleft.pixelcoord[0] - dist_from_gridline_to_tile_center), (bottomleft.pixelcoord[1] - dist_from_gridline_to_tile_center))

    topleft = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, ((grid.shape[0]-1),0), dist_from_gridline_to_tile_center, window)
    topleft.turn_on_framework()
    bottomright = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, (0,(grid.shape[0]-1)), dist_from_gridline_to_tile_center, window)
    bottomright.turn_on_framework()
    topright = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, ((grid.shape[0]-1),(grid.shape[0]-1)), dist_from_gridline_to_tile_center, window)
    topright.turn_on_framework()
    toprightcornerofgrid_coord = ((topright.pixelcoord[0] + dist_from_gridline_to_tile_center), (topright.pixelcoord[1] + dist_from_gridline_to_tile_center))

    #draw center 4 grid framework squares and set them to redraw on each window.flip()
    topleftcenter = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, (((object_grid.shape[1]/2)-1), ((object_grid.shape[0]/2))), dist_from_gridline_to_tile_center, window)
    topleftcenter.turn_on_framework()
    toprightcenter = Tile(object_grid, object_dict, tile_dict,  vis_grid_pix_dict, ((object_grid.shape[1]/2), (object_grid.shape[0]/2)), dist_from_gridline_to_tile_center, window)
    toprightcenter.turn_on_framework()
    bottomleftcenter = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, (((object_grid.shape[1]/2)-1), ((object_grid.shape[0]/2)-1)), dist_from_gridline_to_tile_center, window)
    bottomleftcenter.turn_on_framework()
    bottomrightcenter = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, ((object_grid.shape[1]/2), ((object_grid.shape[0]/2)-1)), dist_from_gridline_to_tile_center, window)
    bottomrightcenter.turn_on_framework()

    #create fixation point
    fixationpt = visual.DotStim(window, units = 'pix', color = [1.0,-1,-1], colorSpace='rgb', dotSize=25, fieldPos = [0,0], depth=-1) 
    fixationpt.draw()  
    fixationpt.setAutoDraw(True)

    # activate center tile and show instructions
    examplesquare = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, choose_tile(grid), dist_from_gridline_to_tile_center, window)
    active_tile = examplesquare
    active_tile.turn_on_color()
    print_directions('Consider the fixation point\nin the center of the grid\nto be position (0,0).\n\nUse '\
                              'given directions\nto visualize each grid\nposition in your mind\'s\neye.  As shown '\
                              'in the\nexample to the right\nfor position %s.\n\nDuring each trial,\nmultiple '\
                              'squares will\nilluminate simultaneously.\nIf your imagined square\nis one of them:\n'\
                              'press spacebar.\n\n\nPress spacebar to\nturn off grid and continue.\n' %str(active_tile.cartesiancoord), window)

    window.flip()
    #wait for user input
    begin_key = event.waitKeys(keyList='space')

    print_directions('To begin experiment,\npress spacebar.', window)
    active_tile.turn_off()
    window.flip()
     #wait for user input
    begin_key = event.waitKeys(keyList='space')
    window.flip()

    #builds ordered list of tiles for every square in object_grid array
    all_tiles = []
    for i in range(0, object_grid.shape[0]):
        for j in range(0, object_grid.shape[1]):
            new_tile = Tile(object_grid, object_dict, tile_dict, vis_grid_pix_dict, (i,j) , dist_from_gridline_to_tile_center, window)
            all_tiles.append(new_tile)

    #build dictionary to track number of times envisioned tiles (keys) are included in list_of_active_tiles_for_whole_exp
    all_possible_envisioned_tiles = all_tiles
    envisioned_tile_dict = {}
    for tile in all_possible_envisioned_tiles:
        envisioned_tile_dict[tile] = 0

    #creates a list of tiles to be envisioned during experiment
    list_of_active_tiles_for_whole_exp = []
    for trial in xrange(num_of_trials_in_exp):
        envisioned_tile = all_possible_envisioned_tiles[randint(0,(len(all_possible_envisioned_tiles)-1))]
        while envisioned_tile == active_tile:
            envisioned_tile = all_possible_envisioned_tiles[randint(0,(len(all_possible_envisioned_tiles)-1))]
        while envisioned_tile_dict[envisioned_tile] == 2:
            envisioned_tile = all_possible_envisioned_tiles[randint(0,(len(all_possible_envisioned_tiles)-1))]
        if envisioned_tile_dict[envisioned_tile] == 0:
            envisioned_tile_dict[envisioned_tile] = 1
        elif envisioned_tile_dict[envisioned_tile] == 1:
            envisioned_tile_dict[envisioned_tile] = 2
        list_of_active_tiles_for_whole_exp.append(envisioned_tile)



    #run the experiment
    nTrial = 0
    for EnvisionedTile in list_of_active_tiles_for_whole_exp: 
        get_key_strokes(event, core, data_file, globalClock, active_tile)
        nTrial += 1

        active_tile = list_of_active_tiles_for_whole_exp[nTrial-1]
        ##data_file.write('\nTrial: %s\n' %nTrial)
        
        #creates list of patterns with random numbers of circle stimuli 
        #to be shown for each envisioned tile
        list_of_patterns_to_be_shown_for_envisioned_tile=[]
        random_num_of_patterns = randint(min_number_of_patterns, max_number_of_patterns)
        for pattern in xrange(random_num_of_patterns):
            list_of_circles = []
            random_num_of_circles_in_pattern = randint(min_number_of_circles_in_a_pattern, max_number_of_circles_in_a_pattern)
            for circle_in_pattern in xrange(random_num_of_circles_in_pattern):
                circle_dim = (randint(min_circle_stim_size, max_circle_stim_size))
                circle_x_coord = randint(bottomleftcornerofgrid_coord[0], toprightcornerofgrid_coord[0])
                circle_y_coord = randint(bottomleftcornerofgrid_coord[1], toprightcornerofgrid_coord[1])
                circle_pixelcoord = (circle_x_coord, circle_y_coord)
                circle_obj = visual.Circle(win=window, radius = circle_dim, pos=circle_pixelcoord, fillColor=[1,1,1]) 
                
                #makes dictionary with keys as pixel coordinates position and values as circle_grid circle object
                circle_dict[circle_pixelcoord] = circle_obj
                list_of_circles.append(circle_obj)
            list_of_patterns_to_be_shown_for_envisioned_tile.append(list_of_circles)


        # ADD HERE: audio_cues.play_audio_for_directions(last.directions_to(new))
        
        
        #display grid to refresh subject's memory
        active_tile.turn_off()
        window.clearBuffer()
        window.flip()
        print_directions('Remember the grid', window)
        for i in range(0, object_grid.shape[0]):
            for j in range(0, object_grid.shape[1]):
                object_grid[i][j].setFillColor('Black')
                object_grid[i][j].setLineColor('White')
                object_grid[i][j].draw()
        window.flip()
        data_file.write('%s\t%s\t%s\t%s\n' \
                %(globalClock.getTime(), 'grid', 'grid', 'grid'))
        envisioning_timer.reset()
        a = envisioning_timer.getTime()
        while a < length_of_time_for_grid_reminder:
            a = envisioning_timer.getTime()
        
        #print timestamp when envision task given and logs keystrokes during envisioning period
        print_directions(('Envision position:\n%s' %str(active_tile.cartesiancoord)), window)
        if control_trial == 1:
                active_tile.turn_on_color()
        window.flip()
        data_file.write('%s\t%s\t%s\t%s\n' \
                %(globalClock.getTime(), active_tile.cartesiancoord[0], active_tile.cartesiancoord[1], 'none'))
        envisioning_timer.reset()
        a = envisioning_timer.getTime()
        while a < length_of_time_to_envision:
            a = envisioning_timer.getTime()
            
        #shows patterns and logs keystrokes
        nPattern = 0
        for thisPattern in list_of_patterns_to_be_shown_for_envisioned_tile:
            get_key_strokes(event, core, data_file, globalClock, active_tile)
            active_tile.turn_off()
            nPattern += 1
            pattern = list_of_patterns_to_be_shown_for_envisioned_tile[nPattern-1]
            if control_trial == 0:
                pattern_name = ('trial' + str(nTrial) + '_' + 'pattern' + str(nPattern))
            elif control_trial == 1:
                pattern_name = ('control_trial' + str(nTrial) + '_' + 'pattern' + str(nPattern))
                active_tile.turn_on_color()
            for circle in thisPattern:
                circle.draw()
            data_file.write('%s\t%s\t%s\t%s\n' \
                %(globalClock.getTime(), active_tile.cartesiancoord[0], active_tile.cartesiancoord[1], pattern_name))
            
            #screen capture of pattern
            window.getMovieFrame(buffer='back')
            window.saveMovieFrames('%s/%s.jpg' %(TRIAL_FOLDER, pattern_name))
            print_directions(('Each time envisioned\nsquare %s\nis illuminated\npress spacebar' %str(active_tile.cartesiancoord)), window)
            window.flip()
            
            pattern_timer.reset()
            b = pattern_timer.getTime() 
            while b < length_of_time_to_show_patterns:
                b = pattern_timer.getTime() 
                
            active_tile.turn_on_color()
            for circle in thisPattern:
                circle.draw()
            
            #screen capture of pattern with active tile turned on
            window.getMovieFrame(buffer='back')
            window.saveMovieFrames('%s/%splustile.jpg' %(TRIAL_FOLDER, pattern_name))
            
            window.clearBuffer()
            active_tile.turn_off()

    #end of exp
    
    window.close()
    data_file.close()
    sys.exit()

    #list of important information/events
    ##trials.data.addDataType('Trial Number')
    ##trials.data.addDataType('Envisioned Tile')
    ##trials.data.addDataType('Number of Patterns Shown')
    ##patterns.data.addDataType('Pattern Number')
    ##patterns.data.addDataType('Timestamp when Pattern Shown')
    ##patterns.data.addDataType('Number of Tiles in Pattern')
    ##patterns.data.addDataType('List of Tiles in Pattern')
    ##trials.data.addDataType('Is Envisioned Tile present in Pattern?')
    ##trials.data.addDataType('Was Key pressed?')
    ##trials.data.addDataType('Timestamp when Key was pressed')


############################
def get_key_strokes(event, core, data_file, globalClock, active_tile):
        keys = event.getKeys(timeStamped = globalClock)
        for thisKey in keys:

           if thisKey[0] in ['q', 'escape']:
               data_file.write('%s\t%s\t%s\t%s\t%s\n' \
                    %(globalClock.getTime(), 'quit', 'quit', 'quit', 'quit'))
               core.quit() #abort experiment
           elif len(thisKey[0])>0:
                print thisKey
                data_file.write('%s\t%s\t%s\t%s\t%s\n' \
                    %(thisKey[1], active_tile.cartesiancoord[0], active_tile.cartesiancoord[1], thisKey[0], thisKey[0]))


def find_key(dict, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dict.iteritems() if v == val][0]

def choose_tile(grid):
    from random import randint
    dims = grid.shape
    return (randint(0,dims[0]-1), randint(0,dims[1]-1) )

def create_grid_view(object_grid, circle_grid, object_dict, circle_dict, tile_dict, vis_grid_pix_dict, window):
    """Create a square stim for each entry in the grid and return in an array"""
    from psychopy import visual, core
    from random import randint
    
    # assert that grid is expected shape
    assert len(object_grid.shape) == 2
    # get the dimension that each tile should be as per the containing
    # window (which has normalized units, we will assume that width >=
    # height, which is the case with most monitors, and accordingly
    # use height as the limiting dimension
    grid_height_pixels = window.size[1]
    grid_width_pixels = window.size[0]/2
    tile_dim = float(window.size[1]) / float(object_grid.shape[0])
    dist_from_gridline_to_tile_center = tile_dim/2.0
    
    # create a tile for each location, draw it, and make a dictionary that 
    #contains tiles object_grid coordinates and cartesian coordinates with fixationpt = (0,0))
    for i in range(0, object_grid.shape[0]):
        for j in range(0, object_grid.shape[1]):
            y_gridline_position = i * tile_dim
            x_gridline_position = j * tile_dim
            tile_center = (y_gridline_position + dist_from_gridline_to_tile_center - window.size[1] / 2.0,
                           x_gridline_position + dist_from_gridline_to_tile_center - window.size[1] / 2.0)

            rect_obj = visual.Rect(window, width=tile_dim, height=tile_dim, pos=tile_center,
                                     lineColor='White', fillColor='Black', interpolate=True,  closeShape=True)
            object_grid[i][j] = rect_obj
            object_grid[i][j].draw()

            if tile_center[0] < 0:
                x_box_coord = int(((tile_center[0] - dist_from_gridline_to_tile_center))/((grid_width_pixels/2)/(object_grid.shape[0]/2)))
            else:
                x_box_coord = int(((tile_center[0] + dist_from_gridline_to_tile_center))/((grid_width_pixels/2)/(object_grid.shape[0]/2)))
            
            if tile_center[1] < 0:
                y_box_coord = int(((tile_center[1] - dist_from_gridline_to_tile_center))/((grid_width_pixels/2)/(object_grid.shape[1]/2)))
            else:
                y_box_coord = int(((tile_center[1] + dist_from_gridline_to_tile_center))/((grid_width_pixels/2)/(object_grid.shape[1]/2)))
            
            box_coord = (x_box_coord, y_box_coord)
            
            #makes dictionary with keys as object_grid position and values as visual grid position
            tile_dict[(i,j)] = box_coord
            ##print "position %s in object_grid is actually position %s on visual grid" %((i,j), tile_dict[(i,j)])
            
            #makes dictionary with keys as visual grid position and values as object_grid rect object and 
            object_dict[box_coord] = rect_obj
            
            #makes dictionary with keys as visual grid position and values as tuples of center-pixel coordinates
            pix_tuple = (object_grid[i][j].pos[0], object_grid[i][j].pos[1])
            vis_grid_pix_dict[box_coord] = pix_tuple
            ##print "center of tile %s in visual grid is drawn at pixel coordinates %s " %(box_coord, vis_grid_pix_dict[box_coord])



def draw_grid(object_grid):
    """Draw every tile in the grid"""
    for tile in object_grid.ravel(): # flattens from 2-D to 1-D arrays
        tile.draw()

def print_directions(dirStr, window):
    """Print the directions in the upper left hand corner of the given
    window"""
    from psychopy import visual
    my_pos = ( -(window.size[0] / 2.0), (window.size[1] / 2.0) )
    visual.TextStim(window, text=dirStr, pos=my_pos, alignHoriz='left', alignVert='top').draw()




class Tile(object):
    """A tile that belongs to grid."""
    def __init__(self, object_grid, object_dict, tile_dict, vis_grid_pix_dict, pos, dist_from_gridline_to_tile_center, window):
        self.x = pos[0]
        self.y = pos[1]
        self.grid = object_grid
        self.arraycoord = (self.x, self.y)
        self.cartesiancoord = tile_dict[self.arraycoord]
        self.pixelcoord = vis_grid_pix_dict[self.cartesiancoord]
        self.object = object_dict[self.cartesiancoord]
        self.coord_text = self.print_position_on_grid(dist_from_gridline_to_tile_center, window)

    def __eq__(self, other):
        return self.cartesiancoord == other.cartesiancoord

    def relative_directions_to(self, new):
        """Return directions as xy coordinate tuple relative to distance from previous active square"""
        return (new.x-self.x, new.y-self.y)

    def absolute_directions_to(self):
        """Return directions as xy coordinate tuple relative to absolute grid coordinate"""
        return (self.x, self.y)

    def print_position_on_grid(self, dist_from_gridline_to_tile_center, window):
        """Prints xy coordinate tuple in correction position on grid"""
        from psychopy import visual, core
        text_x = self.pixelcoord[0] - (0.88*dist_from_gridline_to_tile_center)
        text_y = self.pixelcoord[1] + (0.25*dist_from_gridline_to_tile_center)
        textcoord = (text_x, text_y)
        dirStr = str(self.cartesiancoord[0]) + "," + str(self.cartesiancoord[1])
        return visual.TextStim(window, text=dirStr, pos=textcoord, alignHoriz='left', alignVert='top')



    def directions_to_as_dict(self, new):
        """Return directions as a dictionary."""
        directions = {}
        # get horizontal
        h_vec = new.x - self.x
        if h_vec > 0:
            directions['horiz_direction'] = 'RIGHT'
            directions['horiz_magnitude'] = h_vec
        elif h_vec < 0:
            directions['horiz_direction'] = 'LEFT'
            directions['horiz_magnitude'] = -h_vec
        else:
            directions['horiz_direction'] = None
        # get vertical
        v_vec = new.y - self.y
        if v_vec > 0:
            directions['vert_direction'] = 'UP'
            directions['vert_magnitude'] = v_vec
        elif v_vec < 0:
            directions['vert_direction'] = 'DOWN'
            directions['vert_magnitude'] = -v_vec
        else:
            directions['vert_direction'] = None
        # give directions
        return directions

    def directions_to_as_str(self, new):
        horiz_str = ''
        vert_str = ''
        d = self.directions_to_as_dict(new)
        if d['horiz_direction'] is not None:
            horiz_str = "%s: %s" % (d['horiz_direction'], d['horiz_magnitude'])
        if d['vert_direction'] is not None:
            vert_str = "%s: %s" % (d['vert_direction'], d['vert_magnitude'])
        return horiz_str + '\n' + vert_str

    def turn_on_framework(self):
        """turns on background grid and sets to draw on each window flip"""
        self.grid[self.x, self.y].setFillColor('Black')
        self.grid[self.x, self.y].draw()
        ##self.grid[self.x, self.y].setAutoDraw(True)
        self.coord_text.draw()
        ##self.coord_text.setAutoDraw(True)
        
    def turn_on_coord(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setFillColor('Black')
        self.grid[self.x, self.y].draw()
        self.coord_text.draw()

    def turn_on_color(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setFillColor('Gray')
        self.grid[self.x, self.y].setLineColor('Gray')
        self.grid[self.x, self.y].draw()

    def turn_on_circle_color(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setColor([1,1,1])
        self.grid[self.x, self.y].draw()

    def turn_off(self):
        self.grid[self.x, self.y].setFillColor('Black')
        self.grid[self.x, self.y].setLineColor('Black')
        self.grid[self.x, self.y].draw()

class Circle(object):
    """A circle that is part of a random stimulus pattern"""
    def __init__(self, circle_grid, circle_dict, tile_dict, vis_grid_pix_dict, pos, dist_from_gridline_to_tile_center, window):
        self.x = pos[0]
        self.y = pos[1]
        self.grid = circle_grid
        self.pixelcoord = (self.x, self.y)
        print self.pixelcoord
        sys.exit()
        self.object = circle_dict[self.pixelcoord]
        self.coord_text = self.print_position_on_grid(dist_from_gridline_to_tile_center, window)

    def __eq__(self, other):
        return self.cartesiancoord == other.cartesiancoord

    def relative_directions_to(self, new):
        """Return directions as xy coordinate tuple relative to distance from previous active square"""
        return (new.x-self.x, new.y-self.y)

    def absolute_directions_to(self):
        """Return directions as xy coordinate tuple relative to absolute grid coordinate"""
        return (self.x, self.y)

    def print_position_on_grid(self, dist_from_gridline_to_tile_center, window):
        """Prints xy coordinate tuple in correction position on grid"""
        from psychopy import visual, core
        text_x = self.pixelcoord[0] - (0.88*dist_from_gridline_to_tile_center)
        text_y = self.pixelcoord[1] + (0.25*dist_from_gridline_to_tile_center)
        textcoord = (text_x, text_y)
        dirStr = str(self.cartesiancoord[0]) + "," + str(self.cartesiancoord[1])
        return visual.TextStim(window, text=dirStr, pos=textcoord, alignHoriz='left', alignVert='top')



    def directions_to_as_dict(self, new):
        """Return directions as a dictionary."""
        directions = {}
        # get horizontal
        h_vec = new.x - self.x
        if h_vec > 0:
            directions['horiz_direction'] = 'RIGHT'
            directions['horiz_magnitude'] = h_vec
        elif h_vec < 0:
            directions['horiz_direction'] = 'LEFT'
            directions['horiz_magnitude'] = -h_vec
        else:
            directions['horiz_direction'] = None
        # get vertical
        v_vec = new.y - self.y
        if v_vec > 0:
            directions['vert_direction'] = 'UP'
            directions['vert_magnitude'] = v_vec
        elif v_vec < 0:
            directions['vert_direction'] = 'DOWN'
            directions['vert_magnitude'] = -v_vec
        else:
            directions['vert_direction'] = None
        # give directions
        return directions

    def directions_to_as_str(self, new):
        horiz_str = ''
        vert_str = ''
        d = self.directions_to_as_dict(new)
        if d['horiz_direction'] is not None:
            horiz_str = "%s: %s" % (d['horiz_direction'], d['horiz_magnitude'])
        if d['vert_direction'] is not None:
            vert_str = "%s: %s" % (d['vert_direction'], d['vert_magnitude'])
        return horiz_str + '\n' + vert_str

    def turn_on_framework(self):
        """turns on background grid and sets to draw on each window flip"""
        self.grid[self.x, self.y].setFillColor('Black')
        self.grid[self.x, self.y].draw()
        ##self.grid[self.x, self.y].setAutoDraw(True)
        self.coord_text.draw()
        ##self.coord_text.setAutoDraw(True)
        
    def turn_on_coord(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setFillColor('Black')
        self.grid[self.x, self.y].draw()
        self.coord_text.draw()

    def turn_on_color(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setFillColor('White')
        self.grid[self.x, self.y].draw()

    def turn_on_circle_color(self):
        """turns on tile and draws its coordinates as text"""
        self.grid[self.x, self.y].setColor([1,1,1])
        self.grid[self.x, self.y].draw()

    def turn_off(self):
        self.grid[self.x, self.y].setFillColor('Black')


"""

class MyAudioCues(object)    :
    #Directional audio cues

    def __init__(self):
        # load cues as dictionary
        self.cues = self.load_cues_from_dir(self.audio_dir())

    def audio_dir(self):
        import os.path as op
        this_dir = op.abspath(op.dirname(__file__))
        return op.join(this_dir,'../../test_data/audio')

    def load_cues_from_dir(self, directory):
        import os
        import psychopy.sound as ps
        results = dict()
        audio_files = os.listdir(directory)
        for af in audio_files:
            filename = af
            filekey = os.path.splitext(filename)[0]
            filepath = os.path.join(directory, filename)
            results[filekey] = ps.SoundPygame(value=filepath, sampleRate=16000, autoLog=False)
        return results

    def play_audio_for_directions(self, directions):
        #Play audio sequence for a given set of directions
        from psychopy import core
        # if there is a change in x...
        x = directions[0]
        y = directions[1]
        if x != 0:
            # play x-direction
            if x > 0:
                self.cues['right'].play()
            else:
                self.cues['left'].play()
            # wait .5 sec
            core.wait(.5)
            # play x-magnitude
            self.cues[str(abs(x))].play()
            # wait .75 sec
            core.wait(.75)
        # if there is a change in y...
        if y != 0:
            # play y-direction
            if y > 0:
                self.cues['up'].play()
            else:
                self.cues['down'].play()
            # wait .5 sec
            core.wait(.5)
            # play y-magnitude
            self.cues[str(abs(y))].play()
            core.wait(.75)
            
            """
            
        
            
        
        
# Entry point #############
if __name__ == '__main__':
    main()
