#!/usr/bin/python

## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.

import re
import sys
import math
import os
import numpy as np

from matplotlib import pyplot, patches
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

legendfontsize = 18
ticksfontsize  = 24 
axisfontsize   = ticksfontsize + 3
graphwidth     = 2
trajwidth      = 1

rc('text', usetex=True)

KNOWN_TAGS = ('parameter', 'com_position', 'com_speed','com_theta','com_angular_speed',
              'support_foot_theta','flying_foot_theta','desired_position',
              'zmp_position', 'foot_position', 'qp_time', 'qp_time_orientation_lineal',
              'predicted_foot_position', 'predicted_com_position', 'predicted_zmp_position', 'jerks', 'angle_jerks', 'visual_feature',
              'current_image_point', 'reference_image_point', 'virtual_reference_image_point',
              'reference_image_from_initial_position','com_acceleration','orientation0', 'objective_function','orientation_objective_function')

def unpack_data(*packed_data):
    unpacked_data = []
    for entry in packed_data:
        try:
            entry = float(entry)
        except ValueError:
            pass
        unpacked_data.append(entry)
    return unpacked_data


def parse_raw_log_entry(entry):
    """expected line format: entry0=value0, entry1=value1, ..."""
    all_elements = []
    for element in entry.split(', '):
        __, value = element.split('=')
        all_elements.append(value.strip())
    return unpack_data(*all_elements)


def load_data(file_name):
    """expected line format: ... [tag]: entry0=value0, entry1=value1, ..."""
    data = {key:[] for key in KNOWN_TAGS}
    search_key = re.compile(r'\[\w+\]\[(\w+)\]').search

    with open(file_name, 'r') as file_handle:
        for line in file_handle:
            key = search_key(line)
            if key:
                key = key.group(1).strip()
                if key not in KNOWN_TAGS:
                    continue
                __, value = line.strip().split(']: ')
                data[key].append(value.strip())
    return data

def draw_desired_position(center,angle, color="red", linewidth=graphwidth):
    """ """
    # nao
    # width = 2*(0.058+0.01)
    # length = 0.1372

    # HRP-2
    width = 2*(0.14+0.025)
    length = 0.25

    x, y = center
    x = x - (length/2)*math.cos(angle) + (width/2)*math.sin(angle)
    y = y - (length/2)*math.sin(angle) - (width/2)*math.cos(angle)
    angle = math.degrees(angle)
    return patches.Rectangle((x, y), length, width, angle, fill=False, edgecolor=color, linewidth=linewidth)

def convert_point_to_foot(center, angle, color="black", linewidth=graphwidth):
    """ """
    # nao
    # width = 0.058
    # length = 0.1372

    # HRP-2
    width = 0.14
    length = 0.25

    x, y = center
    x = x - (length/2)*math.cos(angle) + (width/2)*math.sin(angle)
    y = y - (length/2)*math.sin(angle) - (width/2)*math.cos(angle)
    angle = math.degrees(angle)
    return patches.Rectangle((x, y), length, width, angle, fill=False, edgecolor=color, linewidth=linewidth)

def prepare_predicted_data_for_plot(raw_data):
    """expected input list of strings of the form 'iteration=90, x=1.07848, y=-0.0640668'"""

    def unpack_data(__, i, x, y, z="dummy=0.0"):
        return (i, x, y, z)

    horizon_clean_data= []      
    x_axis_clean_data = []
    y_axis_clean_data = []
    phi_clean_data    = []

    for entry in raw_data:
        ihoriz, x_pos, y_pos, phi = unpack_data(*entry.split(', '))
        __, ihoriz = ihoriz.split('=')
        __, x_pos = x_pos.split('=')
        __, y_pos = y_pos.split('=')
        __, phi = phi.split('=')
        horizon_clean_data.append(int(ihoriz))
        x_axis_clean_data.append(float(x_pos))
        y_axis_clean_data.append(float(y_pos))
        phi_clean_data.append(float(phi))

    return horizon_clean_data, x_axis_clean_data, y_axis_clean_data, phi_clean_data


def prepare_data_for_plot(raw_data):
    """expected input list of strings of the form 'iteration=90, x=1.07848, y=-0.0640668'"""

    def unpack_data(__, x, y, z="dummy=0.0"):
        return (x, y, z)

    x_axis_clean_data = []
    y_axis_clean_data = []
    phi_clean_data    = []

    for entry in raw_data:
        x_pos, y_pos, phi = unpack_data(*entry.split(', '))
        __, x_pos = x_pos.split('=')
        __, y_pos = y_pos.split('=')
        __, phi = phi.split('=')
        x_axis_clean_data.append(float(x_pos))
        y_axis_clean_data.append(float(y_pos))
        phi_clean_data.append(float(phi))
    return x_axis_clean_data, y_axis_clean_data, phi_clean_data


def plot_walk_pattern(all_raw_data, prefix_name):
    """ """
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    # Set desired position
    x_axis, y_axis, phi = prepare_data_for_plot(all_raw_data['desired_position'])
    for point, angle in zip(zip(x_axis, y_axis), phi):
        axis.add_patch(draw_desired_position(point, angle,"red",4))

    x_axis, y_axis, phi = prepare_data_for_plot(all_raw_data['foot_position'])
    for point, angle in zip(zip(x_axis, y_axis), phi):
        axis.add_patch(convert_point_to_foot(point, angle))

    axis.add_patch(convert_point_to_foot((x_axis[-1],y_axis[-1]), phi[-1],"cyan",4))    
    axis.add_patch(convert_point_to_foot((x_axis[-2],y_axis[-2]), phi[-2],"cyan",4)) 

    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['com_position'])
    axis.plot(x_axis, y_axis, 'r', label='CoM', linewidth=graphwidth)

    # If predicted com positions are available
    if (len(all_raw_data['predicted_com_position'])>0):
        i_horiz, x_axis, y_axis, phi = prepare_predicted_data_for_plot(all_raw_data['predicted_com_position'])
        horizmax = max(i_horiz)
        nsteps   = int(len(i_horiz)/(horizmax+1))
        for k in range(0,nsteps):
            xpred  = x_axis[k*(horizmax+1):(k+1)*(horizmax+1)];
            ypred  = y_axis[k*(horizmax+1):(k+1)*(horizmax+1)];
            phipred= phi[k*(horizmax+1):(k+1)*(horizmax+1)];
            axis.plot(xpred, ypred, 'g', linewidth=1)

    # If predicted zmp positions are available
    if (len(all_raw_data['predicted_com_position'])>0):
        i_horiz, x_axis, y_axis, __ = prepare_predicted_data_for_plot(all_raw_data['predicted_zmp_position'])
        horizmax = max(i_horiz)
        nsteps   = int(len(i_horiz)/(horizmax+1))
        for k in range(0,nsteps):
            xpred  = x_axis[k*(horizmax+1):(k+1)*(horizmax+1)];
            ypred  = y_axis[k*(horizmax+1):(k+1)*(horizmax+1)];
            axis.plot(xpred, ypred, 'cyan', linewidth=1)

    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['zmp_position'])
    axis.plot(x_axis, y_axis, label='ZMP', linestyle='-', color='b', linewidth=graphwidth)
    axis.legend(loc=2,ncol=2)
    axis.set_xlabel("x-axis(m)",size=axisfontsize)
    axis.set_ylabel("y-axis(m)",size=axisfontsize)
    axis.set_ylim(-0.7,4.0)
    #axis.set_ylim(-2.0,0.5)
    axis.set_xlim(-0.5,7.0)
    axis.set_yticks(np.arange(-0.5, 4.5, step=0.5))
    axis.set_xticks(np.arange(-0.5, 7.5, step=0.5))
    axis.set_aspect('equal')
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black')
    axis.grid(True, linewidth=graphwidth)

    # axis.annotate('Perturbation',xy=((x_axis[44],y_axis[44])),xycoords='data',
    #               xytext=(130,300), textcoords='axes pixels',
    #               arrowprops=dict(facecolor='black', shrink=0.001, width=0.25),
    #               horizontalalignment='left', verticalalignment='center', size=24
    #               )

    figure.set_size_inches(1.65*figure.get_size_inches())
    figure.savefig('{0}_walk_pattern.pdf'.format(prefix_name), dpi=165, bbox_inches='tight')


def search_parameter_value(raw_data, parameter_name):
    """ """
    for entry in raw_data:
        if parameter_name in entry:
            __, parameter_value = entry.split('=')
            return parameter_value.strip()

    raise ValueError('ERROR: parameter \'{0}\' not found'.format(parameter_name))


def expand_parameter_value(value):
    """ """
    expanded_parameter = []
    for term in value.strip().split(','):
        term_expanded = term.split(':')
        if len(term_expanded) == 1:
            expanded_parameter.append(float(term_expanded[0]))
        elif len(term_expanded) == 2:
           value, times = term_expanded
           expanded_parameter = expanded_parameter + [float(value)]*int(times)
        else:
            raise ValueError('ERROR: Invalid term \'{0}\' found'.format(term))
    return expanded_parameter

def plot_com_conf(all_raw_data, prefix_name):
    """ """
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    theta = [entry.split()[1] for entry in all_raw_data['com_theta']]
    theta = [entry.split('=')[1] for entry in theta]
    theta = [float(entry) for entry in theta]
    try:
        raw_reference_theta = float(search_parameter_value(all_raw_data['parameter'], 'reference.orientation0'))
        axis.plot(raw_reference_theta*np.ones((len(theta),1)), 'y-', label='Reference global orientation', linewidth=graphwidth)
    except:
        pass


    ittheta = [entry.split()[0] for entry in all_raw_data['flying_foot_theta']]
    ittheta = [entry.split('=')[1] for entry in ittheta]
    ittheta = [entry.split(',')[0] for entry in ittheta]
    ittheta = [int(entry) for entry in ittheta]

    fftheta = [entry.split()[1] for entry in all_raw_data['flying_foot_theta']]
    fftheta = [entry.split('=')[1] for entry in fftheta]
    fftheta = [float(entry) for entry in fftheta]

    sftheta = [entry.split()[1] for entry in all_raw_data['support_foot_theta']]
    sftheta = [entry.split('=')[1] for entry in sftheta]
    sftheta = [float(entry) for entry in sftheta]
    #axis = figure.add_subplot(1, 1, 1)   
    # x axis data
    axis.plot(theta, 'r-', label='Trunk global orientation')
    ittheta_disc = []
    fftheta_disc = []
    sftheta_disc = []
    initdoublesupport = 2
    for i in range(initdoublesupport,len(sftheta)):
        ittheta_disc.append(ittheta[i])
        fftheta_disc.append(fftheta[i])
        sftheta_disc.append(sftheta[i])
        if (i%8 == initdoublesupport-1):
            axis.plot(ittheta_disc,fftheta_disc, 'g-',linewidth=1)    
            axis.plot(ittheta_disc,sftheta_disc, 'b-') 
            ittheta_disc = []
            fftheta_disc = []
            sftheta_disc = []
    #axis_plot = range(0,2)
    axis.plot(ittheta[0:initdoublesupport-1],fftheta[0:initdoublesupport-1], 'g-', label='Flying foot global orientation')    
    axis.plot(ittheta[0:initdoublesupport-1],sftheta[0:initdoublesupport-1], 'b-', label='Support foot global orientation') 
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Orientation (rad)",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.grid(True, linewidth=graphwidth)
    axis.legend(loc='best',fontsize=legendfontsize)
    figure.subplots_adjust(wspace=0.3, hspace=0.2)
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_com_theta.pdf'.format(prefix_name))

def plot_com_speed(all_raw_data, prefix_name):
    """ """
    figure = pyplot.figure()
    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['com_speed'])
    print("x_axis.shape = ",len(x_axis))

    # x axis data
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x_axis, 'b-', label=r'$v_{x} \ (m/s)$', linewidth=graphwidth)
    try:
        raw_reference_speed = search_parameter_value(all_raw_data['parameter'], 'reference.x_com_speed')
        axis.plot(expand_parameter_value(raw_reference_speed), 'b-', label='CoM reference speed', linewidth=graphwidth)
    except:
        pass

    axis.legend()

    # y axis data
    #axis = figure.add_subplot(1, 1, 1)    
    axis.plot(y_axis, 'r-', label=r'$v_{y} \ (m/s)$', linewidth=graphwidth)
    try:
        raw_reference_speed = search_parameter_value(all_raw_data['parameter'], 'reference.y_com_speed')
        axis.plot(expand_parameter_value(raw_reference_speed), 'b-', label='CoM reference speed', linewidth=graphwidth)
    except:
        pass

    axis.legend()

    angular = [entry.split()[1] for entry in all_raw_data['com_angular_speed']]
    angular = [entry.split('=')[1] for entry in angular]
    angular = [float(entry) for entry in angular]

    #Angular speed
    #axis = figure.add_subplot(1, 1, 1)    
    axis.plot(angular, 'k-', label=r'$\omega_{z} \ (rad/s)$', linewidth=graphwidth)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Velocities",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.grid(True, linewidth=graphwidth)
    axis.set_ylim(-0.5,0.5)
    axis.legend(loc=4,fontsize=legendfontsize+3, ncol=3)
    figure.subplots_adjust(wspace=0.3, hspace=0.2)
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_com_speed.pdf'.format(prefix_name))

    
def plot_com_acceleration(all_raw_data, prefix_name):
        
    """ """
    figure = pyplot.figure()
    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['com_acceleration'])

    # x axis data
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x_axis, 'b-', label=r'$a_{x} \ (m/s^2)$', linewidth=graphwidth)

    try:
        raw_reference_speed = search_parameter_value(all_raw_data['parameter'], 'reference.x_com_acceleration')
        axis.plot(expand_parameter_value(raw_reference_speed), 'b-', label='CoM reference acceleration')
    except:
        pass

    axis.legend()

    # y axis data
    axis = figure.add_subplot(1, 1, 1)    
    axis.plot(y_axis, 'r-', label=r'$a_{y} \ (m/s^2)$', linewidth=graphwidth)

    try:
        raw_reference_speed = search_parameter_value(all_raw_data['parameter'], 'reference.y_com_acceleration')
        axis.plot(expand_parameter_value(raw_reference_speed), 'b-', label='CoM reference acceleration')
    except:
        pass
    axis.legend()

    angular = [entry.split()[1] for entry in all_raw_data['com_angular_speed']]
    angular = [entry.split('=')[1] for entry in angular]
    angular = [float(entry) for entry in angular]

    #Angular acceleration
    axis = figure.add_subplot(1, 1, 1)    
    axis.plot(angular, 'k-', label=r'$\alpha_{z} \ (rad/s^2)$', linewidth=graphwidth)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel('Accelerations ',size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.grid(True, linewidth=graphwidth)

    axis.annotate('Perturbation',xy=((44,y_axis[44])),xycoords='data',
          xytext=(200,350), textcoords='axes pixels',
          arrowprops=dict(facecolor='black', shrink=0.001, width=0.25),
          horizontalalignment='left', verticalalignment='center', size=24
          )

    axis.legend(loc=4,ncol=3,fontsize=legendfontsize)
    axis.set_ylim(-2.4,2.4)
    #legend.get_frame().set_facecolor('#e7e7e7')
    #figure.set_size_inches(13, 13)
    figure.subplots_adjust(wspace=0.3, hspace=0.2)
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_com_acceleration.pdf'.format(prefix_name))    


def plot_anglejerks(all_raw_data, prefix_name):
    """ """
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    try:
        f_ang, t_ang, __ = prepare_data_for_plot(all_raw_data['angle_jerks'])
        axis.step(f_ang, 'y-', label='Jerks on the foot angle')
        axis.step(t_ang, 'g-', label='Jerks on the theta angle')
    except:
        pass    
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Jerk value",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.legend(loc='best',fontsize=legendfontsize)
    axis.grid(True, linewidth=graphwidth)

    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_angle_jerks.pdf'.format(prefix_name))
      
def plot_jerks(all_raw_data, prefix_name):
    """ """
    figure = pyplot.figure()
    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['jerks'])
    axis = figure.add_subplot(1, 1, 1)

    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Jerk value",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.step(x_axis, 'r-', label='Jerks on the x-axis')
    axis.step(y_axis, 'b-', label='Jerks on the y-axis')
    axis.legend(loc='best',fontsize=legendfontsize)
    axis.grid(True, linewidth=graphwidth)

    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_jerks.pdf'.format(prefix_name))


def plot_qp_time(all_raw_data, prefix_name):
    """ """
    units = all_raw_data['qp_time'][0].split()[1].strip() 
    values = [entry.split()[0] for entry in all_raw_data['qp_time']]
    values = [entry.split('=')[1] for entry in values]
    values = [int(entry) for entry in values]
    vmin = float(min(values))
    vmax = float(max(values))
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_title("QP time")
    axis.set_xlabel("Iterations",size=axisfontsize)
    major_ticks = np.arange(0, int(1.1*vmax), 5000)
    axis.set_yticks(major_ticks)
    axis.set_ylabel("CPU Time ({0})".format(units),size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3)
    axis.plot(values, 'r-', linestyle='-', linewidth=graphwidth)

    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_qp_time.pdf'.format(prefix_name))

def plot_translation_objective_function(all_raw_data, prefix_name):
    """ """
    values = [entry.split()[0] for entry in all_raw_data['objective_function']]
    values = [float(entry) for entry in values]
    vmin = float(min(values))
    vmax = float(max(values))
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Translation Objective function",size=axisfontsize)
    axis.grid(True, linewidth=graphwidth)
    axis.plot(values, 'r-', linestyle='-', linewidth=graphwidth)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black')

    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_translation_objective_function.pdf'.format(prefix_name))

def plot_orientation_objective_function(all_raw_data, prefix_name):
    """ """
    values = [entry.split()[0] for entry in all_raw_data['orientation_objective_function']]
    values = [float(entry) for entry in values]
    vmin = float(min(values))
    vmax = float(max(values))
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Orientation Objective function",size=axisfontsize)
    axis.grid(True, linewidth=graphwidth)
    axis.plot(values, 'r-', linestyle='-', linewidth=graphwidth)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black')
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_orientation_objective_function.pdf'.format(prefix_name))

def plot_camera_images(all_raw_data, prefix_name):

    def _plot_reference_image(axis, user_points, color, label):
        all_points = user_points[:]

        centroid = (sum([p[0] for p in all_points]) / len(all_points),
                    sum([p[1] for p in all_points]) / len(all_points))
        all_points.sort(key=lambda p: math.atan2(p[1]-centroid[1], p[0] - centroid[0]))

        axis.scatter([point[0] for point in all_points],
                     [point[1] for point in all_points], color=color, label=label,linewidths=7)

    # determine the number of reference points for reference 0
    nRef    =  int(len([entry for entry in all_raw_data['reference_image_point'] if 'reference_id=000' in entry]))
    
    # points in the first reference image
    raw_reference_points = [entry for entry in all_raw_data['reference_image_point'] if 'reference_id=000' in entry]
    reference_points = []
    for raw_point in raw_reference_points:
        __, __, __, x, y, __ = parse_raw_log_entry(raw_point)
        reference_points.append((x, y))

    # points in the first virtual reference image
    raw_virtual_reference_points = [entry for entry in all_raw_data['virtual_reference_image_point'] if 'reference_id=000' in entry]                              
    virtual_reference_points = []
    for raw_point in raw_virtual_reference_points:
        __, __, __, x, y, __ = parse_raw_log_entry(raw_point)
        virtual_reference_points.append((x, y))

    # points in the initial image
    raw_initial_points = [entry for entry in all_raw_data['reference_image_from_initial_position'] if 'reference_id=000' in entry]
    initial_points = []
    for raw_point in raw_initial_points:
        __, __, __, x, y, __ = parse_raw_log_entry(raw_point)
        initial_points.append((x, y))

    # trajectories
    raw_trajectories = [all_raw_data['current_image_point'][x : x + nRef]
                        for x in range(0, len(all_raw_data['current_image_point']), nRef)]

    raw_trajectories = [entry for entry in all_raw_data['current_image_point'] if 'reference_id=000' in entry] 
    all_trajectories = [[]] *nRef 
    point_draw = [[]]

    # For each reference point
    for i in range(nRef):
        points_in_trajectory = [data for data in raw_trajectories if ('point_id='+('{:0>3d}'.format(i))) in data]
        all_trajectories[i]  = []
        for raw_point in points_in_trajectory:
            __, __, __, x, y, v = parse_raw_log_entry(raw_point)
            all_trajectories[i].append((x, y, v))

    # Start plotting
    figure = pyplot.figure()
    axis   = figure.add_subplot(1, 1, 1)

    # Plotting virtual reference points 
    if (len(virtual_reference_points)>0):
        _plot_reference_image(axis, virtual_reference_points, 'green', 'Virtual reference features')

    # Plotting reference points
    _plot_reference_image(axis, reference_points, 'red', 'Reference features')
    # Plotting initial features points    
    _plot_reference_image(axis, initial_points, 'blue', 'Initial features')

    # Create a colormap for red, green and blue and a norm to color
    #  < -0.5 red, f' > 0.5 blue, and the rest green
    cmap = ListedColormap(['black','green'])
    norm = BoundaryNorm([-10.0, 0.5, 10.0], cmap.N)
    # Plotting trajectories
    for trajectory in all_trajectories:
        x_coords = [x_ for x_, __, __ in trajectory]
        y_coords = [y_ for __, y_, __ in trajectory]
        v_coords = np.array([v_ for __, __, v_ in trajectory])
        points   = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(v_coords)
        lc.set_linewidth(trajwidth)
        axis.add_collection(lc)
        axis.plot(x_coords, y_coords, color='black', linewidth=trajwidth)

    axis.set_xlabel("u-coordinate",size=axisfontsize)
    axis.set_ylabel("v-coordinate",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black')
    axis.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2,fontsize=legendfontsize)
    #legend.get_frame().set_facecolor('#e7e7e7')
    axis.grid(True, linewidth=graphwidth)

    figure.set_size_inches(1.3*figure.get_size_inches())
    figure.savefig('{0}_camera.pdf'.format(prefix_name), bbox_inches="tight")


def plot_detailed_visual_features(all_raw_data, prefix_name):

    def _split_raw_string(raw_entry):
        #__, name, actual, predicted, expected, ground_truth = raw_entry.split(', ')
        __, name, actual, predicted, expected = raw_entry.split(', ')
        __, name = name.split('=')
        __, actual = actual.split('=')
        __, predicted = predicted.split('=')
        __, expected = expected.split('=')
        #__, ground_truth = ground_truth.split('=')
        #return (name, float(actual), float(predicted), float(expected), float(ground_truth))
        return (name, float(actual), float(predicted), float(expected))

    def _parse_visual_entries(all_raw_entires):
        parsed_data = {}
        for entry in all_raw_entires:
            #name, actual, predicted, expected, ground_truth = _split_raw_string(entry)
            name, actual, predicted, expected = _split_raw_string(entry)
            if name in parsed_data:
                parsed_data[name][0].append(actual)
                parsed_data[name][1].append(predicted)
                parsed_data[name][2].append(expected)
                #parsed_data[name][3].append(ground_truth)
            else:
                #parsed_data[name] = ([actual], [predicted], [expected],[ground_truth])
                parsed_data[name] = ([actual], [predicted], [expected])
        return parsed_data


    parsed_data = _parse_visual_entries(all_raw_data['visual_feature'])

    figure = pyplot.figure()
    colors = ['red','green','blue','magenta','cyan','brown','yellow']
    axis = figure.add_subplot(1, 1, 1)
    for index, name in enumerate(parsed_data.keys()):
        #actual_values, predicted_values, expected_values, ground_truth_values = parsed_data[name]
        actual_values, predicted_values, expected_values = parsed_data[name]
        axis.plot(actual_values, color=colors[index], label=r'%s observed'%(name), linestyle='-', linewidth=graphwidth)
        axis.plot(predicted_values, color=colors[index], label=r'%s predicted'%(name), linestyle='--', linewidth=graphwidth)
        #axis.plot(ground_truth_values, color=colors[index], label=r'%s ground truth'%(name), linestyle=':', linewidth=graphwidth)
        #axis.plot(expected_values, color=colors[index], label=r'%s reference'%(name), linestyle='--', linewidth=graphwidth)     

    axis.grid(True, linewidth=graphwidth)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Visual constraint elements",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3) 
    axis.set_ylim(-1.2,1.2)

    try:
        __, __, expected_values, = parsed_data['h11']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth) 
        axis.legend(loc='best',fontsize=legendfontsize-3,ncol=3)
        __, __, expected_values, = parsed_data['h12']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth)
    except KeyError:
        __, __, expected_values, __ = parsed_data['e12']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth)
        axis.legend(loc='best',fontsize=legendfontsize-3,ncol=3)
    
    figure.subplots_adjust(wspace=0.3, hspace=0.3)
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_detailed_visual_features.pdf'.format(prefix_name))


def main():
    """ main function """
    if len(sys.argv) < 2:
        print( "Usage:")
        print("    python {0} <options> <log>\n".format(sys.argv[0]))
        sys.exit(-1)

    for log_file in sys.argv[1:]:
        print("Analyzing {0}...".format(log_file))
        try:
            raw_data = load_data(log_file)
        except IOError:
            print("ERROR: Invalid log file '{0}'".format(log_file))
            sys.exit(-1)

        prefix_name, __ = os.path.splitext(log_file)

        plot_functions = (plot_detailed_visual_features,plot_walk_pattern,plot_com_conf,plot_camera_images,plot_com_speed,plot_translation_objective_function)
        #plot_functions = (plot_walk_pattern,plot_detailed_visual_features)
        #plot_functions = (plot_walk_pattern,plot_camera_images,plot_com_acceleration)
        #plot_functions = (plot_com_speed,plot_com_conf)
        for function in plot_functions:
            try:
                function(raw_data, prefix_name)
            except:
                print("WARNING: Unable to plot '{}' skipping ...".format(function.func_name))
                raise


if __name__ == '__main__':
    main()
