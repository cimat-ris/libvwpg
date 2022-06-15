import subprocess
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
from matplotlib.patches import Ellipse

#Intrinsic camera parameters of the HRP-2 
fx_hrp = 391.59372
fy_hrp = 391.59472
u0_hrp = 340.82448
v0_hrp = 274.59020

legendfontsize = 18
ticksfontsize  = 14
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

def dib_robot(location,side=0.4):
	"""
	dib_robot(location,attributes,lado,color)
	location: [x y chi]
	attributes: Ej attibutes: 'r-'   'b.'
	side: (optativo) tamano del robot
	color: (optativo) Ej color: [0.1 1 0.5]

	Modelo robot: 
              |x
              |
        b-----e-----c
        |    /|\    |
        |   / | \   |
  y ----------0  \  |
        | /       \ |
        |/         \|
        a-----------d

	"""
	side_coseno = side*math.cos(location[2])
	side_seno   = side*math.sin(location[2])
	model = np.array([
		              [-1,  1, 1],#a
					  [ 1,  1, 1],
					  [ 1, -1, 1],
					  [-1, -1, 1],
					  [-1,  1, 1],
					  [ 1,  0, 1],
					  [-1, -1, 1] 
					  ])

	transformation = np.array([
								[side_coseno, -side_seno  , location[0]],
								[side_seno  ,  side_coseno, location[1]],
								[     0     ,       0     ,      1]
							 ])

	robot = np.dot( transformation,model.transpose() )
	robot = robot.transpose()

	return robot

	# figure = pyplot.figure()
	# axis = figure.add_subplot(1, 1, 1)
	# axis.plot(robot[:,0],robot[:,1],attributes,color=color)
	# return figure

def run_camera_parameters_experiment(file_name,percentace_focal,percentace_principal_point,exe,ini_file,fx=fx_hrp,fy=fy_hrp,u0=u0_hrp,v0=v0_hrp):
    fx_perturbed = np.random.normal(0.0,fx*percentace_focal/2,1)
    fy_perturbed = np.random.normal(0.0,fy*percentace_focal/2,1)
    u0_perturbed = np.random.normal(0.0,u0*percentace_principal_point/2,1)
    v0_perturbed = np.random.normal(0.0,v0*percentace_principal_point/2,1)

    camerafx = "-camera.fx=" + str(fx+fx_perturbed[0])
    camerafy = "-camera.fy=" + str(fy+fy_perturbed[0])
    camerau0 = "-camera.u0=" + str(u0+u0_perturbed[0])
    camerav0 = "-camera.v0=" + str(v0+v0_perturbed[0])
    camerano = "-camera.sigma_noise=4.0"
    print camerafx, camerafy, camerau0, camerav0, camerano

    with open(file_name, "w") as f:
        subprocess.call([exe,ini_file,camerafx,camerafy,camerau0,camerav0,camerano], stdout=f)

def run_simple_experiment_with_an_desired_angle(file_name,exe,ini_file,angle=0.0):
    print "The angles is: ", angle
    angle_new = "reference.orientation0=" + str(angle)
    with open(file_name, "w") as f:
        subprocess.call([exe,ini_file,angle_new], stdout=f)    

def run_camera_parameters_experiment_variable_gains(file_name,percentace_focal,percentace_principal_point,exe,ini_file,list_homo_elements,fx=fx_hrp,fy=fy_hrp,u0=u0_hrp,v0=v0_hrp):

	fx_perturbed = np.random.normal(0.0,fx*percentace_focal,1)
	fy_perturbed = np.random.normal(0.0,fy*percentace_focal,1)
	u0_perturbed = np.random.normal(0.0,u0*percentace_principal_point,1)
	v0_perturbed = np.random.normal(0.0,v0*percentace_principal_point,1)

	camerafx = "-camera.fx=" + str(fx+fx_perturbed[0])
	camerafy = "-camera.fy=" + str(fy+fy_perturbed[0])
	camerau0 = "-camera.u0=" + str(u0+u0_perturbed[0])
	camerav0 = "-camera.v0=" + str(v0+v0_perturbed[0])
	camerano = "-camera.sigma_noise=2.0"
	print camerafx, camerafy, camerau0, camerav0, camerano

	with open(file_name, "w") as f:
	    subprocess.call([exe,ini_file,camerafx,camerafy,camerau0,camerav0,camerano,list_homo_elements[0],list_homo_elements[1],list_homo_elements[2],list_homo_elements[3],list_homo_elements[4],list_homo_elements[5]], stdout=f)



def unpack_data(*packed_data):
    unpacked_data = []
    for entry in packed_data:
        try:
            entry = float(entry)
        except ValueError:
            pass
        unpacked_data.append(entry)
    return unpacked_data



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

def convert_point_to_foot(center, angle, color="black", linewidth=graphwidth,alpha=1.0):
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
    return patches.Rectangle((x, y), length, width, angle, fill=False, edgecolor=color, linewidth=linewidth,alpha=alpha)



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


def plot_walk_pattern(all_raw_data, figure):
    """ """

    axis = figure.add_subplot(1, 1, 1)

    # Set desired position
    x_axis, y_axis, phi = prepare_data_for_plot(all_raw_data['desired_position'])
    for point, angle in zip(zip(x_axis, y_axis), phi):
        axis.add_patch(draw_desired_position(point, angle,"red",4))

    x_axis, y_axis, phi = prepare_data_for_plot(all_raw_data['foot_position'])
    for point, angle in zip(zip(x_axis, y_axis), phi):
        axis.add_patch(convert_point_to_foot(point, angle))

    axis.add_patch(convert_point_to_foot((x_axis[-1],y_axis[-1]), phi[-1],"cyan",4,1.5))    
    axis.add_patch(convert_point_to_foot((x_axis[-2],y_axis[-2]), phi[-2],"cyan",4,1.5))   

    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['com_position'])
    axis.plot(x_axis, y_axis, 'r', label='CoM', linewidth=graphwidth)

    x_axis, y_axis, __ = prepare_data_for_plot(all_raw_data['zmp_position'])
    axis.plot(x_axis, y_axis, label='ZMP', linestyle='-', color='b', linewidth=graphwidth)
    axis.legend(loc=2,ncol=2)
    axis.set_xlabel("x-axis(m)",size=axisfontsize)
    axis.set_ylabel("y-axis(m)",size=axisfontsize)
    axis.set_ylim(-0.5,5.0)
    #axis.set_ylim(-2.0,0.5)
    axis.set_xlim(-1.0,7.0)
    axis.set_yticks(np.arange(-0.5, 5.5, step=0.5))
    axis.set_xticks(np.arange(-1.5, 7.5, step=0.5))
    axis.set_aspect('equal')
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black')
    axis.grid(True, linewidth=1.0)

def plot_detailed_visual_features(all_raw_data, prefix_name):

    def _split_raw_string(raw_entry):
        __, name, actual, predicted, expected, ground_truth = raw_entry.split(', ')
        __, name = name.split('=')
        __, actual = actual.split('=')
        __, predicted = predicted.split('=')
        __, expected = expected.split('=')
        __, ground_truth = ground_truth.split('=')
        return (name, float(actual), float(predicted), float(expected), float(ground_truth))

    def _parse_visual_entries(all_raw_entires):
        parsed_data = {}
        for entry in all_raw_entires:
            name, actual, predicted, expected, ground_truth = _split_raw_string(entry)
            if name in parsed_data:
                parsed_data[name][0].append(actual)
                parsed_data[name][1].append(predicted)
                parsed_data[name][2].append(expected)
                parsed_data[name][3].append(ground_truth)
            else:
                parsed_data[name] = ([actual], [predicted], [expected],[ground_truth])
        return parsed_data


    parsed_data = _parse_visual_entries(all_raw_data['visual_feature'])

    figure = pyplot.figure()
    colors = ['red','green','blue','magenta','cyan','brown','yellow']
    axis = figure.add_subplot(1, 1, 1)
    for index, name in enumerate(parsed_data.keys()):
        actual_values, predicted_values, expected_values, ground_truth_values = parsed_data[name]
        axis.plot(actual_values, color=colors[index], label=r'%s observed'%(name), linestyle='-', linewidth=graphwidth)
        axis.plot(predicted_values, color=colors[index], label=r'%s predicted'%(name), linestyle=':', linewidth=graphwidth)
        #axis.plot(ground_truth_values, color=colors[index], label=r'%s ground truth'%(name), linestyle=':', linewidth=graphwidth)
        #axis.plot(expected_values, color=colors[index], label=r'%s reference'%(name), linestyle='--', linewidth=graphwidth)     

    axis.grid(True, linewidth=graphwidth)
    axis.set_xlabel("Iterations",size=axisfontsize)
    axis.set_ylabel("Visual constraint elements",size=axisfontsize)
    axis.tick_params(labelsize=ticksfontsize,labelcolor='black',pad=3) 
    axis.set_ylim(-0.8,1.2)

    try:
        __, __, expected_values,__ = parsed_data['h11']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth) 
        axis.legend(loc='best',fontsize=legendfontsize-3,ncol=3)
        __, __, expected_values, __ = parsed_data['h12']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth)
    except KeyError:
        __, __, expected_values, __ = parsed_data['e12']
        axis.plot(expected_values, color='black', label='reference', linestyle='--', linewidth=graphwidth)
        axis.legend(loc='best',fontsize=legendfontsize-5,ncol=3)
    
    figure.subplots_adjust(wspace=0.3, hspace=0.3)
    figure.set_size_inches(1.25*figure.get_size_inches())
    figure.savefig('{0}_detailed_visual_features.pdf'.format(prefix_name))

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def main():
    exe      = "/home/noe/Documentos/webots/projects/robots/aldebaran/aldebaran/simulator-sdk/bin/test1-vwpg"
    ini_file = "/home/noe/Documentos/bitbucket/visualmemory/code/nao_simulations/libvwpg/working_configs/homography/hrp_hlineal_8.ini"

    fx = fx_hrp
    fy = fy_hrp
    u0 = u0_hrp
    v0 = v0_hrp
    percentace_focal = 0.05
    percentace_principal_point = 0.0
    elements_homo = ["h11_h12","h11_h13","h11_h31","h11_h32","h11_h33","h12_h13","h12_h31","h12_h32","h12_h33","h13_h31","h13_h32","h13_h33","h31_h32","h31_h33","h32_h33"]
    list_homo_elements = [ ["-qp.betah11=3.0","-qp.betah12=1.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.0"], 
                           ["-qp.betah11=3.0","-qp.betah12=0.0","-qp.betah13=3.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.0"],
                           ["-qp.betah11=3.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=2.0","-qp.betah32=0.0","-qp.betah33=0.0"],
                           ["-qp.betah11=3.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=1.0","-qp.betah33=0.0"],
                           ["-qp.betah11=3.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.5"],
                           ["-qp.betah11=0.0","-qp.betah12=1.0","-qp.betah13=3.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.0"],
                           ["-qp.betah11=0.0","-qp.betah12=1.0","-qp.betah13=0.0","-qp.betah31=2.0","-qp.betah32=0.0","-qp.betah33=0.0"], 
                           ["-qp.betah11=0.0","-qp.betah12=1.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=1.0","-qp.betah33=0.0"],
                           ["-qp.betah11=0.0","-qp.betah12=1.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.5"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=3.0","-qp.betah31=2.0","-qp.betah32=0.0","-qp.betah33=0.0"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=3.0","-qp.betah31=0.0","-qp.betah32=1.0","-qp.betah33=0.0"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=3.0","-qp.betah31=0.0","-qp.betah32=0.0","-qp.betah33=0.5"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=2.0","-qp.betah32=1.0","-qp.betah33=0.0"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=2.0","-qp.betah32=0.0","-qp.betah33=0.5"],
                           ["-qp.betah11=0.0","-qp.betah12=0.0","-qp.betah13=0.0","-qp.betah31=0.0","-qp.betah32=1.0","-qp.betah33=0.5"]]

    for i, homo_elements in enumerate(list_homo_elements):
        elements = elements_homo[i]
        file_name_base = "diagonal_automatic_" + elements + "_"
        file_name = file_name_base + str(0) + ".txt"

        number_of_tests = 101
        x_final_position = []
        y_final_position = []

        # Figure to plot the final configurations.
        figure = pyplot.figure()

        # This part corresponds to the experiment with real camera parameters.
        run_camera_parameters_experiment_variable_gains(file_name,0.0,0.0,exe,ini_file,homo_elements,fx,fy,u0,v0)
        raw_data = load_data(file_name)

        # Print the walking pattern with the real camera parameters.
        plot_walk_pattern(raw_data, figure)

        #This part corresponds to the experiments with bad camera parameters
        axis = figure.add_subplot(1, 1, 1)

        for i in range(1,number_of_tests):
            file_name = file_name_base + str(i) + ".txt"
            run_camera_parameters_experiment_variable_gains(file_name,percentace_focal,percentace_principal_point,exe,ini_file,homo_elements,fx,fy,u0,v0)
            raw_data = load_data(file_name)
            x_axis, y_axis, phi = prepare_data_for_plot(raw_data['foot_position'])
            robot = dib_robot([(x_axis[-1]+x_axis[-2])/2,(y_axis[-1]+y_axis[-2])/2,phi[-1]],0.05)
            axis.plot(robot[:,0],robot[:,1],'r-',color='red',alpha=0.25)
            x_final_position.append((x_axis[-1]+x_axis[-2])/2)
            y_final_position.append((y_axis[-1]+y_axis[-2])/2)

        cov    = np.cov(np.array(x_final_position),np.array(y_final_position))
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        nstd = 2
        w, h = 2 * nstd * np.sqrt(vals)
        axis.add_patch(Ellipse(xy=( np.mean(x_final_position), np.mean(y_final_position) ), 
                                width=w, height=h, angle=theta,
                            edgecolor='green', fc='None', lw=2, alpha=1.2)) 
        figure.savefig('{0}_robot.pdf'.format(("hrp_hlineal_8_walk_pattern_" + elements)), dpi=165, bbox_inches='tight')

        mean_x = "mean_x: " + str(np.mean(x_final_position)) + " , "
        mean_y = "mean_y: " + str(np.mean(y_final_position)) + " , "
        cov_matrix = "cov_matrix: " + str(cov) + " , "
        with open(("stadistics_vales_" + elements + ".txt"), "w") as f:
            f.write(mean_x)
            f.write(mean_y)
            f.write(cov_matrix)

def one_gain():
    exe      = "/home/noe/Documentos/webots/projects/robots/aldebaran/aldebaran/simulator-sdk/bin/test1-vwpg"
    ini_file = "/home/noe/Documentos/bitbucket/visualmemory/code/nao_simulations/libvwpg/working_configs/homography/hrp_hlineal_8.ini"

    fx = fx_hrp
    fy = fy_hrp
    u0 = u0_hrp
    v0 = v0_hrp
    percentace_focal = 0.0
    percentace_principal_point = 0.0

    elements = "all_four"
    file_name_base = "diagonal_automatic_" + elements + "_"
    file_name = file_name_base + str(0) + ".txt"

    number_of_tests = 101
    x_final_position = []
    y_final_position = []

    # Figure to plot the final configurations.
    figure = pyplot.figure()

    # This part corresponds to the experiment with real camera parameters.
    run_camera_parameters_experiment(file_name,0.0,0.0,exe,ini_file,fx,fy,u0,v0)
    raw_data = load_data(file_name)

    # Print the walking pattern with the real camera parameters.
    plot_walk_pattern(raw_data, figure)
    plot_detailed_visual_features(raw_data, ("homography_" + elements) )

    #This part corresponds to the experiments with bad camera parameters
    axis = figure.add_subplot(1, 1, 1)

    for i in range(1,number_of_tests):
        file_name = file_name_base + str(i) + ".txt"
        run_camera_parameters_experiment(file_name,percentace_focal,percentace_principal_point,exe,ini_file,fx,fy,u0,v0)
        raw_data = load_data(file_name)
        x_axis, y_axis, phi = prepare_data_for_plot(raw_data['foot_position'])
        robot = dib_robot([(x_axis[-1]+x_axis[-2])/2,(y_axis[-1]+y_axis[-2])/2,phi[-1]],0.05)
        axis.plot(robot[:,0],robot[:,1],'r-',color='red',alpha=0.25)
        x_final_position.append((x_axis[-1]+x_axis[-2])/2)
        y_final_position.append((y_axis[-1]+y_axis[-2])/2)

    cov    = np.cov(np.array(x_final_position),np.array(y_final_position))
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    nstd = 2
    w, h = 2 * nstd * np.sqrt(vals)
    axis.add_patch(Ellipse(xy=( np.mean(x_final_position), np.mean(y_final_position) ), 
                            width=w, height=h, angle=theta,
                        edgecolor='green', fc='None', lw=2, alpha=1.2)) 
    figure.savefig('{0}_robot.pdf'.format(("hrp_hlineal_8_walk_pattern_" + elements)), dpi=165, bbox_inches='tight')

    mean_x = "mean_x: " + str(np.mean(x_final_position)) + " , "
    mean_y = "mean_y: " + str(np.mean(y_final_position)) + " , "
    cov_matrix = "cov_matrix: " + str(cov) + " , "
    with open(("stadistics_vales_" + elements + ".txt"), "w") as f:
        f.write(mean_x)
        f.write(mean_y)
        f.write(cov_matrix)

def qp_time(all_raw_data,tag='qp_time'):
    """ """
    units = all_raw_data[tag][0].split()[1].strip() 
    values = [entry.split()[0] for entry in all_raw_data[tag]]
    values = [entry.split('=')[1] for entry in values]
    values = [int(entry) for entry in values]
    return np.mean(values)

def com_theta(all_raw_data):    
    theta = [entry.split()[1] for entry in all_raw_data['com_theta']]
    theta = [entry.split('=')[1] for entry in theta]
    theta = [float(entry) for entry in theta]
    return theta[-1]

def support_foot_theta(all_raw_data):
    sftheta = [entry.split()[1] for entry in all_raw_data['support_foot_theta']]
    sftheta = [entry.split('=')[1] for entry in sftheta]
    sftheta = [float(entry) for entry in sftheta]
    return sftheta[-1]

def simple_experiments():
    exe      = "/home/noe/Documentos/webots/projects/robots/aldebaran/aldebaran/simulator-sdk/bin/test1-vwpg"
    ini_file = "/home/noe/Documentos/bitbucket/visualmemory/code/nao_simulations/libvwpg/working_configs/homography-nonlinear/hrp_hnlineal_u0.ini"
    elements = "u0_prev"
    file_name_base = "nonlinear_45_" + elements + "_"

    final_foot_euc_distance   = []
    final_CoM_euc_distance    = []
    qp_time_final             = []
    final_CoM_orientation     = []
    final_foot_orientation    = []
    final_orientation_desired = []

    x_final_position = []
    y_final_position = []

    # Figure to plot the final configurations.
    figure = pyplot.figure()

    # This part corresponds to the experiment with real camera parameters.
    file_name = file_name_base + str(0) + ".txt"
    #run_simple_experiment_with_an_desired_angle(file_name,exe,ini_file,45*math.pi/180)
    raw_data = load_data(file_name)

    # Print the walking pattern with the real camera parameters.
    plot_walk_pattern(raw_data, figure)

    #This part corresponds to the experiments with bad camera parameters
    axis = figure.add_subplot(1, 1, 1)

    for i in range(1,101):
        file_name = file_name_base + str(i) + ".txt"
        #run_simple_experiment_with_an_desired_angle(file_name,exe,ini_file,45*math.pi/180)
        raw_data = load_data(file_name)
        x_desired, y_desired, phi_desired = prepare_data_for_plot(raw_data['desired_position'])
        final_orientation_desired.append(phi_desired[0])
        x_axis, y_axis, phi = prepare_data_for_plot(raw_data['foot_position'])  
        norm_dif_foot_position  = np.linalg.norm(np.array(( (x_axis[-1]+x_axis[-2])/2 , (y_axis[-1]+y_axis[-2])/2 ))-np.array((x_desired[0],y_desired[0])))
        final_foot_euc_distance.append(norm_dif_foot_position) 
        x_final_position.append((x_axis[-1]+x_axis[-2])/2)
        y_final_position.append((y_axis[-1]+y_axis[-2])/2)
        robot = dib_robot([(x_axis[-1]+x_axis[-2])/2,(y_axis[-1]+y_axis[-2])/2,phi[-1]],0.05)
        axis.plot(robot[:,0],robot[:,1],'r-',color='red',alpha=0.25)
        x_axis, y_axis, __ = prepare_data_for_plot(raw_data['com_position'])
        norm_dif_CoM_position   = np.linalg.norm(np.array((x_axis[-1],y_axis[-1]))-np.array((x_desired[-1],y_desired[-1])))
        final_CoM_euc_distance.append(norm_dif_CoM_position) 
        qp_time_translation = qp_time(raw_data,'qp_time')
        qp_time_final.append(qp_time_translation) 
        final_CoM_orientation.append(com_theta(raw_data))
        final_foot_orientation.append(support_foot_theta(raw_data))

    cov    = np.cov(np.array(x_final_position),np.array(y_final_position))
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    nstd = 2
    w, h = 2 * nstd * np.sqrt(vals)
    axis.add_patch(Ellipse(xy=( np.mean(x_final_position), np.mean(y_final_position) ), 
                width=w, height=h, angle=theta,
            edgecolor='green', fc='None', lw=2, alpha=1.2)) 
    figure.savefig('{0}_robot.pdf'.format(("hrp_hlineal_45_walk_pattern_" + elements)), dpi=165, bbox_inches='tight')

    mean_norm_dif_foot_position    = "mean_norm_dif_foot_position: " + str(np.mean( np.array(final_foot_euc_distance))) + " , "
    mean_norm_dif_CoM_position     = "mean_norm_dif_CoM_position: " + str(np.mean(np.array(final_CoM_euc_distance))) + " , "
    mean_qp_time_final             = "mean_qp_time_final: " + str(np.mean(np.array(qp_time_final))) + " , "
    var_norm_dif_foot_position     = "var_norm_dif_foot_position: " + str(np.var(np.array(final_foot_euc_distance))) + " , "
    var_norm_dif_CoM_position      = "var_norm_dif_CoM_position: " + str(np.var(np.array(final_CoM_euc_distance))) + " , "
    var_qp_time_final              = "var_qp_time_final: " + str(np.var(np.array(qp_time_final))) + " , "
    values_final_foot_euc_distance = "values_final_foot_euc_distance:" + str(final_foot_euc_distance) + " , "
    values_final_CoM_euc_distance  = "values_final_CoM_euc_distance:" + str(final_CoM_euc_distance) + " , "
    values_qp_time_final           = "values_qp_time_final:" + str(qp_time_final) + " , "
    values_final_foot_orientation  = "values_final_foot_orientation:" + str(final_foot_orientation) + " , "
    values_final_CoM_orientation   = "values_final_CoM_orientation:" + str(final_CoM_orientation) + " , "
    values_orientation_desired    = "values_orientation_desired" + str(final_orientation_desired) + " , "
    with open(("stadistics_vales_" + elements + ".txt"), "w") as f:
        f.write(mean_norm_dif_foot_position)
        f.write(mean_norm_dif_CoM_position)
        f.write(mean_qp_time_final)
        f.write(var_norm_dif_foot_position)
        f.write(var_norm_dif_CoM_position)
        f.write(var_qp_time_final)
        f.write(values_final_foot_euc_distance)
        f.write(values_final_CoM_euc_distance)
        f.write(values_qp_time_final)
        f.write(values_final_foot_orientation)
        f.write(values_final_CoM_orientation)
        f.write(values_orientation_desired)

if __name__ == '__main__':
    simple_experiments()