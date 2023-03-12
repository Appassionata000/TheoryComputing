import numpy as np
import matplotlib.pyplot as plt

ANG_FREQUENCY_SQUARED = 1
START_TIME = 0
END_TIME = 20
INITIAL_POSITION = 0
INITIAL_VELOCITY = 0.2

cmap1 = ['#fe817d', '#81b8df']

def deriv(t, phase):
    # phase[0], phase[1]: position, velocity
    return np.array([phase[1], -ANG_FREQUENCY_SQUARED * phase[0]])

def RK4(deriv, phase, t, h):
    '''
    One step of RK4 Method
    phase[0], phase[1]: position, velocity
    h: step size
    '''
    k1 = deriv(t, phase)
    k2 = deriv(t + h/2, phase + h/2 * k1)
    k3 = deriv(t + h/2, phase + h/2 * k2)
    k4 = deriv(t + h, phase + h * k3)
    phase = phase + h*(k1 + 2 * k2 + 2 * k3 + k4) / 6
    t = t + h
    return (t, phase)

def solve_ODE(phase, t, h):

    time_list = np.array([t])     # to store all times
    phase_list = np.array([phase])     # and all solution points

    for i in range(int(END_TIME / h)):  # take enough steps (or so)
        (t, phase) = RK4(deriv, phase, t, h)
        time_list = np.append(time_list, t)
        phase_list = np.concatenate((phase_list, np.array([phase])))

    return (time_list, phase_list)

def plot_solutions(t_lists, pos_lists, v_lists, h_list):

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    k = 0

    for i in range(2):
        for j in range(3):
            axs[i, j].plot(t_lists[k], pos_lists[k], color=cmap1[0], label='position')
            axs[i, j].plot(t_lists[k], v_lists[k], color=cmap1[1], label='velocity')
            axs[i, j].set_title('$h$={}'.format(h_list[k]))
            axs[i, j].set_xlabel('t(s)')
            k += 1

    axs[0, 1].legend(loc='lower left', bbox_to_anchor=(0, 1.2, 1, 0.2), mode='expand', ncol=2)
    fig.tight_layout()

def phase_portrait(pos_lists, v_lists, h_list):

    fig, axs = plt.subplots(2, 3, figsize=(8, 5), subplot_kw=dict(box_aspect=1),
                            constrained_layout=True)

    k = 0

    for i in range(2): 
        for j in range(3):
            axs[i, j].plot(pos_lists[k], v_lists[k], color=cmap1[0])
            axs[i, j].set_title('$h$={}'.format(h_list[k]))
            axs[i, j].set_xlabel('$x$', fontsize=12)
            axs[i, j].set_ylabel('$v$', fontsize=12)
            
            k += 1

    fig.tight_layout()

initial = np.array([INITIAL_POSITION, INITIAL_VELOCITY])
h_list = [0.02, 0.1, 0.5, 1, 2, 3]

t_lists = []
pos_lists = []
v_lists = []
    
for i in range(len(h_list)):

    t_list, phase_list = solve_ODE(initial, START_TIME, h_list[i])
    [pos_list, v_list] = phase_list.transpose()

    t_lists.append(t_list)
    pos_lists.append(pos_list)
    v_lists.append(v_list)

plot_solutions(t_lists, pos_lists, v_lists, h_list)
phase_portrait(pos_lists, v_lists, h_list)

plt.show()