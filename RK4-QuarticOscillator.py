import numpy as np
import matplotlib.pyplot as plt


# cmap = ['#fe817d', '#81b8df', '#8ECFC9', '#FFBE7A']
cmap = ['#ED5E58', '#4BA78E', '#51569E', '#FFBE7A']
# cmap = ['#ED5E58', '#51569E']




ANG_FREQUENCY_SQUARED = 1
START_TIME = 0
END_TIME = 200
INITIAL_POSITION = 0
INITIAL_VELOCITY = 0.2

h = 0.01

# CHANGE COEFFICIENTS HERE
coeff_list = np.array([0, -10])

cmap1 = ['#fe817d', '#81b8df']

def deriv(t, phase, coeff):
    # phase[0], phase[1]: position, velocity
    return np.array([phase[1], -ANG_FREQUENCY_SQUARED * (phase[0] + coeff * phase[0]**3)])

def RK4(deriv, phase, t, coeff):
    '''
    One step of RK4 Method
    phase[0], phase[1]: position, velocity
    h: step size
    '''
    k1 = deriv(t, phase, coeff)
    k2 = deriv(t + h/2, phase + h/2 * k1, coeff)
    k3 = deriv(t + h/2, phase + h/2 * k2, coeff)
    k4 = deriv(t + h, phase + h * k3, coeff)
    phase = phase + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t = t + h
    return (t, phase)

def solve_ODE(phase, t, coeff):

    time_list = np.array([t])          # to store all times
    phase_list = np.array([phase])     # and all solution points

    for i in range(int(END_TIME / h)):  # take enough steps (or so)
        (t, phase) = RK4(deriv, phase, t, coeff)
        time_list = np.append(time_list, t)
        phase_list = np.concatenate((phase_list, np.array([phase])))

    return (time_list, phase_list)

def plot_solutions(t_lists, pos_lists, v_lists, coeff_list):

    fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    for i in range(len(coeff_list)):
        ax.plot(t_lists[i], pos_lists[i], color=cmap[i], label='$\lambda=${}'.format(coeff_list[i]),)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), )
    fig.tight_layout()
    plt.savefig('Figures/QuarticSolution_1.png', dpi=1000)

def phase_portrait(pos_lists, v_lists, coeff_list):

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(box_aspect=1),
                            constrained_layout=True)

    for i in range(len(pos_lists)):
        ax.plot(pos_lists[i], v_lists[i], color=cmap[i], label='$\lambda=${}'.format(coeff_list[i]))

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=len(coeff_list))
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    fig.tight_layout()
    plt.savefig('Figures/QuarticPhasePortrait_1.png', dpi=1000)



initial = np.array([INITIAL_POSITION, INITIAL_VELOCITY])

t_lists = []
pos_lists = []
v_lists = []

for i in range(len(coeff_list)):

    t_list, phase_list = solve_ODE(initial, START_TIME, coeff_list[i])

    [pos_list, v_list] = phase_list.transpose()

    t_lists.append(t_list)
    pos_lists.append(pos_list)
    v_lists.append(v_list)


plot_solutions(t_lists, pos_lists, v_lists, coeff_list)
# phase_portrait(pos_lists, v_lists, coeff_list)


plt.show()












