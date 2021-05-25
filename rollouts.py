from globals import *
from utils import *
from rollout_utils import *
from cartpole import *


# ----------------------- SIMPLE BEHAVIOUR OF SYSTEM -------------------------------

pole_vels_bound = [3, 7, 11, 13.7]
pole_vels_unbound = [14, 14.3, 15, 18]
pole_vels_transition = [13.8, 14]
ICs_bound = [np.array([0, 0, -np.pi, x]) for x in pole_vels_bound]
ICs_unbound = [np.array([0, 0, -np.pi*3, x]) for x in pole_vels_unbound]
ICs_transition = [np.array([0, 0, -np.pi*3, x]) for x in pole_vels_transition]

def pendulum_phase_portrait():

	for IC_set, c in ((ICs_bound, "tab:blue"), (ICs_unbound[1:], "tab:green"), (ICs_transition, "tab:orange")):
		for IC in IC_set:
			for IC1 in [-IC, IC]:
				states = rollout(IC1, 150, 0.02)
				states[:,2][np.abs(states[:,2])>2.2*np.pi] = np.nan
				plt.plot(states[:,2], states[:,3], c=c, lw=1)	
		
	plt.show()

def pendulum_time_evolutions():

	for IC in [ICs_bound[0], ICs_bound[-1], ICs_unbound[1], ICs_transition[0]]:
		plot_rollout(IC, rollout, 150)

def cart_vel_induced_oscillations():


	# ICs = [[0, 0, np.pi, 5], [0, 0, np.pi, 12.2], [0, 0, np.pi, 12.3], [0, 0, np.pi, 15]]
	ICs = [[0, x, -np.pi, 0] for x in [1, 2, 4, 8, 10, 50]]
	ICs2 = [[0, x, -np.pi+0.0001, 0] for x in [1, 2, 4, 8, 10, 50]]
	ICs.extend(ICs2)

	for IC in ICs:
		states = rollout(IC, 100, 0.02)
		plt.plot(states[:,2], states[:,3], lw=1)
	
	plt.show()


def cart_waveforms():

	for IC_set, c in ((ICs_bound, "r"), (ICs_unbound, "b")):
		for IC in IC_set:
			states = rollout(-IC, 150, 0.02)
			local_minima = argrelextrema(states[:,1], np.less)[0]
			minima = [idx for idx in local_minima if states[idx,1] < 0.1]

			if len(minima) > 0:
				first_min = minima[0]
			else:
				first_min = len(states)

			single_wave = states[:first_min, 1]
			plt.plot(np.linspace(0, 1, len(single_wave)), single_wave, c, lw=1)


	plt.show()



pendulum_phase_portrait()
pendulum_time_evolutions()
cart_vel_induced_oscillations()
cart_waveforms()



# ----------------------- COMPARING MODELS -------------------------------



# from main import *
# C = np.load("../lin_model.npy")
# print(C)
# exit()
# C = np.load("../lin_model1.npy")





from linear_model import get_good_linear_fit
from nonlinear_model import get_good_nonlinear_fit

def rollout1(IC, N):

	T = np.arange(0, N)

	states = np.zeros((len(T), 4))
	states[0] = IC
	# state = IC

	lin_model = get_good_linear_fit()

	for t in T[1:]:
		state = states[t-1] + lin_model(states[t-1])
		state[2] = remap_angle(state[2])
		states[t] = state


	return states

def rollout2(IC, N):

	T = np.arange(0, N)

	states = np.zeros((len(T), 4))
	states[0] = IC
	# state = IC

	nonlin_model = get_good_nonlinear_fit()

	for t in T[1:]:
		state = states[t-1] + nonlin_model(states[t-1])
		state[2] = remap_angle(state[2])
		states[t] = state


	return states


# while True:
# 	plot_rollout_comparison(rand_state(), rollout, rollout1, 20, t_step=0.2)


plot_rollout_comparison([0, 0, np.pi, 1], rollout, rollout2, 20, t_step=0.2)
plot_rollout_comparison([0, 0, 0.1, 0], rollout, rollout2, 20, t_step=0.2)
plot_rollout_comparison([0, 0, 0, 1], rollout, rollout2, 20, t_step=0.2)
plot_rollout_comparison([0, 0, 0, 3], rollout, rollout2, 20, t_step=0.2)

