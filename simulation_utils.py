from globals import *
from utils import *
from cartpole import *


# Repeatedly evolves the dynamics.
def rollout(IC, N, t_step=0.2):

	global sys # from cartpole.py
	assert_t_step(t_step)

	sys.setState(IC)

	T = np.arange(0, N)

	states = np.zeros((len(T), 4))
	states[0] = sys.getState()

	for t in T[1:]:
		sys.performAction(0.)
		states[t] = sys.getState()
		# state = 
		# state[2] = remap_angle(state[2])
		# states[t] = state

	return states


def generalised_rollout(update_fn):
	def fn(IC, N):
		T = np.arange(0, N)

		states = np.zeros((len(T), 4))
		states[0] = IC[:4]

		for t in T[1:]:
			state = states[t-1] + update_fn(states[t-1])
			state[2] = remap_angle(state[2])
			states[t] = state
		return states

	return fn



def rollout_scores_1(model_rollout_fn, IC, N):
	true_states = rollout(IC, N)
	modelled_states = model_rollout_fn(IC, N)

	true_states[:,2] = remap_angle_v(true_states[:,2])

	errors = true_states - modelled_states


	for i, elem in enumerate(errors[:,2]):
		if np.abs(elem) > np.pi:
			errors[i,2] = 2*np.pi - np.abs(errors[i,2])

	errors = np.abs(errors)
	weighted_errors = .5*errors/P_RANGE4[np.newaxis,:]

	plt.plot(np.arange(len(errors)), weighted_errors)

	return weighted_errors


def rollout_until_mismatch(model_update_fn, IC, threshold=0.1, max_it=50, identify_n_cycles=False):

	max_search_window = 10

	true_states = [IC]
	modelled_states = [IC]
	matches = []
	max_abs_theta_dot = 0
	for it in range(max_it):
		if it >= max_search_window:
			theta_dots = [x[3] for x in true_states[-max_search_window:]]
			x_dots = [x[1] for x in true_states[-max_search_window:]]
			theta_dot_range = np.abs(max(theta_dots) - min(theta_dots))
			x_dot_range = np.abs(max(x_dots) - min(x_dots))

			# print(theta_dot_range, x_dot_range)

		if it == max_search_window:
			for j in range(max_search_window):
				matches.append(states_match_2(true_states[j], modelled_states[j], theta_dot_range, x_dot_range, P_RANGE4[0], threshold))
			if not all(matches):
				break

		true_states.append(single_action4(true_states[-1]))
		modelled_states.append(model_update_fn(modelled_states[-1]))

		if it >= max_search_window:
			matches.append(states_match_2(true_states[-1], modelled_states[-1], theta_dot_range, x_dot_range, P_RANGE4[0], threshold))
			if not matches[-1]:
				break

	true_states = np.array(true_states)
	modelled_states = np.array(modelled_states)

	final_state = true_states[-min(len(true_states), 5)]


	# for i in range(5):
		# final_state = single_action4(final_state)


	N_matches = np.argmin(matches) # stops at first False
	if N_matches == 0:
		N_matches = max_it

	# if N_matches <= 2:# or N_matches == 4:
	# 	print("true states", true_states[:,3])
	# 	print("modelled", modelled_states[:,3])
	# 	print("max", max_abs_theta_dot)
	# 	print("rel_vals", np.abs(true_states[:,3] - modelled_states[:,3])/max_abs_theta_dot)
	# 	print("matches", matches)
	# 	print(N_matches)


	if identify_n_cycles:
		max_idxs = np.array(argrelextrema(np.abs(true_states[:,3]), np.greater)[0])
		# print("max_idxs", max_idxs)

		if len(max_idxs) > 1:

			n_full_cycles = 0.5*(len(max_idxs)) # definition wobbly... Use the long oscillation period
			mean_cycle_length = 2.0*np.mean(max_idxs[1:] - max_idxs[:-1])
			fraction = max_idxs[0]/mean_cycle_length	
			# print(n_full_cycles, mean_cycle_length, fraction)

			return N_matches, final_state, n_full_cycles + fraction

		else:

			return N_matches, final_state, np.nan


	return N_matches, final_state


def states_match_1(true_state, test_state, threshold=0.2):
	true_state1 = true_state.copy()
	true_state1[2] = remap_angle_v(true_state1[2])
	error = true_state1 - test_state

	if np.abs(error[2]) > np.pi:
		error[2] = 2*np.pi - np.abs(error[2])

	error = np.abs(error)
	weighted_error = 0.5*error/P_RANGE4

	return all(weighted_error < threshold)


def states_match_2(true_state, test_state, val_range, threshold=0.2):

	error = true_state[3] - test_state[3]
	error = np.abs(error)
	weighted_error = error/val_range

	return weighted_error < threshold


def states_match_2(true_state, test_state, theta_dot_range, x_dot_range, x_range, threshold=0.2):

	theta_dot_error = np.abs(true_state[3] - test_state[3])
	weighted_td_error = theta_dot_error/theta_dot_range

	x_dot_error = np.abs(true_state[1] - test_state[1])
	weighted_xd_error = x_dot_error/x_dot_range

	x_error = np.abs(true_state[0] - test_state[0])
	weighted_x_error = x_error/x_range

	# print(weighted_td_error, weighted_xd_error, weighted_x_error)

	criterion = all([v < threshold for v in (weighted_x_error, weighted_xd_error, weighted_td_error)])
	return criterion


# CHANGE THESE SO ROLLOUT AND MODEL WORK IN THE SAME WAY?

def plot_rollout(IC, rollout, N=200, remap=False):

	for j in range(4):
		for t_step in [0.02, 0.2]:
			N_steps = N if t_step==0.02 else int(round(0.1*N))

			T = np.arange(N_steps)*t_step

			states = rollout(IC, N_steps, t_step)

			if remap:
				remapped = remap_angle_v(states[:,2])
				for i in range(1, len(T)):
					if remapped[i] < -(np.pi-1) and remapped[i-1] > (np.pi-1):
						remapped[i] = np.nan
					elif remapped[i] > (np.pi-1) and remapped[i-1] < -(np.pi-1):
						remapped[i] = np.nan
				

			if t_step == 0.02:
				p = plt.plot(T, states[:,j], label=VAR_STR[j], lw=1.5)

			else:
				plt.plot(T, states[:,j], ls="--", lw=1, c=p[0].get_color(), marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
	

	plt.gcf().set_size_inches((4.8, 4.0))
	plt.title(r"State variable evolution for I.C. " + format_IC(IC))
	plt.xlabel("Time (s)", labelpad=2.0)
	plt.ylabel("State variable value", va="top")
	plt.grid(which="both", alpha=0.2)
	plt.legend(loc="upper right")


def plot_rollout_comparison(IC, rollout, model_fn, N=50, remap=True, t_step=0.2, ax=None):

	standalone_fig = True
	if ax is None:
		ax = plt.gca()
		standalone_fig = False

	T = np.arange(N)*t_step
	T3 = np.arange(N*10)*t_step/10
	states1 = rollout(IC, N, t_step)
	states2 = model_fn(IC, N)
	states3 = rollout(IC, N*10, t_step/10)

	for j in range(0,4):

		if remap:
			for states in [states1, states2, states3]:
				remapped = remap_angle_v(states[:,2] - np.pi) + np.pi
				# for i in range(1, len(T)):
					# if remapped[i] < -(np.pi-1) and remapped[i-1] > (np.pi-1):
						# remapped[i] = np.nan
					# elif remapped[i] > (np.pi-1) and remapped[i-1] < -(np.pi-1):
						# remapped[i] = np.nan
				states[:,2] = remapped

		p = ax.plot(T, states2[:,j], label=VAR_STR[j])
		# plt.plot(T3, states3[:,j], ls="--", lw=0.7, c=p[0].get_color())#, marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
		ax.plot(T, states1[:,j], lw=0, c=p[0].get_color(), marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
	

	ax.set_title(r"Modelled trajectory for I.C. " + format_IC(IC))
	ax.set_xlabel("Time (s)", labelpad=2.0)
	ax.set_ylabel("Variable value", va="top")

	if standalone_fig:
		plt.gcf().set_size_inches((4.8, 4.0))
		plt.grid(which="both", alpha=0.2)
		plt.legend(loc="upper right")


def plot_rollout(IC, rollout, N=200, remap=False):

	for j in range(4):
		for t_step in [0.02, 0.2]:
			N_steps = N if t_step==0.02 else int(round(0.1*N))

			T = np.arange(N_steps)*t_step

			states = rollout(IC, N_steps, t_step)

			if remap:
				remapped = remap_angle_v(states[:,2])
				for i in range(1, len(T)):
					if remapped[i] < -(np.pi-1) and remapped[i-1] > (np.pi-1):
						remapped[i] = np.nan
					elif remapped[i] > (np.pi-1) and remapped[i-1] < -(np.pi-1):
						remapped[i] = np.nan
				

			if t_step == 0.02:
				p = plt.plot(T, states[:,j], label=VAR_STR[j], lw=1.5)

			else:
				plt.plot(T, states[:,j], ls="--", lw=1, c=p[0].get_color(), marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
	

	plt.gcf().set_size_inches((4.8, 4.0))
	plt.title(r"State variable evolution for I.C. " + format_IC(IC))
	plt.xlabel("Time (s)", labelpad=2.0)
	plt.ylabel("State variable value", va="top")
	plt.grid(which="both", alpha=0.2)
	plt.legend(loc="upper right")
