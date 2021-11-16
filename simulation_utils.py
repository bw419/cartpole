from globals import *
from utils import *
from cartpole import *


# Repeatedly evolves the dynamics.
def rollout(IC, N, t_step=0.2):

	sys1 = CartPole(t_step, False)

	sys1.setState(IC)

	T = np.arange(0, N)

	states = np.zeros((len(T), 4))
	states[0] = sys1.getState()

	for t in T[1:]:
		sys1.performAction(0.)
		states[t] = sys1.getState()
		# state = 
		# state[2] = remap_angle(state[2])
		# states[t] = state

	return states


def generalised_rollout(model_fn):

	update_fn = to_update_fn_w_action(model_fn)
	def fn(IC, N):
		T = np.arange(0, N)

		states = np.zeros((len(T), 4))
		states[0] = IC[:4]

		for t in T[1:]:
			state = update_fn(states[t-1])
			state[2] = remap_angle(state[2])
			states[t] = state
		return states

	return fn




def states_match_1(true_state, test_state, threshold=0.2):
	true_state1 = true_state.copy()
	true_state1[2] = remap_angle_v(true_state1[2])
	error = true_state1 - test_state

	if np.abs(error[2]) > np.pi:
		error[2] = 2*np.pi - np.abs(error[2])

	error = np.abs(error)
	weighted_error = 0.5*error/P_RANGE4

	return all(weighted_error < threshold)


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




def rollout_until_mismatch_sim_1(model_update_fn, IC, max_search_window, threshold, max_it):

	true_states = [IC]
	modelled_states = [IC]
	matches = []

	state_abs_bigger = [False, False, False, False]

	for it in range(max_it):
		# get the range of theta dot and x dot for states_match_2
		if it >= max_search_window:
			theta_dots = [x[3] for x in true_states[-max_search_window:]]
			x_dots = [x[1] for x in true_states[-max_search_window:]]
			theta_dot_range = np.abs(max(theta_dots) - min(theta_dots))
			x_dot_range = np.abs(max(x_dots) - min(x_dots))
		
			# print(theta_dot_range, x_dot_range)


		# when first reaching end of search window
		if it == max_search_window:
			for j in range(max_search_window):
				matches.append(states_match_2(true_states[j], modelled_states[j], theta_dot_range, x_dot_range, P_RANGE4[0], threshold))
			if not all(matches):
				break

		# do the actual update
		true_states.append(fast_single_action(true_states[-1]))
		modelled_states.append(model_update_fn(modelled_states[-1]))


		if it >= max_search_window:
			matches.append(states_match_2(true_states[-1], modelled_states[-1], theta_dot_range, x_dot_range, P_RANGE4[0], threshold))
			if not matches[-1]:
				break


	N_matches = np.argmin(matches)-1 # stops at first False, ignoring IC match
	if N_matches == -1:
		N_matches = max_it

	return np.array(true_states), np.array(modelled_states), N_matches


# rolls out approx to the nearest cycle
def rollout_until_mismatch_sim_2(model_update_fn, IC, threshold, max_it):

	true_states = [IC]
	modelled_states = [IC]
	matches = []

	for it in range(max_it):

		# mismatch conditions:

		if it >= 2:
			for i in range(1, 4):
				trial_extremum = true_states[-2][i]
				if np.sign(true_states[-1][i] - trial_extremum) != np.sign(trial_extremum - true_states[-3][i]):

					# local extremum!
					error = np.abs((modelled_states[-2][i] - trial_extremum)/trial_extremum)
					print("local extremum! elem", i, "fractional error", error)
					if error > threshold:
						it -= 1 # adjust back a step
						break

		x_error = np.abs((modelled_states[-1][0] - true_states[-1][0])/true_states[-1][0])
		if x_error > threshold:
			break

		# do the actual update
		true_states.append(fast_single_action(true_states[-1]))
		modelled_states.append(model_update_fn(modelled_states[-1]))

	N_matches = it

	return np.array(true_states), np.array(modelled_states), N_matches



def rollout_until_mismatch(model_update_fn, IC, threshold=0.1, max_it=50, identify_n_cycles=False):


	true_states, modelled_states, N_matches = rollout_until_mismatch_sim_1(model_update_fn, IC, 10, threshold, max_it)
	# true_states, modelled_states, N_matches = rollout_until_mismatch_sim_2(model_update_fn, IC, threshold, max_it)


	final_state = true_states[-min(len(true_states), 5)]


	# if N_matches <= 2:# or N_matches == 4:
	# 	print("true states", true_states[:,3])
	# 	print("modelled", modelled_states[:,3])
	# 	print("matches", matches)
	# 	print(N_matches)


	if identify_n_cycles:
		max_idxs = np.array(argrelextrema(np.abs(true_states[:,3]), np.greater)[0])

		if len(max_idxs) > 1:

			n_full_cycles = 0.5*(len(max_idxs)) # definition wobbly... Use the long oscillation period
			mean_cycle_length = 2.0*np.mean(max_idxs[1:] - max_idxs[:-1])
			fraction = max_idxs[0]/mean_cycle_length	
			# print(n_full_cycles, mean_cycle_length, fraction)

			return N_matches, final_state, n_full_cycles + fraction

		else:

			return N_matches, final_state, np.nan


	return N_matches, final_state








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

	standalone_fig = False
	if ax is None:
		ax = plt.gca()
		standalone_fig = True

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


# states can be a 4-vector or 5-vector
def plot_states(states, actions=None, ax=None, show_F=True, show_vars=(0,1,2,3), markers=True, line=True, standalone_fig=False):

	colours = [f"tab:{x}" for x in ["blue", "orange", "green", "red", "purple"]]

	if ax is None:
		ax = plt.gca()
		standalone_fig = True

	N = len(states)

	T = np.arange(N)*0.2

	if actions is not None:
		states = np.hstack((states, [[a] for a in actions]))

	for j in show_vars:
		if line:
			p = ax.plot(T, states[:,j], label=VAR_STR[j], c=colours[j])
		if markers:
			ax.plot(T, states[:,j], "--", lw=1, c=colours[j], marker="s", ms=3, mew=1, mec="k", mfc=colours[j])
			# ax.plot(T, states[:,j], lw=0, c=colours[j], marker="s", ms=3, mew=1, mec="k", mfc=colours[j])

	if show_F:
		ax.plot(0, np.nan, label=VAR_STR[4])

	#ax.set_title(r"Modelled trajectory"))
	ax.set_xlabel("Time (s)", labelpad=2.0)
	ax.set_ylabel("State variable value")

	if standalone_fig:
		plt.legend(loc="upper right")

	max_var = np.max(np.abs(states[:,show_vars]))
	ax.set_ylim([-max_var*1.1, max_var*1.1])

	if show_F:

		ax1 = ax.twinx()
		p = ax1.plot(T, states[:,4], label=VAR_STR[4], c=colours[4])
		if markers:
			ax1.plot(T, states[:,4], lw=0, c=colours[4], marker="s", ms=3, mew=1, mec="k", mfc=colours[4])

		max_f = np.max(np.abs(states[:,4]))
		ax1.set_ylim([-max_f*1.1, max_f*1.1])

		ax1.set_ylabel("Force value")

	if standalone_fig:
		plt.gcf().set_size_inches((4.8, 4.0))
		plt.grid(which="both", alpha=0.2)
		# plt.legend(loc="lower right")