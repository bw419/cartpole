from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
from linear_model import *
from nonlinear_model import *




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



# pendulum_phase_portrait()
# pendulum_time_evolutions()
# cart_vel_induced_oscillations()
# cart_waveforms()



# ----------------------- COMPARING MODELS -------------------------------





def time_until_mismatch_plot(model_update_fn, max_it=100, oscillations=False, N_trials=500, ax=None):

	if ax is None:
		ax = plt.gca()

	match_lens = [[],[],[]]

	if oscillations:
		max_hist = max_it/4
	else:
		max_hist = max_it


	for j in range(N_trials):
		state = rand_state4(P_RANGE4)
		while get_tot_state_energy(state) > 15**2/48:
			state = rand_state4(P_RANGE4)

		n, f_state, n_c = rollout_until_mismatch(model_update_fn, state, max_it=max_it, threshold=.2, identify_n_cycles=True)
		# print(n)

		if oscillations:
			if n == max_it:
				n = max_hist
			elif np.isnan(n):
				n = 0 # max_hist
			else:
				n = n_c


		# n_c = int(n_c)
		if get_tot_state_energy(state) > 0 and get_tot_state_energy(f_state) > 0:
			match_lens[1].append(n)
		elif get_tot_state_energy(state) > 0:
			match_lens[2].append(n)
		else:
			match_lens[0].append(n)

		# if n >= 150:
		# plot_rollout_comparison(state, rollout, nonlin_rollout, 100, t_step=.2)
		# plt.plot([.2*n, .2*n], [-15, 15], "k--")
		# plt.show()

		# print(n)

	# for j, x in enumerate(match_lens):
		# print(x)
		# x[np.isnan(x)] = 30


	N_bins = 25
	hists = np.zeros((3, N_bins))

	for j in range(3):
		hists[j,:], b_e = np.histogram(match_lens[j], bins=N_bins, range=(0,max_hist))
		hists[j,:] /= len(match_lens[j])

	x = np.arange(len(hists[0,:])) * max_hist/len(hists[0,:])
	width = max_hist/len(hists[0,:])

	ax.bar(x, hists[0,:], width=width)
	ax.bar(x, hists[1,:], bottom=hists[0,:], width=width)
	ax.bar(x, hists[2,:], bottom=hists[1,:]+hists[0,:], width=width)

	# SCALE SO THAT TOTAL # RUNS ADDS TO 1?
	ax.tick_params(left=False, labelleft=False)


	# GET RID OF THIS?
	# ax.set_xlim([-1, max_hist+1])




def make_time_to_mismatch_plots():

	fig, ax = plt.subplots(2, 3)
	ax = ax.flatten()

	for i, tup in enumerate([[5, 7], [6,8], [7,9], [8,10], [9,11], [10, 12]]):
		print(i, tup)

		M_exp, N_exp = tup
		nonlin_model = get_nonlinear_fit(2**N_exp, 2**M_exp, incl_f=False)

		# time_until_mismatch_plot(to_update_fn(nonlin_model), ax=ax[i])
		time_until_mismatch_plot(to_update_fn(nonlin_model), ax=ax[i], oscillations=True, N_trials=10000)

	plt.show()


if __name__ == "__main__":


	lin_model = get_good_linear_fit()
	n_lin_model = get_good_noisy_linear_fit(0.1)
	# nonlin_model = get_good_nonlinear_fit()


	lin_rollout = generalised_rollout(lin_model)
	n_lin_rollout = generalised_rollout(lin_model)
	# nonlin_rollout = generalised_rollout(nonlin_model)




	# lin_model = get_good_linear_fit(incl_f=False)
	# nonlin_model = get_good_nonlinear_fit(incl_f=False)

	# lin_rollout = generalised_rollout(lin_model)
	# nonlin_rollout = generalised_rollout(nonlin_model)

	def comparison_plots():

		plot_rollout_comparison([0, 0, np.pi, 1], rollout, generalised_rollout(lin_model), 20, t_step=0.2)
		plot_rollout_comparison([0, 0, np.pi, 1], rollout, generalised_rollout(n_lin_model), 20, t_step=0.2)
		plt.show()
		plot_rollout_comparison([0, 0, 0.1, 0], rollout, rollout2, 20, t_step=0.2)
		plot_rollout_comparison([0, 0, 0, 1], rollout, rollout2, 20, t_step=0.2)
		plot_rollout_comparison([0, 0, 0, 3], rollout, rollout2, 20, t_step=0.2)

	comparison_plots()




	while True:

		# fig, ax = plt.subplots(2, 1)

		state = rand_state4(P_RANGE4*0.5)


		# plt.sca(ax[0])
		# plot_rollout_comparison(state, rollout, lin_rollout, 100, t_step=0.2)
		plot_rollout_comparison(state, rollout, nonlin_rollout, 100, t_step=.2)
		n, cycles = rollout_until_mismatch(to_update_fn(nonlin_model), state, max_it=100, threshold=.2, identify_n_cycles=True)
		print(n, cycles)
		plt.plot([.2*n, .2*n], [-15, 15], "k--")
		# plt.sca(ax[1])
		# rollout_scores_1(nonlin_rollout, state, max_it=10)

		plt.show()


