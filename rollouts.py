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

	for IC_set, c in ((ICs_bound, "tab:pink"), (ICs_unbound[1:], "tab:orange"), (ICs_transition, "tab:red")):
		for IC in IC_set:
			for IC1 in [-IC, IC]:
				states = rollout(IC1, 150, 0.02)
				states[:,2][np.abs(states[:,2])>2.2*np.pi] = np.nan
				plt.plot(states[:,2], states[:,3], c=c, lw=1)	
		
	# plt.show()

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

	match_lens = [[],[]]#,[]]

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
		if get_tot_state_energy(state) > 0:# and get_tot_state_energy(f_state) > 0:
			match_lens[1].append(n)
		# elif get_tot_state_energy(state) > 0:
		# 	match_lens[2].append(n)
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


	# print(match_lens)


	N_bins = 25
	hists = np.zeros((2, N_bins))

	for j in range(len(match_lens)):
		hists[j,:], b_e = np.histogram(match_lens[j], bins=N_bins, range=(0,max_hist))
		hists[j,:] /= len(match_lens[j])

	x = np.arange(len(hists[0,:])) * max_hist/len(hists[0,:])
	width = max_hist/len(hists[0,:])

	print(hists[0,:])
	print(hists[1,:])

	ax.bar(x, hists[0,:], width=width)
	ax.bar(x, hists[1,:], bottom=hists[0,:], width=width)
	# ax.bar(x, hists[2,:], bottom=hists[1,:]+hists[0,:], width=width)

	# SCALE SO THAT TOTAL # RUNS ADDS TO 1?
	ax.tick_params(left=False, labelleft=False)

	# GET RID OF THIS?
	# ax.set_xlim([-1, max_hist+1])



def model_match_score(model_update_fn, frac=.95, thresh=.2, max_it=100, N_trials=200):

	ns = []
	for j in range(N_trials):
		state = rand_state4(P_RANGE4)
		while get_tot_state_energy(state) > 15**2/48:
			state = rand_state4(P_RANGE4)

		n, f_state, n_c = rollout_until_mismatch(model_update_fn, state, max_it=max_it, threshold=thresh, identify_n_cycles=True)
		ns.append(n+1) # for scores between 0 and 1
	

	# all counts are +1 currently
	hist, b_e = np.histogram(ns, bins=max(ns), range=(0,max(ns)))
	cumdist = np.cumsum(hist)/np.sum(hist)

	# print(hist)
	# print(cumdist)

	cumdist -= (1-frac)
	idx = np.argmax(cumdist > 0)

	if idx == 0:
		return 0.
	else:
		return round(idx-(cumdist[idx])/(cumdist[idx] - cumdist[idx-1]), 2)



def make_time_to_mismatch_plots(oscillations=False):

	fig, ax = plt.subplots(2, 1, sharex=True)
	ax = ax.flatten()
	# ax = [ax]

	# for i, M_exp in enumerate([9,11]):
		# print(i, M_exp)

		# time_until_mismatch_plot(to_update_fn(nonlin_model), ax=ax[i])
		# ax[i].set_title("$M=2^{"+ str(M_exp) +"}$")
	time_until_mismatch_plot(to_update_fn_w_action(load_model_function("nonlin_noisy_10_16_obs_0_001")), ax=ax[0], oscillations=oscillations, max_it=250, N_trials=1000)

	plt.show()


if __name__ == "__main__":


	# make_time_to_mismatch_plots(oscillations=True)


	# exit()

	lin_model = get_good_linear_fit()
	n_lin_model = get_good_noisy_linear_fit(0.1)
	# nonlin_model = get_good_nonlinear_fit(2**11)
	# nonlin_model = load_model_function("nonlin_16_12", log=False)


	lin_rollout = generalised_rollout(lin_model)
	n_lin_rollout = generalised_rollout(n_lin_model)
	# nonlin_rollout = generalised_rollout(nonlin_model)

	for i in range(5):
		print(i)
		sigma_obs_ax = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, .5]
		sigma_dyn_ax = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]

		scores = []
		for i in range(len(sigma_obs_ax)):

			# noise_str = str(sigma_obs_ax[i]).replace(".", "_")
			noise_str1 = str(sigma_dyn_ax[i]).replace(".", "_")
			# fname = f"nonlin_noisy_10_16_obs_{noise_str}"
			fname1 = f"nonlin_noisy_10_16_dyn_{noise_str1}"
			# s = model_match_score(to_update_fn_w_action(load_model_function(fname, log=False)), frac=.9, thresh=0.2, max_it=12, N_trials=500)
			s = model_match_score(to_update_fn_w_action(load_model_function(fname1, log=False)), frac=.9, thresh=0.2, max_it=12, N_trials=500)


		plt.plot(sigma_dyn_ax, scores, "kx", ls="")

	plt.title("Iterations until 10% of runs diverge vs $\sigma_{dyn}$")
	plt.xlabel("$\sigma_{dyn}$")
	plt.semilogx()
	plt.show()

	exit()

	# lin_model = get_good_linear_fit(incl_f=False)
	# nonlin_model = get_good_nonlinear_fit(incl_f=False)

	# lin_rollout = generalised_rollout(lin_model)
	# nonlin_rollout = generalised_rollout(nonlin_model)

	def comparison_plots():

		# plot_rollout_comparison([0, 0, np.pi, 1], rollout, generalised_rollout(lin_model), 20, t_step=0.2)
		# plot_rollout_comparison([0, 0, np.pi, 1], rollout, generalised_rollout(n_lin_model), 20, t_step=0.2)
		plt.show()
		plot_rollout_comparison([0, 0, 0.1, 0], rollout, rollout2, 20, t_step=0.2)
		plot_rollout_comparison([0, 0, 0, 1], rollout, rollout2, 20, t_step=0.2)
		plot_rollout_comparison([0, 0, 0, 3], rollout, rollout2, 20, t_step=0.2)

	# comparison_plots()


	# while True:
	# 	state = rand_state4(P_RANGE4*0.5)
	# 	plot_rollout_comparison(state, rollout, nonlin_rollout, 100, t_step=.2)
	# 	plt.show()


	while True:

		# fig, ax = plt.subplots(2, 1)

		state = rand_state4(P_RANGE4*1.5)


		# plt.sca(ax[0])
		# plot_rollout_comparison(state, rollout, lin_rollout, 100, t_step=0.2)
		plot_rollout_comparison(state, rollout, nonlin_rollout, 100, t_step=.2)

		n, cycles, f = rollout_until_mismatch(to_update_fn_w_action(nonlin_model), state, max_it=100, threshold=.2, identify_n_cycles=True)
		print(n, cycles)
		plt.plot([.2*n, .2*n], [-15, 15], "k--")
		# plt.sca(ax[1])
		# rollout_scores_1(nonlin_rollout, state, max_it=10)

		plt.show()


