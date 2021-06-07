from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
from nonlinear_model import *
import collections
from time import perf_counter


single_action = fast_single_action

# model_fn = load_model_function("nonlin_13_11")
# model_fn = load_model_function("nonlin_16_12")
# learned_update_fn = to_update_fn_w_action(model_fn)


P = np.array([0, 0, 0, 0])
THETA_ONLY_P = np.array([0, 0, 20, 2.1]) #theta can range from 13-23ish, theta dot more fixed
X_ONLY_P = np.array([-10, -5.5, 0, 0])
THETA_XDOT_ONLY_P = np.array([0, 1.1, 15, 2.5])

## LINEAR CONTROLLER ################################################
## NOISE FREE #######################################################
## OPTIMISED ON TRUE DYNAMICS #######################################
## optimisation results... ##########################################
# initial manual pick...
# [.3, .6, 15, 2.5]
# using bounds of [[0, 1], [0, 2], [10, 20], [1,3]]...
# [1, 1, 17, 2.5]
# [.5, 1.8, 16, 3.]
# [.25, .48, 15.4, 2.0]
# [.75, 1., 17.5, 2.5]
# [.5, 1., 18.3, 2.7]
# [.486, 1.05, 17.6, 2.71] - best, Ns=11
# next... make N runs=50, N_its=100

# thresholds, starting at [.0, .0, .0, .0] to obtain 90% success:
# [.4, .3, 18, 1.95] is at the min for each!
# max_vals  ~ [.4, .8, .3, .5]xP_RANGE
# 100% success rate over 500 runs & 200 iterations for 32% of max_vals

GOOD_LOSS_SCALES = [1, 3.3, .3, 5]
# GOOD_LOSS_SCALES = [5, 3.3, .3, 5]
GOOD_P = np.array([.4, .3, 18, 1.95]) 
GOOD_P_001 = np.array([ 0.49200333,  0.65492707, 13.85773113,  2.11572089]) 
BEST_P = np.array([0.06746362, 0.54062834, 15.14143545, 2.16702088])

#####################################################################

# TODO: 
# - optimise on modelled dynamics
# - optimise for observation noise
# - optimise for dynamics noise
# - optimise for both?



def get_loss_fn(loss_sig=None, range_prop=None):
	if loss_sig is None and range_prop is None:
		loss_sig = GOOD_LOSS_SCALES

	if range_prop is not None:
		loss_sig = range_prop * P_RANGE4

	return lambda x: 1-np.exp(-(
		np.square((x[0])*INV_SQRT2/loss_sig[0]) +
		np.square((x[1])*INV_SQRT2/loss_sig[1]) +
		np.square(np.sin(.5*x[2])*INV_SQRT2/loss_sig[2]) +
		np.square((x[3])*INV_SQRT2/loss_sig[3])
	))



def speed_comparison():
	sigmas = np.array([.2, .3, 3, .2])
	loss_fn3 = lambda x: 1-np.exp(-(
			np.square((x[0])*INV_SQRT2/loss_sig[0]) +
			np.square((x[1])*INV_SQRT2/loss_sig[1]) +
			np.square(np.sin(.5*x[2])*INV_SQRT2/loss_sig[2]) +
			np.square((x[3])*INV_SQRT2/loss_sig[3])
		))
	loss_fn1 = lambda x: 1-np.exp(-(
			np.square((x[0])*INV_SQRT2*.2) +
			np.square((x[1])*INV_SQRT2*.3) +
			np.square(np.sin(.5*x[2])*INV_SQRT2*3) +
			np.square((x[3])*INV_SQRT2*.2)))

	t = time.perf_counter()
	for i in range(1000):
		loss_fn1(np.array([1, 1, 1, 1]))
	print(time.perf_counter() - t)

	t = time.perf_counter()
	for i in range(1000):
		loss_fn2(np.array([1, 1, 1, 1]))
	print(time.perf_counter() - t)

	t = time.perf_counter()
	for i in range(1000):
		loss_fn3(np.array([1, 1, 1, 1]))
	print(time.perf_counter() - t)


###########################################################################
# some formulations of the nonlinear policy for testing.
def policy_prototypes():

	def nonlinear_matrix_param_policy(x, X, W, w):
		x1 = (x - X)
		return np.dot(w, np.exp(-0.5 * np.sum(x1 * (W @ x1[:, :, np.newaxis])[:,:,0], axis=1)))

	def nonlinear_matrix_sum_policy(x, X, W_diag, W_UR, w):
		# print(np.transpose(W_UR) + W_UR + W_diag)
		x1 = (x - X)
		return np.dot(w, np.exp(-0.5 * np.sum(x1 * (W @ x1[:, :, np.newaxis])[:,:,0], axis=1)))

	# X = basis_fn_centres = 4*N_basis_function
	# w_elems = 10 + N_basis_function
	def nonlinear_W_elems_policy(x, basis_fn_centres, W_elems, w):
		W = [[W_elems[0], W_elems[4], W_elems[7], W_elems[9]],
			 [W_elems[4], W_elems[1], W_elems[5], W_elems[8]],
			 [W_elems[7], W_elems[5], W_elems[2], W_elems[6]],
			 [W_elems[9], W_elems[8], W_elems[6], W_elems[3]]]
		x1 = (x - basis_fn_centres)
		return np.dot(w, np.exp(-0.5 * np.sum(x1 * (W @ x1[:, :, np.newaxis])[:,:,0], axis=1)))


	# X = basis_fn_centres = 4*N_basis_function
	# w_elems = 10 + N_basis_function
	def nonlinear_w_elems_policy(x, basis_fn_centres, w_elems):
		W = [[w_elems[0], w_elems[4], w_elems[7], w_elems[9]],
			 [w_elems[4], w_elems[1], w_elems[5], w_elems[8]],
			 [w_elems[7], w_elems[5], w_elems[2], w_elems[6]],
			 [w_elems[9], w_elems[8], w_elems[6], w_elems[3]]]
		w = w_elems[10:]
		x1 = (x - basis_fn_centres)
		return np.dot(w, np.exp(-0.5 * np.sum(x1 * (W @ x1[:, :, np.newaxis])[:,:,0], axis=1)))
###########################################################################


# create a new policy function given a set of parameters.
# using this formulation, the parameters are built in to the function
# and don't need to be inefficiently passed to it every time.
def get_policy_fn(param_object, which="linear", obs_noise_f=None, log=True):

	if which == "linear":
		# parameters are a simple 4-vector of linear coefficients
		P = param_object

		if obs_noise_f is None:
			def p(x, it=None):
				return np.dot(P, remapped_angle(x))
		else:
			def p(x, it=None):
				return corrupt_force(np.dot(P, remapped_angle(corrupt_msmt(x, obs_noise_f))), obs_noise_f)

	# this doesn't work at all.
	elif which == "linear_decreasing":
		P = param_object

		if obs_noise_f is None:
			def p(x, it=None):
				return np.dot(P, remapped_angle(x))
		else:
			def p(x, it=0):
				return corrupt_force(np.dot(P, remapped_angle(corrupt_msmt(x, obs_noise_f)))*np.exp(-(it/50)**2), obs_noise_f) 


	elif which == "nonlinear_1":
		# parameters are more complex. W_elems has 10 elements, w_vector and basis_fn_centres have same number of elements.
		basis_fn_centres, W_elems, w_vector = param_object

		W_matrix = [[W_elems[0], W_elems[4], W_elems[7], W_elems[9]],
					[W_elems[4], W_elems[1], W_elems[5], W_elems[8]],
					[W_elems[7], W_elems[5], W_elems[2], W_elems[6]],
					[W_elems[9], W_elems[8], W_elems[6], W_elems[3]]]
		W_matrix = np.array([np.array(x) for x in W_matrix])

		NONLIN_TEMP = np.zeros((len(w_vector), 4))

		def p(x, it=None):
			NONLIN_TEMP[:,:] = x - basis_fn_centres 
			return np.dot(w_vector, np.exp(-0.5 * np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1)))

	else:
		raise("policy type does not exist")

	return p



N_req_within_threshold_noisy = 25
default_gains = np.array([2.2,7.4,4.1,8.7])

def policy_simulation(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True, ret_success=True, obs_noise_f=None, gains=None):

	if obs_noise_f is not None:
		if gains is None:
			gains = default_gains

		def success_criterion(states, it):
			# global N_req_within_threshold_noisy
			# print("converged yet?", np.mean(np.std(states[-N_req_within_threshold_noisy:], axis=0)/(P_RANGE4*gains)) )
			return it >= N_req_within_threshold_noisy and np.mean(np.std(states[-N_req_within_threshold_noisy:], axis=0)/(P_RANGE4*gains)) < 1.5*obs_noise_f
	else:
		def success_criterion(states, it):
			return it >= 10 and np.mean(states[-10:]) < 1e-8


	states = [IC]
	actions = [policy_fn(IC)]
	losses = [loss_fn(states[-1])]
	prev_loss = np.inf
	# states_int = np.zeros(4)

	for it in range(max_it):
		# states.append(update_fn([states[-1][0], states[-1][1], states[-1][2], states[-1][3], actions[-1]]))
		states.append(update_fn(states[-1], actions[-1]))
		# xpos_int[0] += 0.01*states[-1][0]
		# states_int += 0.01*states[-1]
		actions.append(policy_fn(states[-1], it))# + states_int))
		losses.append(loss_fn(states[-1]))
		# if it > 15 and np.max(losses[-15:]) < 0.01:
			# print(states, actions, losses[-10:])
			# return states, actions, np.mean(losses)


		if it % 5 == 0 and stop_early:
			if ret_success:
				if np.abs(states[-1][0]) > 3*P_RANGE4[0]:
					return states, actions, 0.
				if success_criterion(states, it):
					return states, actions, 1.
			else:
				if obs_noise_f is not None:
					raise Exception("this criterion isn't well-defined for noise")
				if np.abs(states[-1][0]) > 3*P_RANGE4[0]:
					return states, actions, 1.0
				if np.abs(np.sum(losses) - prev_loss) < 1e-8:
					return states, actions, np.sum(losses)/max_it
				prev_loss = np.sum(losses)

	if ret_success:
		if success_criterion(states, it):
			return states, actions, 1.
		else:
			return states, actions, 0.
	else:
		return states, actions, np.mean(losses)




def policy_loss(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True, log=False):

	state = IC
	L = 0
	prev_loss = np.inf
	for it in range(max_it):

		L += loss_fn(state)
		action = policy_fn(state, it)
		state = update_fn(state, action) # this takes the majority of the computation time

		# test to terminate early
		if it % 5 == 0 and stop_early: 
			if np.abs(state[0]) > 3*P_RANGE4[0]:
				to_ret = 1.0
				break
			# if np.abs(L - prev_loss) < 1e-8:
			# 	print("stopping early")
			# 	to_ret = L/max_it
			# 	break
			prev_loss = L

	to_ret = L/max_it

	if log:
		return np.log(to_ret)
	else:
		return to_ret





# so that when there are mulitple runs, each has a different IC.
def get_IC_gen_fn(IC_proportions, same_seed=False, add_pi=False):
	to_add = [0, 0, np.pi, 0] if add_pi else [0, 0, 0, 0]

	def gen_IC():
		return IC_proportions*rand_state4() + to_add

	# if same_seed:
	# 	def gen_IC():
	# 		return IC_proportions*rand_state4() + to_add
	# else:
	# 	def gen_IC():
	# 		return IC_proportions*rand_state4() + to_add

	return gen_IC



def policy_loss_N_runs(IC_gen_fn, loss_fn, update_fn, policy_fn, N_its, N_runs, log=True, seed=None):
	if seed is not None:
		np.random.seed(seed)
	return sum([
		policy_loss(IC_gen_fn(), loss_fn, update_fn, policy_fn, N_its, log=log)
		for i in range(N_runs)
		])/(N_runs)



def policy_success(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True, obs_noise_f=None):
	return policy_simulation(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=stop_early, ret_success=True, obs_noise_f=obs_noise_f)[2]


def policy_success_rate(IC_gen_fn, loss_fn, update_fn, policy_fn, N_its, N_runs, obs_noise_f=None, gains=None):
	return np.count_nonzero([
		policy_simulation(IC_gen_fn(), loss_fn, update_fn, policy_fn, N_its, stop_early=True, ret_success=True, obs_noise_f=obs_noise_f, gains=gains)[2]
		for i in range(N_runs)
		])/(N_runs)


# obtain a function mapping policy parameters to a loss value
def get_policy_loss_fn(IC_gen_fn, loss_sig, model_fn, N_its, N_runs, which="linear", log=True, seed=False, obs_noise_f=None):
	if seed:
		seed = np.random.randint(0, 10000000)
	else:
		seed = None
	if which == "linear":
		def loss(P):
			return policy_loss_N_runs(IC_gen_fn, get_loss_fn(loss_sig), model_fn, 
										get_policy_fn(P, which, obs_noise_f=obs_noise_f),
										N_its, N_runs, log=log, seed=seed)
	elif which == "nonlinear_1":
		def loss(centres, W_elems, w):
			return policy_loss_N_runs(IC_gen_fn, get_loss_fn(loss_sig), model_fn, 
										get_policy_fn((centres, W_elems, w), which, obs_noise_f=obs_noise_f),
										N_its, N_runs, log=log, seed=seed)
	return loss






def probe_stability_region(P, dynamics_fn, success_threshold=1.0, N_axis=101, plot=True, delay_plot=False, obs_noise_f=None, stop_early=False, N_its=50, N_runs=50, gains=None):

	print("probing stability...\r", end="")
	axis_vals = np.linspace(0., 1., N_axis)
	success_rates = np.zeros((5, N_axis))
	thresholds = np.zeros(5)

	for idx in range(4):
		for i, IC_frac in enumerate(axis_vals):
			IC_proportions = np.zeros(4)
			IC_proportions[idx] = IC_frac
			# print(IC_proportions)
			success_rates[idx, i] = policy_success_rate(get_IC_gen_fn(IC_proportions), get_loss_fn(), dynamics_fn, 
														get_policy_fn(P, which="linear", obs_noise_f=obs_noise_f), 
														N_its, N_runs, obs_noise_f=obs_noise_f, gains=gains)
			if success_rates[idx, i] < success_threshold:
				break

		if plot:
			plt.plot(axis_vals, success_rates[idx,:])

	failures = success_rates[:4, :] < success_threshold

	thresholds[:4] = axis_vals[np.argmax(failures, axis=1)]

	for i in range(4):
		if thresholds[i] == 0 and success_rates[i][-1] == 1.0:
			thresholds[i] = 1.0


	if np.all(thresholds[:4] == 0.):
		print("all zeros.                   \r", end="")
		return thresholds

	print("finished single vars...\r", end="")

	# print(thresholds)
	# axis_vals = np.linspace(0., .5, N_axis)
	for i, frac in enumerate(axis_vals):
		success_rates[4, i] =  policy_success_rate(get_IC_gen_fn(thresholds[:4]*frac), get_loss_fn(), dynamics_fn, 
												   get_policy_fn(P, which="linear", obs_noise_f=obs_noise_f),
												   N_its, N_runs, obs_noise_f=obs_noise_f)
		if success_rates[4, i] < success_threshold:
			break
	
	# print(success_rates[4])

	thresholds[4] = axis_vals[np.argmax(success_rates[4,:] < success_threshold)]

	if thresholds[4] == 0 and success_rates[4][-1] == 1.0:
		thresholds[4] = 1.0

	if plot:
		# print(thresholds)

		if delay_plot:
			plt.figure()

		plt.plot(axis_vals, success_rates[4,:])

		if not delay_plot:
			plt.show()

	print("finished.                  \r")


	return thresholds#, np.prod(thresholds)




def optimise_linear_policy(IC_proportions, start_pos, dynamics_fn, N_its=20, N_runs=50, log=False, IC_gen_fn=None, obs_noise_f=None):

	if IC_gen_fn is None:
		IC_gen_fn = get_IC_gen_fn(IC_proportions)


	loss_fn = get_policy_loss_fn(IC_gen_fn, GOOD_LOSS_SCALES, dynamics_fn, N_its, N_runs, seed=True, obs_noise_f=obs_noise_f)



	if log:
		print("initial loss:", loss_fn(start_pos), start_pos)

	def callback(xk):
		print("optimising...",  xk, "\r", end="")
		# print("doing stuff", loss_fn(xk), xk, "\r", end="")
		return False

	# loss_fn1 = lambda x: loss_fn(x) + (1-np.exp(-x[3]**2))#np.sum(np.square(x))))

	res = scipy.optimize.minimize(loss_fn, start_pos, method="Nelder-Mead", options={}, callback=callback)
	print("finished optimising.                                                                          \r", end="")

	# print("final loss:", res.fun, res.x)
	return res.x, res.fun


##########################################################
##########################################################
##############       OPTIMISATION       ##################
##########################################################
##########################################################
##########################################################
def multi_IC_optimise(dynamics_fn, fname, rand=False):

	P = GOOD_P
	loss = -15
	IC_frac = 2**(-7.5)
	history = (P, loss, IC_frac)#, np.nan)
	for it in range(21):

		# TEMPORARY
		# P = GOOD_P
		if rand:

			# if it >= 5:
			# 	continue 

			trials = []
			for i in range(25):
				P = 20*(np.random.random(4)*2-1)

				print(f"it={it}.{i}, L={loss}, f={IC_frac}, P={P}") 
				P, loss = optimise_linear_policy(IC_frac, P, dynamics_fn, N_runs=100) # 20-200ish
				trials.append(np.array([P, loss, IC_frac]))

			trials = np.array(trials)
			best_trial = np.argmin(trials[:,1])
			
			history = np.vstack((history, tuple(trials[best_trial])))

			IC_frac *= np.sqrt(2)


		else:
				print(f"it={it}, L={loss}, f={IC_frac}, P={P}") 
				P, loss = optimise_linear_policy(IC_frac, P, dynamics_fn, N_runs=25) # 20-200ish
				data = (P, loss, IC_frac)
				history = np.vstack((history, data))



		if False:# it == 0 or IC_frac >= 0.6: 
			policy_loss_fn = get_policy_loss_fn(get_IC_gen_fn(IC_frac), GOOD_LOSS_SCALES, dynamics_fn, 20, 50, which="linear")
		
			print(f"doing six planes plot. it={it}, L={loss}, f={IC_frac}") 
			# six_planes(policy_loss_fn, start_state=start_state, bounds=[25, 25, 25, 25], NX=40, NY=40, incl_f=False, pi_multiples=False, levels=levels)
			six_planes(policy_loss_fn, start_state=P, varying_bounds=True, bounds=[[-10, -6, 0, -1], [10, 6, 30, 5]], NX=20, NY=20, incl_f=False, pi_multiples=False, levels=np.linspace(loss-.5, 0, 10))
		
			IC_frac *= np.sqrt(2)

	plt.show()


	# turns out the most robust solution is the one optimised for nearby ICs!
	thresholds = []
	for row in history:
		thresholds.append(probe_stability_region(row[0], dynamics_fn, plot=False, N_its=50, N_runs=50))
		# print(thresholds[-1])



	print("-------------------------------")

	history = np.hstack((history, thresholds))

	save_data(history, fname)

	for row in history:
		print(*[item for item in row])

	print("-------------------------------")



# def random_optimise_success_rate():



def get_robust_policy(dynamics_fn, N_runs=100, second_scale=2**-4, obs_noise_f=None, N_stability_runs=150, get_robustness=False):

	P, loss = optimise_linear_policy(2**-10, GOOD_P, dynamics_fn, N_runs=N_runs, obs_noise_f=obs_noise_f)
	# P, loss = optimise_linear_policy(2**-7, P, dynamics_fn, N_runs=100)
	P, loss = optimise_linear_policy(second_scale, P, dynamics_fn, N_runs=N_runs, obs_noise_f=obs_noise_f)

	if not get_robustness:
		print(P, loss)
		return P, loss


	if obs_noise_f is not None:
		gains = get_noise_amplification(Pa, np.zeros(4), noise_f)

		stability_thresholds = probe_stability_region(P, single_action, plot=False, obs_noise_f=obs_noise_f, N_its=50, N_runs=N_stability_runs, gains=gains)

		print(P, loss, gains, stability_thresholds, np.prod(stability_thresholds))
		return P, loss, gains, stability_thresholds

	else:
		stability_thresholds = probe_stability_region(P, single_action, plot=False, N_its=50, N_runs=N_stability_runs)

		print(P, loss, stability_thresholds, np.prod(stability_thresholds))
		return P, loss, stability_thresholds



def linear_policy_contour_plots(model_fn, P=np.zeros(4), IC_proportions=.01, loss_scales=GOOD_LOSS_SCALES, N_runs=50, N_its=20, levels=np.linspace(-15, 0, 10), filled=True, cmap="viridis", axs=None, N=15, obs_noise_f=None):
	
	# plt.title(f"x = {round(s, 1)}")

	NX = N
	NY = N

	IC_gen_fn = get_IC_gen_fn(IC_proportions)

	if True:
		start_state = P.copy()
		if axs is None:
			fig, axs = plt.subplots(2,2)
			axs = axs.flatten()
			plt.suptitle(f"Loss surfaces, initial policy, IC props.: {IC_proportions}")


		if True:
			print("starting plot")
			loss_scales1 = loss_scales.copy()
			loss_scales1[0] = 1e6
			loss_scales1[0] = 1e6

			# plt.figure()

			# OPTIMISING FOR ONLY THETA, THETA DOT: (ENSURE loss_fn(P) includes x,x dot part, and i.c. range is sensible)
			policy_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales1, single_action, N_its, N_runs, which="linear", obs_noise_f=obs_noise_f)
			contour_plot(policy_loss_fn, start_state=start_state, bounds=[[0, 0, -30, -10], [0, 0, 30, 10]], xi=2, yi=3, NX=N, NY=N, 
						incl_f=False, pi_multiples=False, levels=levels, ax=axs[0], cmap=cmap, filled=filled, cb=False)
			# plt.show()


		if True:
			axs = axs.flatten()

			print("starting plot")
			loss_scales1 = loss_scales.copy()
			loss_scales1[0] = 1e6

			# theta dot, x dot plane with theta dot = 2.12
			if True:

				start_state = P.copy()

				for i, s in enumerate([2.12]):#enumerate(np.linspace(1, 4, 5)):

					cbar = i == len(axs)-1

					# plt.figure()
					start_state[3] = s
				
					policy_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales1, single_action, N_its, N_runs, which="linear", obs_noise_f=obs_noise_f)		
					contour_plot(policy_loss_fn, start_state=start_state, bounds=[[0, -5, 0, 0], [0, 5, 30, 0]], xi=1, yi=2, NX=N, NY=N,
								incl_f=False, pi_multiples=False, levels=levels, ax=axs[1], cmap=cmap, filled=filled, cb=cbar)
			
				# plt.show()

			# scan of theta, theta dot planes with theta = 14
			if True:
				start_state = P.copy()
				loss_scales1 = loss_scales.copy()

				for s in [14]:#np.linspace(0, 30, 10):
					start_state[2] = s
					# plt.figure()

					# OPTIMISING FOR FIRST 3:
					policy_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales1, single_action, N_its, N_runs, which="linear", obs_noise_f=obs_noise_f)		
					contour_plot(policy_loss_fn, start_state=start_state, bounds=[[0, -1.0, 0, .7], [0, 2.5, 0, 4.3]], xi=1, yi=3, NX=N, NY=N,
								incl_f=False, pi_multiples=False, levels=levels, ax=axs[2], cmap=cmap, filled=filled, cb=False)

					x = np.array([-1., 2.5])
					axs[2].plot(x, 1.85 + .5*x, "k--")
				
				# plt.show()


		if True:
			print("starting plot")
			start_state = P.copy()

			for s in [.5]:#np.linspace(-2, 3, 10):
				start_state[0] = s
				# plt.figure()
				# plt.title(s)
				print(s)
				## OPTIMISING FOR ALL 4:
				policy_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales, single_action, N_its, N_runs, which="linear", obs_noise_f=obs_noise_f)
				def modified_loss_fn(P):
					P[3] = 1.85 + 0.5*P[1]
					return policy_loss_fn(P)
				contour_plot(modified_loss_fn, start_state=start_state, bounds=[[-2, 0, 0, 0], [3, 0, 30, 0]], xi=0, yi=2, NX=N, NY=N,
							 incl_f=False, pi_multiples=False, levels=levels, ax=axs[3], cmap=cmap, filled=filled, cb=False)
			
			# plt.show()
		
		mappable = plt.cm.ScalarMappable(norm=Normalize(levels[0], 0), cmap=cmap)
		plt.colorbar(mappable, ax=axs)

		return axs
		plt.show()


	if False:
		# Now, do contour plots about this global min.
		start_state = P.copy()
		policy_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales, single_action, N_its, N_runs, which="linear", obs_noise_f=obs_noise_f)

		# six_planes(policy_loss_fn, start_state=start_state, bounds=[25, 25, 25, 25], NX=40, NY=40, incl_f=False, pi_multiples=False, levels=levels)
		six_planes(policy_loss_fn, start_state=start_state, varying_bounds=True, bounds=[[-10, -2, 5, 0], [5, 2, 25, 3]],
		 NX=30, NY=30, incl_f=False, pi_multiples=False, levels=levels)
		plt.show()




def policy_set_robustness_plot(data, re_probe=True):

	# x = 2**(0.5*np.arange(len(data))-10)
	x = 2**(5 + np.arange(len(data)))
	y = []
	z = []
	for row in data:
		print(row)
		if re_probe:
			y.append(probe_stability_region(row[0], single_action, N_its=50, N_runs=20, plot=False))
		else:
			y.append(row[3:8])
		z.append(np.prod(y[-1]))

	plt.plot([x[0], x[-1]], [.2, .2], "--", c="tab:blue")
	plt.plot([x[0], x[-1]], [.2, .2], "--", c="tab:green")
	plt.plot([x[0], x[-1]], [.24, .24], "--", c="tab:orange")
	plt.plot([x[0], x[-1]], [.34, .34], "--", c="tab:red")
	plt.plot([x[0], x[-1]], [.41, .41], "--", c="tab:purple")

	# plt.title("Success rate vs tested fraction of initial condition range")
	# plt.xlabel("$M$")
	plt.title("Success rate vs M, $f_{IC}=2^{-4}$")
	plt.xlabel("$M$")

	plt.semilogx()
	plt.ylabel("Success rate ($f^1$ to $f'$)")


	ls = plt.plot(x, y)
	plt.twinx()
	plt.semilogx()
	plt.plot([x[0], x[-1]], [.00155, .00155], "k--")

	plt.ylabel("Success rate ($f_{tot}$)")
	ls.append(plt.plot(x, z, "k")[0])
	plt.legend(ls, ["$f^1$", "$f^2$", "$f^3$", "$f^4$", "$f'$", "$f_{tot}$"], loc="upper left")

	plt.show()



def quick_linear_policy_sim(P, IC, loss_sig, N_its=50, markers=False, obs_noise_f=None, which="linear", title="Control simulation"):
	fig, ax = plt.subplots(1, 1)

	loss_fn = get_loss_fn(loss_sig)
	policy_fn = get_policy_fn(P, which=which, obs_noise_f=obs_noise_f)

	states, actions, L = policy_simulation(IC, loss_fn, single_action, policy_fn, N_its, ret_success=True, stop_early=True, obs_noise_f=obs_noise_f)
	plot_states(states, actions, ax=ax, show_F=True, markers=markers, standalone_fig=True)

	y= 1e-8 if obs_noise_f is None else 10*obs_noise_f*np.mean(P_RANGE4)
	
	plt.title(title)
	print(L, len(states), P)
	# states, actions, L = policy_simulation(IC, loss_fn, learned_update_fn, policy_fn, N_its, stop_early=True)
	# plot_states(states, actions, ax=ax[1], show_F=False, markers=markers)
	plt.show()


def get_noise_amplification(P, IC, obs_noise_f, which="linear", model_fn=single_action):

	global N_req_within_threshold_noisy

	prev_val = N_req_within_threshold_noisy

	N_req_within_threshold_noisy = 2000
	N_its = N_req_within_threshold_noisy + 50

	loss_fn = get_loss_fn(GOOD_LOSS_SCALES)
	policy_fn = get_policy_fn(P, which=which, obs_noise_f=obs_noise_f)
	states, actions, success = policy_simulation(IC, loss_fn, model_fn, policy_fn, N_its, ret_success=True, stop_early=True, obs_noise_f=obs_noise_f)
	
	N_req_within_threshold_noisy = prev_val

	if success:
		out_noise = np.std(states[-450:], axis=0)/P_RANGE4

		print("amplification:", out_noise/obs_noise_f)

		return out_noise/obs_noise_f
	else:
		print("simulation diverged.")
		return None

def rand_quick_runs(P=GOOD_P, IC_props=[.1,.1,.1,.1], **kwargs):
	while True:
		quick_linear_policy_sim(P, get_IC_gen_fn(IC_props)(), GOOD_LOSS_SCALES, 20, **kwargs)


def quick_success_rate(P, IC_props=[.1,.1,.1,.1], obs_noise_f=None):
	loss_fn = get_loss_fn(GOOD_LOSS_SCALES)
	policy_fn = get_policy_fn(P, which="linear")
	s = policy_success_rate(get_IC_gen_fn(IC_props), loss_fn, single_action, policy_fn, 50, 500, obs_noise_f=obs_noise_f)


BEST_FIC = 2**-4
BEST_P = load_data("lin_policy_ps_6.txt")[13][0]
print(BEST_FIC, BEST_P)

if __name__ == "__main__":

	# results = []
	# for M in range(5, 14):
	# 	model_fn = to_update_fn_w_action(get_optimal_nonlin_fit(M))
	# 	results.append(get_robust_policy(model_fn))

	# save_data(results, "model_optima")
	# exit()

	# for row in load_data("lin_policy_ps_6.txt"):
	# 	t = probe_stability_region(row[0], single_action, N_its=50, N_runs=20, plot=False)
	# 	print("\r", *row[0], end=" ")
	# 	print(row[1], end=" ")
	# 	print(*t)

	# exit()

	# data = load_data("model_optima")
	# # print(data)
	# row = data[3]
	# # policy_set_robustness_plot(data, re_probe=False)
	# rand_quick_runs(P=row0], IC_props=0.1, markers=True)
	# rand_quick_runs(P=row[0], IC_props=np.array(row[3:7])*row[7], markers=True)
	# rand_quick_runs(P=BEST_P, IC_props=.2, markers=True)



	# multi_IC_optimise(single_action, "lin_policy_ps_random2.txt", rand=True)
	# policy_set_robustness_plot(load_data("lin_policy_ps_random2.txt")[1:], re_probe=False)


	def plot_targets(ax, xs, tot=False):
		if tot:
			ax.plot([xs[0], xs[-1]], [.00155, .00155], "k--")
		else:
			ax.plot([xs[0], xs[-1]], [.22, .22], "--", c="tab:blue")
			ax.plot([xs[0], xs[-1]], [.21, .21], "--", c="tab:green")
			ax.plot([xs[0], xs[-1]], [.26, .26], "--", c="tab:orange")
			ax.plot([xs[0], xs[-1]], [.34, .34], "--", c="tab:red")
			ax.plot([xs[0], xs[-1]], [.34, .34], "--", c="tab:red")

	def noisy_old_policy():

		xs = np.array(np.linspace(-9, -6, 50))
		# xs = np.array(np.linspace(-15, -5, 5))
		ys = []
		amps = []
		for exp in []:#xs:#[]:#xs:
			# probe_stability_region(BEST_P, single_action, N_its=50, plot=True, obs_noise_f=2**exp, delay_plot=True)
			# plt.title("$2^{" + str(exp) + "}$ or " + str(2**exp))

			# amps.append(get_noise_amplification(BEST_P, get_IC_gen_fn(BEST_FIC)(), 2**exp, which="linear"))
		
			# while True:
			# 	quick_linear_policy_sim(BEST_P, get_IC_gen_fn(BEST_FIC)(), GOOD_LOSS_SCALES, N_its=50,
			# 						 which="linear", obs_noise_f=2**exp, title="$\sigma_{obs}$ = " + f"{2**exp:.2e}")

			thresholds = probe_stability_region(BEST_P, single_action, N_its=50, N_runs=100, plot=False, obs_noise_f=2**exp, stop_early=True)
			print(thresholds, exp)
			ys.append(thresholds)


		# save_data([xs, ys], "best_noiseless_policy_in_noise1")
		xs, ys = load_data("best_noiseless_policy_in_noise1")
		xs = [2**x for x in xs]
		zs = [np.product(y) for y in ys]

		# print("xs, ys", xs, ys)

		plt.semilogx()
		plt.ylabel("$f^1$ to $f'$")

		ls = plt.plot(xs, [y[:4] for y in ys])#, ls="", marker="x")
		plot_targets(plt.gca(), xs)
		# plt.plot([xs[0], xs[-1]], [.41, .41], "--", c="tab:purple")

		plt.twinx()
		plt.semilogx()

		plot_targets(plt.gca(), xs, tot=True)

		plt.ylabel("$f_{tot}$")
		ls.append(plt.plot(xs, zs, "k")[0])#, "k", ls="", marker="x")[0])
		plt.legend(ls, ["$f^1$", "$f^2$", "$f^3$", "$f^4$", "$f'$", "$f_{tot}$"], loc="upper left")

		plt.show()


		#superimpose [0.21	0.24	0.2	0.33	0.43]


	# noisy_old_policy()



	# loss_scales1 = GOOD_LOSS_SCALES.copy()
	# loss_scales1[0] = 1e6
	# loss_scales1[0] = 1e6

	# policy_loss_fn = get_policy_loss_fn(get_IC_gen_fn(0.001), loss_scales1, single_action, 20, 50, which="linear", obs_noise_f=0.0005)
	# contour_plot(policy_loss_fn, start_state=np.zeros(4), bounds=[[0, 0, -30, -30], [0, 0, 30, 30]], xi=2, yi=3, NX=25, NY=25, 
	# 			incl_f=False, pi_multiples=False)
	# plt.show()


	# linear_policy_contour_plots(single_action, N_runs=50, levels=np.linspace(-8, 0, 10), N=10, obs_noise_f=None)
	# linear_policy_contour_plots(single_action, N_runs=50, levels=np.linspace(-1, 0, 10), N=10, obs_noise_f=0.00001)
	# plt.show()



	def noisy_optimisation():

		noise_exps = np.arange(-10, -5, 0.5)
		# IC_exps = np.arange(-10, -4, 1)
		# noise_f = 0.005#2**-7
		IC_scale = 2**-4

		# IC_gen_fn = get_IC_gen_fn(BEST_FIC)

		noiseless_params = []
		noisy_params = []

		for noise_exp in []:# noise_exps:
			noise_f = 2**(noise_exp)
		# for IC_exp in []:#IC_exps:
			# IC_scale = 2.**(IC_exp)

			results = [{}, {}]

			Pa, La = get_robust_policy(single_action, second_scale=IC_scale, get_robustness=False)
			Pb, Lb = get_robust_policy(single_action, second_scale=IC_scale, obs_noise_f=noise_f, get_robustness=False)
			results[0]["P"] = Pa
			results[0]["L"] = La
			results[1]["P"] = Pb
			results[1]["L"] = Lb		
			results[0]["x"] = noise_f
			results[1]["x"] = noise_f		

			for i, res in enumerate(results):
				P, L = res["P"], res["L"]

				# print("P, L:", P, L)

				gains = get_noise_amplification(P, np.zeros(4), noise_f)
				# print("gains:", gains)
				ts = probe_stability_region(P, single_action, plot=False, obs_noise_f=noise_f, N_its=50, N_runs=150, gains=gains)

				results[i]["g"] = gains
				results[i]["thresh"] = ts

			print(results[0])
			print(results[1])

			noiseless_params.append(results[0])
			noisy_params.append(results[1])


		save_data([noiseless_params, noisy_params], "trash")
		# noiseless_params, noisy_params = load_data("fixed_noise_0_005_vs_f_IC")
		noiseless_params, noisy_params = load_data("2^-4_optimised_vs_noise_3")
		print(noisy_params)
		# noiseless_params, noisy_params = load_data("2^-4_optimised_vs_noise_attempt2")

		# axis = 2**noise_exps
		axis = 2.**np.arange(len(noisy_params))#IC_exps

		fig = plt.figure()

		fig.suptitle("Policies optimised for $f_{IC}=2^{-7}$ vs. $\sigma_{obs}$")

		axs = [fig.add_subplot(1, 4, 1)]
		axs.append(fig.add_subplot(1, 4, 2, sharey=axs[-1]))
		axs[-1].tick_params(labelleft=False)
		axs.append(fig.add_subplot(1, 4, 4))
		axs.append(fig.add_subplot(1, 4, 3, sharey=axs[-1]))
		axs[-1].tick_params(left=False, labelleft=False, right=True, labelright=False)
		axs[-2].tick_params(left=False, labelleft=False, right=True, labelright=True)


		axs[0].semilogx()
		axs[0].plot(axis, [x["thresh"] for x in noiseless_params], ls="", marker="o")
		axs[1].semilogx()
		axs[1].plot(axis, [x["thresh"] for x in noisy_params], ls="", marker="x",  mew=2)
		axs[3].semilogx()
		axs[3].plot(axis, [x["g"] if x["g"] is not None else [np.nan]*4 for x in noiseless_params], ls="", marker="o")
		axs[2].semilogx()
		axs[2].plot(axis, [x["g"] if x["g"] is not None else [np.nan]*4 for x in noisy_params], ls="", marker="x", mew=2)

		axs[0].set_ylabel("$f^i$ (robustness of policies)")
		axs[2].set_ylabel("$g_i$ (noise gains of policies)")
		axs[2].yaxis.set_label_position("right")

		# plot_targets(axs[0], axis)
		# plot_targets(axs[1], axis)


		fig1 = plt.figure()
		axs = [fig1.add_subplot(1, 2, 1)]
		axs.append(fig1.add_subplot(1, 2, 2))
		axs[-1].tick_params(left=False, labelleft=False, right=True, labelright=True)

		axs[0].plot(axis, [np.product(x["thresh"]) for x in noiseless_params], c="k", ls="", marker="o")
		axs[0].plot(axis, [np.product(x["thresh"]) for x in noisy_params], c="k", ls="", marker="x", mew=2)
		axs[0].semilogx()
		axs[0].set_title("$f_{tot}$")
		# plot_targets(axs[0], axis, tot=True)
		
		axs[1].plot(axis, [np.linalg.norm(x["g"]) if x["g"] is not None else np.nan for x in noiseless_params], c="k", ls="", marker="o")
		axs[1].plot(axis, [np.linalg.norm(x["g"]) if x["g"] is not None else np.nan for x in noisy_params], c="k", ls="", marker="x", mew=2)
		axs[1].semilogx()
		axs[1].set_title("$|g|^2$")

		fig2 = plt.figure()
		for i in range(4):
			plt.plot(np.nan, np.nan, ":", lw=8, label=VAR_STR[i] + " component")
		plt.plot(np.nan, np.nan, ":", lw=8, label="$f'$")
		plt.plot(np.nan, np.nan, "tab:grey", ls="", marker="o", label="Trained\nwithout noise")
		plt.plot(np.nan, np.nan, "tab:grey", ls="", marker="x", mew=2, label="Trained\nwith noise")
		plt.legend()
		plt.axis('off')
		plt.show()	

		# print(P3, L3)
		# print(ts3)

		
		# P1, L1 = optimise_linear_policy(None, GOOD_P, single_action, N_runs=100, IC_gen_fn=IC_gen_fn) #N_its make noise filtering better?
		# P2, L2 = optimise_linear_policy(None, GOOD_P, single_action, N_runs=100, IC_gen_fn=IC_gen_fn, obs_noise_f=noise_f)
		# print(P1, L1)
		# print(P2, L2)
		# ts1 = probe_stability_region(P1, single_action, N_its=50, N_runs=50, plot=False, obs_noise_f=noise_f, stop_early=True)
		# ts2 = probe_stability_region(P2, single_action, N_its=50, N_runs=50, plot=False, obs_noise_f=noise_f, stop_early=True)
		# print(ts1, ts2)


	# [.54, .71, 13.2, 2.2]

		# for i in range(10):
		# 	quick_linear_policy_sim(Pa, get_IC_gen_fn(tsb[-4]*np.array(tsb[:4]))(), GOOD_LOSS_SCALES, N_its=200,
		# 						 which="linear", obs_noise_f=noise_f, title="A, $\sigma_{obs}$ = " + f"{0.001:.2e}")

		# for i in range(10):
		# 	quick_linear_policy_sim(Pb, get_IC_gen_fn(tsb[-4]*np.array(tsb[:4]))(), GOOD_LOSS_SCALES, N_its=200,
		# 						 which="linear", obs_noise_f=noise_f, title="B, $\sigma_{obs}$ = " + f"{0.001:.2e}")


	noisy_optimisation()
	# exit()

	def superposed_model_plots():
		# n_f = 0.05 # levels = -1
		n_f = 0.01 # levels = -3

		print("doing contour plots")

		bounds = [[-15, -4, 0, 0], [15, 6, 30, 3]]
		# bounds = [[-25, -10, -15, -10], [25, 15, 30, 10]]
		# model_fn = to_update_fn_w_action(get_optimal_nonlin_fit(7))
		N = 20

		# axs= linear_policy_contour_plots(single_action, N_runs=50, levels=np.linspace(-12, 0, 10), N=N)
		# plt.suptitle("")
		# axs[1].set_title("Loss surfaces, $\sigma_{obs}=" + str(n_f) + "$, $f_{IC} = 0.01$")
		# linear_policy_contour_plots(single_action, N_runs=50, levels=np.linspace(-3, 0, 10), filled=False, cmap="plasma", axs=axs, N=N, obs_noise_f=n_f)
		# plt.show()


		P, L = optimise_linear_policy(0.001, GOOD_P_001, single_action, N_runs=100)
		L = -15
		P = GOOD_P_001
		print(L)
		policy_loss_fn = get_policy_loss_fn(get_IC_gen_fn(0.001), GOOD_LOSS_SCALES, single_action, 20, 50, which="linear")
		axs = six_planes(policy_loss_fn, start_state=P, varying_bounds=True, bounds=bounds, 
						NX=N, NY=N, incl_f=False, pi_multiples=False, levels=np.linspace(-12, 0, 10))


		IC_gen_fn = get_IC_gen_fn(0.001)

		P, L = optimise_linear_policy(0.001, GOOD_P, single_action, N_runs=100, IC_gen_fn=IC_gen_fn, obs_noise_f=n_f)
		print(L)
		policy_loss_fn = get_policy_loss_fn(IC_gen_fn, GOOD_LOSS_SCALES, single_action, 20, 50, which="linear", obs_noise_f=n_f)
		six_planes(policy_loss_fn, start_state=P, varying_bounds=True, bounds=bounds, NX=N, NY=N, 
					incl_f=False, pi_multiples=False, filled=False, cmap="plasma", levels=np.linspace(-3, 0, 10), axs=axs)

		plt.show()

	superposed_model_plots()