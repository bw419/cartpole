from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
from nonlinear_model import *
import collections
from time import perf_counter


single_action = fast_single_action

model_fn = load_model_function("nonlin_15_13")
# model_fn = load_model_function("nonlin_16_14")
learned_update_fn = to_update_fn_w_action(model_fn)


# could make this different for each component
loss_sig = np.array([.5, .5, .5, .5])
# loss_sig = np.array([5, .5, 1e4, 1e4])

def get_loss_fn(loss_sig=None, range_prop=None):
	if loss_sig is None and range_prop is None:
		raise Exception("give some scale information")

	if range_prop is not None:
		loss_sig = range_prop * P_RANGE4
		loss_sig[2]
	return lambda x: 1-np.exp(-(
		np.square((x[0])*INV_SQRT2/loss_sig[0]) +
		np.square((x[1])*INV_SQRT2/loss_sig[1]) +
		np.square(np.sin(.5*x[2])*INV_SQRT2/loss_sig[2]) +
		np.square((x[3])*INV_SQRT2/loss_sig[3])
	))



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
def get_policy_fn(param_object, which="linear"):

	if which == "linear":
		# parameters are a simple 4-vector of linear coefficients
		P = param_object

		def p(x):
			return np.dot(P, remapped_angle(x))

	elif which == "nonlinear_1":
		# parameters are more complex. W_elems has 10 elements, w_vector and basis_fn_centres have same number of elements.
		basis_fn_centres, W_elems, w_vector = param_object

		W_matrix = [[W_elems[0], W_elems[4], W_elems[7], W_elems[9]],
					[W_elems[4], W_elems[1], W_elems[5], W_elems[8]],
					[W_elems[7], W_elems[5], W_elems[2], W_elems[6]],
					[W_elems[9], W_elems[8], W_elems[6], W_elems[3]]]
		W_matrix = np.array([np.array(x) for x in W_matrix])

		NONLIN_TEMP = np.zeros((len(w_vector), 4))

		def p(x):
			NONLIN_TEMP[:,:] = x - basis_fn_centres 
			return np.dot(w_vector, np.exp(-0.5 * np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1)))

	return p




def policy_simulation(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True):

	states = [IC]
	actions = [policy_fn(IC)]
	losses = [loss_fn(states[-1])]
	prev_loss = np.inf
	states_int = np.zeros(4)

	for it in range(max_it):
		# states.append(update_fn([states[-1][0], states[-1][1], states[-1][2], states[-1][3], actions[-1]]))
		states.append(update_fn(states[-1], actions[-1]))
		# xpos_int[0] += 0.01*states[-1][0]
		# states_int += 0.01*states[-1]
		actions.append(policy_fn(states[-1] + states_int))
		losses.append(loss_fn(states[-1]))
		# if it > 15 and np.max(losses[-15:]) < 0.01:
			# print(states, actions, losses[-10:])
			# return states, actions, np.mean(losses)

		if it % 5 == 0 and stop_early:
			if it % 5 == 0 and np.abs(states[-1][0]) > 3*P_RANGE4[0]:
				return states, actions, 1.0
			if stop_early and np.abs(np.sum(losses) - prev_loss) < 0.0001:
				return states, actions, np.sum(losses)/max_it
			prev_loss = np.sum(losses)

	return states, actions, np.mean(losses)



# this is where the bulk of computation happens.
# potentially make its parts more efficient?
def policy_loss(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True):

	state = IC
	L = 0
	prev_loss = np.inf
	for it in range(max_it):

		L += loss_fn(state)
		action = policy_fn(state)
		state = update_fn(state, action) # this takes the majority of the computation time

		# test to terminate early
		if it % 5 == 0 and stop_early: 
			if np.abs(state[0]) > 3*P_RANGE4[0]:
				return 1.0
			if np.abs(L - prev_loss) < 0.0001:
				return L/max_it
			prev_loss = L

	return L/max_it






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



def policy_loss_N_runs(IC_gen_fn, loss_fn, update_fn, policy_fn, N_its, N_runs):
	return sum([
		policy_loss(IC_gen_fn(), loss_fn, update_fn, policy_fn, N_its)
		for i in range(N_runs)
		])/(N_runs)


def policy_success(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=True):
	loss = policy_loss(IC, loss_fn, update_fn, policy_fn, max_it, stop_early=stop_early)
	return loss < 0.1

def policy_success_rate(IC_gen_fn, loss_fn, update_fn, policy_fn, N_its, N_runs):
	return np.count_nonzero([
		policy_loss(IC_gen_fn(), loss_fn, update_fn, policy_fn, N_its) < 0.1
		for i in range(N_runs)
		])/(N_runs)

# obtain a function mapping policy parameters to a loss value
def get_policy_loss_fn(IC_gen_fn, loss_sig, model_fn, N_runs, N_its, which="linear"):
	if which == "linear":
		def loss(P):
			return policy_loss_N_runs(IC_gen_fn, get_loss_fn(loss_sig), model_fn, get_policy_fn(P, which), N_its, N_runs)
	elif which == "nonlinear_1":
		def loss(centres, W_elems, w):
			return policy_loss_N_runs(IC_gen_fn, get_loss_fn(loss_sig), model_fn, get_policy_fn((centres, W_elems, w), which), N_its, N_runs)
	return loss







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

GOOD_LOSS_SCALES = [5, 5, 1, 5]
ALL_P = np.array([.4, .3, 18, 1.95]) 
#####################################################################

# TODO: 
# - optimise on modelled dynamics
# - optimise for observation noise
# - optimise for dynamics noise
# - optimise for both?





import inspect
def optimise_linear_policy(IC_proportions, N_runs=50, N_its=100):
	loss_fn = get_policy_loss_fn(get_IC_gen_fn(IC_proportions), get_loss_fn(loss_sig), single_action, N_runs, N_its)
	for i in range(5):
		print(loss_fn(ALL_P))

	bounds= [[0, 1], [0, 2], [10, 20], [1,3]]
	
	def callback(xk):
		print("doing stuff", xk)
		return False

	Ns = 11
	print(Ns)
	print(scipy.optimize.brute(loss_fn, ranges=bounds, Ns=Ns, full_output=True)[:2])
	# print(scipy.optimize.minimize(loss_fn, [.3,.6,15,2.5], bounds=bounds, method="SLSQP", options={"eps":0.1, "ftol":0.001, "maxiter": 100, "disp":True}, callback=callback))




def linear_policy_contour_plots(IC_proportions=[1., .1, .1, .1], N_runs=15, N_its=50):

	IC_gen_fn = get_IC_gen_fn(IC_proportions)

	start_state = np.zeros(4)
	for s in np.linspace(0.2, 0.6, 1):

		s=.3
		print("it", s)
		start_state[0] = s
		plt.figure()
		plt.title(f"x = {round(s, 1)}")


		# # OPTIMISING FOR ONLY THETA, THETA DOT: (ENSURE loss_fn(P) includes x,x dot part, and i.c. range is sensible)
		# policy_loss_fn = get_policy_loss_fn(IC_gen_fn, [1e6, 1e6, 1, 5], single_action, N_runs, N_its, which="linear")
		# contour_plot(policy_loss_fn, start_state=start_state, bounds=[[0, 0, 0, -3], [0, 0, 30, 7]], xi=2, yi=3, NX=10, NY=20, incl_f=False, pi_multiples=False)

		# # OPTIMISING FOR FIRST 3:
		# policy_loss_fn = get_policy_loss_fn(IC_gen_fn, [1e6, 5, 1, 5], single_action, N_runs, N_its, which="linear")
		# contour_plot(policy_loss_fn, start_state=start_state, bounds=[[0, -1.0, 0, .7], [0, 2.5, 0, 4.3]], xi=1, yi=3, NX=15, NY=15, incl_f=False, pi_multiples=False, levels=np.linspace(0, 1, 11))
	

		## OPTIMISING FOR ALL 4:
		policy_loss_fn = get_policy_loss_fn(IC_gen_fn, [5, 5, 1, 5], single_action, N_runs, N_its, which="linear")
		def modified_loss_fn(P):
			P[3] = 1.8 + 0.5*P[1]
			return policy_loss_fn(P)
		contour_plot(modified_loss_fn, start_state=start_state, bounds=[[0, 0.0, 10, 0], [0, 2.5, 25, 0]], xi=1, yi=2, NX=10, NY=10, incl_f=False, pi_multiples=False, levels=np.linspace(0, 1, 11))
	
	plt.show()
	



def quick_linear_policy_sim(P, IC, loss_sig, N_its=50, markers=False):
	fig, ax = plt.subplots(1, 2)

	loss_fn = get_loss_fn(loss_sig)
	policy_fn = get_policy_fn(P, which="linear")

	states, actions, L = policy_simulation(IC, loss_fn, single_action, policy_fn, N_its, stop_early=True)
	plot_states(states, actions, ax=ax[0], show_F=False, markers=markers)
	states, actions, L = policy_simulation(IC, loss_fn, learned_update_fn, policy_fn, N_its, stop_early=True)
	plot_states(states, actions, ax=ax[1], show_F=False, markers=markers)
	plt.title(f"Loss={L}")
	plt.show()




if __name__ == "__main__":

	# optimise_linear_policy([.3, .3, .3, .3])

	# t = time.perf_counter()
	# gen_IC = get_IC_gen_fn([.3, .3, .3, .3])
	# loss_fn = get_loss_function(loss_sig, single_action, gen_IC, 100, 100)
	# print(loss_fn([.5, 1.8, 16, 3.]))
	# print(time.perf_counter() -t)

	def rand_quick_runs(P=ALL_P, IC_props=[.1,.1,.1,.1]):
		while True:
			quick_linear_policy_sim(P, get_IC_gen_fn(IC_props)(), GOOD_LOSS_SCALES, 500)


	# IC_PROPS = np.array([.0, .0, .0, .5])

	def quick_success_rate(P, IC_props=[.1,.1,.1,.1]):
		loss_fn = get_loss_fn(GOOD_LOSS_SCALES)
		policy_fn = get_policy_fn(P, which="linear")
		s = policy_success_rate(get_IC_gen_fn(IC_props), loss_fn, single_action, policy_fn, 300, 500)
		print(s)



	IC_PROPS = np.array([.4, .8, .3, .5])

	rand_quick_runs([.4, .3, 18, 1.95], 0.33*IC_PROPS)



	# for x in np.linspace(.3, .5, 25):
	# 	quick_success_rate([.4, .3, 18, 1.8 + 0.5*.3], x*IC_PROPS)
	# 	print("- ", x)

	print("doing contour plots")
	linear_policy_contour_plots(IC_proportions=IC_PROPS, N_runs=15, N_its=200)


	plt.show()