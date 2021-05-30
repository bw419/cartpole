from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
from nonlinear_model import *
import collections
from time import perf_counter


model_fn = load_model_function("nonlin_15_13")
# model_fn = load_model_function("nonlin_16_12")
update_fn = to_update_fn_w_action(model_fn)


# could make this different for each component
loss_sig = np.array([.5, .5, .5, .5])
# loss_sig = np.array([5, .5, 1e4, 1e4])

loss_fn = lambda x: 1-np.exp(-(
	np.square((x[0])*INV_SQRT2/loss_sig[0]) +
	np.square((x[1])*INV_SQRT2/loss_sig[1]) +
	np.square(np.sin(.5*x[2])*INV_SQRT2/loss_sig[2]) +
	np.square((x[3])*INV_SQRT2/loss_sig[3])
))

# loss function given a state vector. the elements of the state vector are
# def loss_fn(state):
	# return 1 - np.exp(-np.sum((remapped_angle(state[:4])**2)/loss_sig**2)/2)


def policy_loss(IC, update_fn, policy_fn, max_it, stop_early=True):

	state = IC
	L = 0
	prev_loss = np.inf
	for it in range(max_it):
		L += loss_fn(state)
		action = policy_fn(state)
		state = update_fn(state, action)

		if it % 5 == 0 and stop_early: 
			if np.abs(state[0]) > 3*P_RANGE4[0]:
				return states, actions, 1.0
			if np.abs(L - prev_loss) < 0.001:
				return states, actions, L/max_it
			prev_loss = L

	return L/max_it


def policy_simulation(IC, update_fn, policy_fn, max_it, stop_early=True):

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
			if stop_early and np.abs(np.sum(losses) - prev_loss) < 0.001:
				return states, actions, np.sum(losses)/max_it
			prev_loss = np.sum(losses)

	return states, actions, np.mean(losses)



def linear_policy(P, state):
	return np.dot(P, remapped_angle(state))

def linear_loss(P, IC, update_fn, loss_window=5):
	return policy_loss(IC, update_fn, lambda state: linear_policy(P, state), loss_window)





P = np.array([0, 0, 0, 0])

THETA_ONLY_P = np.array([0, 0, 20, 2.1]) #theta can range from 13-23ish, theta dot more fixed
X_ONLY_P = np.array([-10, -5.5, 0, 0]) #theta can range from 13-23ish, theta dot more fixed
THETA_XDOT_ONLY_P = np.array([0, 1.1, 15, 2.5]) #theta can range from 13-23ish, theta dot more fixed
ALL_P = np.array([.3, .6, 15, 2.5]) #theta can range from 13-23ish, theta dot more fixed




if __name__ == "__main__":

	# IC_range = [1, 1, np.pi, 10]
	IC_range = [.1, .1, .1, .1]
	N_runs = 15
	N_its = 50

	def gen_IC():
		return IC_range*rand_state4()# + [0, 0, np.pi, 0]

	def loss(P):
		return sum([linear_loss(P, gen_IC(), single_action, N_its) for i in range(N_runs)])/(N_runs)


	def quick_policy_sim(P, IC):
		fig, ax = plt.subplots(1, 2)
		states, actions, L = policy_simulation(IC, single_action, lambda state: linear_policy(P, state), 100, stop_early=True)
		plot_states(states, actions, ax=ax[0])
		states, actions, L = policy_simulation(IC, update_fn, lambda state: linear_policy(P, state), 100, stop_early=True)
		plot_states(states, actions, ax=ax[1])
		plt.title(f"Loss={L}")
		plt.show()

	loss_sig = np.array([1e6, 1e6, .5, .5])
	# loss_sig = np.array([5, .5, 1e6, 1e6])

	# quick_policy_sim(X_ONLY_P, [1, 1, np.pi, 10]*rand_state4())


	start_state = np.zeros(4)
	print(IC_range*rand_state4())

	loss_sig = np.array([1, 5, 1, 5])
	while True:
		quick_policy_sim(ALL_P, [.1, .1, .1, .1]*rand_state4() + [0, 0, 2*np.pi, 0])


	print("doing a contour plot")

	for s in np.linspace(0, .8, 9):
		print("it", s)
		start_state[0] = s
		plt.figure()
		plt.title(f"x = {round(s, 1)}")



		# OPTIMISING FOR ONLY x, x dot: (allow IC range [1, 1, pi, 10] centered around pi for third element, not that that matters here)
		# loss_sig = np.array([5, .5, 1e4, 1e4])
		# contour_plot(loss, start_state=start_state, bounds=[[-30, -10, 0, 0], [0, 0, 0, 0]], xi=0, yi=1, NX=25, NY=25, incl_f=False, pi_multiples=False)
		# Also try this for [.5, .5, 1e4, 1e4] and other x loss scales?


		# OPTIMISING FOR ONLY THETA, THETA DOT: (ENSURE loss(P) includes x,x dot part, and i.c. range is sensible)
		# loss_sig = np.array([1e6, 1e6, 1, 5])
		# contour_plot(loss, start_state=start_state, bounds=[[0, 0, 0, -3], [0, 0, 30, 7]], xi=2, yi=3, NX=10, NY=20, incl_f=False, pi_multiples=False)
	

		# OPTIMISING FOR FIRST 3:
		# loss_sig = np.array([1e6, 5, 1, 5])
		# X, Y, Z, ax, cb = contour_plot(loss, start_state=start_state, bounds=[[0, -1.0, 0, .7], [0, 2.5, 0, 4.3]], xi=1, yi=3, NX=15, NY=15, incl_f=False, pi_multiples=False, levels=np.linspace(0, 1, 11))
		# contour_plot(loss, start_state=start_state, bounds=[30, 30, 30, 15], xi=0, yi=1, incl_f=False, pi_multiples=False)
	
		# OPTIMISING FOR ALL 4:
		loss_sig = np.array([1, 5, 1, 5])
		def modified_loss(P):
			P[3] = 1.8 + 0.5*P[1]
			return sum([linear_loss(P, gen_IC(), single_action, N_its) for i in range(N_runs)])/(N_runs)
		X, Y, Z, ax, cb = contour_plot(modified_loss, start_state=start_state, bounds=[[0, 0.0, 10, 0], [0, 2.0, 20, 0]], xi=1, yi=2, NX=15, NY=15, incl_f=False, pi_multiples=False, levels=np.linspace(0, 1, 11))
		# contour_plot(loss, start_state=start_state, bounds=[30, 30, 30, 15], xi=0, yi=1, incl_f=False, pi_multiples=False)
	


	plt.show()

