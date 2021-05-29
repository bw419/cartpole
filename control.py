from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
import collections
from time import perf_counter


# could make this different for each component
# loss_sig = np.array([.5, .5, .5, .5])
loss_sig = np.array([.5, .5, 1e4, 1e4])

# loss function given a state vector. the elements of the state vector are
def loss_fn(state):
	state[3] = remap_angle(state[3])
	return 1 - np.exp(-np.sum((state[:4]**2)/loss_sig**2)/2)


def policy_loss(IC, update_fn, policy_fn, max_it):

	state = IC
	L = 0

	for it in range(max_it):
		action = policy_fn(state)
		L += loss_fn(state)
		state = update_fn(state, action)

	return L


def policy_simulation(IC, update_fn, policy_fn, max_it):

	states = [IC]
	L = 0

	for it in range(max_it):
		action = policy_fn(state)
		L += loss_fn(state)
		state = update_fn(state, action)

	return states




def linear_policy(P, state):
	return np.dot(P, state)

def linear_loss(P, IC, update_fn, loss_window=5):
	return policy_loss(IC, update_fn, lambda state: linear_policy(P, state), loss_window)





P = np.array([0, 0, 0, 0])

THETA_ONLY_P = np.array([0, 0, 20, 2.1]) #theta can range from 13-23ish, theta dot more fixed




if __name__ == "__main__":

	IC_range = [1, 1, 0, 0]
	def gen_IC():
		return IC_range*rand_state4() + [0, 0, np.pi, 0]

	N_runs = 5
	N_its = 10
	def loss(P):
		return sum([linear_loss(P, gen_IC(), single_action, N_its) for i in range(N_runs)])/(N_runs*N_its)


	loss, states = simulate()


	start_state = np.zeros(4)
	print(IC_range*rand_state4())

	for s in np.linspace(-10, 10, 1):
		print("it")
		start_state[0] = s
		plt.figure()
		contour_plot(loss, start_state=start_state, bounds=[30, 30, 30, 15], xi=0, yi=1, incl_f=False, pi_multiples=False)
		# contour_plot(loss, start_state=start_state, bounds=[30, 30, 30, 15], xi=0, yi=1, incl_f=False, pi_multiples=False)
	
	plt.show()

