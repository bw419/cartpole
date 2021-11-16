from globals import *
from utils import *
from simulation_utils import *
from rollouts import pendulum_phase_portrait
from cartpole import *
from control import *
from nonlinear_control_policy import *
from time import perf_counter

single_action = fast_single_action



def optimise(start_pos, loss_fn, IC_gen_fn, offset=np.zeros(4), N_its=40, N_runs=100, log=True):




	p_loss_fn = get_policy_loss_fn(IC_gen_fn, None, single_action, N_its, N_runs, seed=True, which="nonlinear", loss_fn=loss_fn)
	# p_loss_fn = get_policy_loss_fn(IC_gen_fn, loss_scales, single_action, N_its, N_runs, seed=True, which="nonlinear")


	# contour_plot(p_loss_fn, start_pos, bounds=[0., 0., 20, 10], xi=2, yi=3, NX=10, NY=15, pi_multiples=False)
	# plt.show()
	# # p_loss_fn = reparametrise(p_loss_fn, parametrisation)

	if log:
		print("initial loss:", p_loss_fn(start_pos), start_pos)

	def callback(xk):
		print("optimising...",  xk[:5], "\r", end="")
		return False

	# loss_fn1 = lambda x: loss_fn(x) + (1-np.exp(-x[3]**2))#np.sum(np.square(x))))

	res = scipy.optimize.minimize(p_loss_fn, start_pos, method="Nelder-Mead", options={}, callback=callback)


	print("finished optimising.                                                                          \r", end="")
	print(res.x, res.fun)
	return res.x



if __name__ == "__main__":


	loss_scales = GOOD_LOSS_SCALES.copy()
	loss_scales[0] = 1e5
	loss_scales[1] = 1e5
	S_loss_fn = get_loss_fn(loss_scales)


	E_SCALE = 1
	E_loss_fn = lambda x: 1-np.exp(-np.square(get_tot_pole_energy(x))/E_SCALE)
	



	# policy = get_policy_fn([0.3, .1, 10], which="nonlinear")
	# contour_plot(policy, incl_f=False, bounds=P_RANGE4)#, levels=np.linspace(-MAX_F, MAX_F, 9)
	# plt.figure()
	# policy = lambda x, it=0: np.dot(THETA_ONLY_P, remapped_angle(x))
	# contour_plot(policy, incl_f=False, bounds=P_RANGE4)#, levels=np.linspace(-MAX_F, MAX_F, 9)
	# plt.show()



	# policy = get_nonlin_policy(parametrisation_A(MAX_F, np.pi/2, 7, 14, 7, 1, -1))
	# policy = get_nonlin_policy(parametrisation_A1([np.pi/2, 7, 14, 7, 1, -1]))
	
	# linear policy with phase portrait
	if False:
		THETA_ONLY_P = np.array([0, 0, 20, 2.1]) #theta can range from 13-23ish, theta dot more fixed

		policy = lambda x, it=0: np.dot(THETA_ONLY_P, remapped_angle(x))
		contour_plot(policy, incl_f=False, bounds=P_RANGE4*2)#, levels=np.linspace(-MAX_F, MAX_F, 9)
		# pendulum_phase_portrait()
		# plt.xlim(-2*np.pi, 2*np.pi)
		plt.show()



	# FOR A
	params0 = [np.pi/2, 7, 14, 7, -1, 1]

	# FOR C
	params0 = [1.1, .5, 10]
	# params = [1.14091947e+00, 5.05683574e-01, 10]
	# FOR C5
	# (C_weight, C_pos, C_perp_scale, B_weight, theta_scale, theta_d_scale, H_weight, H_scale1, H_scale2, H_pos)
	params0 = [1, 5, 10, 1, 1, 7, 1, 0.5, 5, 1]
	# params0 = [ 2.98994372e+00,  1.90602191e+00,  1.24360033e+01, -5.44214201e-04, 1.41111068e+00,  1.10785614e+01,  8.51842056e-01, -2.53318594e+00, 2.54671399e+01]
	# params1 = [ 1.05564992,  0.48756235,  9.2086839,   1.02386185,  1.06839144,  6.48885736, 0.39184368, 1.09900579, 14.4590388]
	# params1 = [ 1.78501623,  0.61379702, 10.94947416, -0.01105474,  1.87556208,  2.43621067, 0.35452325,  1.2828691,   9.84625962]

	# FOR D: C_weight, C_pos, C_perp_scale, D_weight, theta_scale, theta_d_scale, D_pos
	# params0 = [1., .1, 10, .5, 1., 7., 20]

	# FOR E1: D_weight, D_pos1, D_pos2, E_weight, E_pos1, E_pos2, t_scale, td
	# params0 = [1, 24, 24, 5., 2, 2, .5, 2.5]
	# FOR E2: C_weight, C_pos, C_perp_scale, D_weight, D_pos1, D_pos2, E_weight, E_pos1, E_pos2
	# params0 = [1.0, .1, 10, 1.2, 17, 20, 1.2, 1.5, 7]
	# FOR E3: C_weight, C_pos, C_perp_scale
	# params0 = [1.0, .05, 10]


	# FOR H: (H_weight, theta_scale, theta_d_scale, H_pos)
	# params0 = [0.98226734, 0.58075192, 2.56362654, 3.39621148]
	#optimised to get to E = 0 as fast as possible: [5.50545268 0.61243916 3.64428872 0.62930987]

	# FOR I: 
	# opt?  [ 1.01784687, -1.01220907,  1.06890036,  0.21019188,  5.45656662, 14.74992787, 15.2306644,  20.12657658]
	# params0 = [1, -1, 1, .2, 5, 15, 20, 20]

	# full_policy1
	# (P_scale, Q_scale, dp2, dq2, w2, w3):#, d3=0, w3=0):
	params0 = [.3, 1, .5, 0, 0, 1, .5]#, 0.0, 0., 0.]
	# params0 = [ 0.55582017,  5.3664485,   0.39576763,  0.96512624,  5.09120022, 15.2828192,  1.01749172,  1.02051497]
	
	policy = get_policy_fn(params0, which="nonlinear")
	policy1 = lambda x, it=0: np.dot(THETA_ONLY_P, remapped_angle(x))


	# def grad(f, x=[0,0,0,0]):
	# 	x=np.array(x)
	# 	d = np.array([0, 0, 0.0001, 0])
	# 	return (f(x+d)-f(x))/d[2]

	# lin_grad = grad(policy1)

	# def f(x):
	# 	p = params0.copy()
	# 	p[2] = x
	# 	return grad(get_policy_fn(p, which="nonlinear")) - lin_grad

	# res = scipy.optimize.root_scalar(f, x0=0, x1=5)
	# params0[2] = res.root
	# policy = get_policy_fn(params0, which="nonlinear")

	# print(res.root)

	# contour_plot(policy, incl_f=False, bounds=P_RANGE4, NX=100, NY=100)
	line_plot(policy, incl_f=False, bounds=[-np.pi, np.pi], xi=2)
	line_plot(policy1, incl_f=False, bounds=[-np.pi, np.pi], xi=2)

	plt.show()




	# fig, axs = plt.subplots(2, 1)
	# plt.sca(axs[0])
	# line_plot(policy, incl_f=False, bounds=[-np.pi, np.pi], xi=2)
	# plt.sca(axs[1])
	# line_plot(policy, incl_f=False, bounds=[-15, 15], xi=3)
	# plt.show()


	def IC_gen_fn_Egt0():
		state = rand_state4(P_RANGE4*1.0)
		while get_tot_pole_energy(state) < 0:
			state = rand_state4(P_RANGE4*1.0)
		return state

	IC_gen_fn = get_IC_gen_fn(0.1, offset=[0,0,0,0])
	# IC_gen_fn = get_IC_gen_fn(.2, offset=[0,0,np.pi,0])
	# IC_gen_fn = get_IC_gen_fn(0.3)
	
	# policy = lambda x, it=0: np.dot(THETA_ONLY_P, remapped_angle(x))


	if False:

		# params = optimise(params0, E_loss_fn, IC_gen_fn)
		params = optimise(params0, S_loss_fn, IC_gen_fn)
		# params = optimise(params0, E_loss_fn, IC_range=0.0, N_runs=1, offset=[0,0,np.pi,0])

		policy = get_policy_fn(params0, which="nonlinear")
		contour_plot(policy, incl_f=False, bounds=P_RANGE4)#, levels=np.linspace(-MAX_F, MAX_F, 9)
		plt.figure()


		print("optimal params:", params)
		policy = get_policy_fn(params, which="nonlinear")

		contour_plot(policy, incl_f=False, bounds=P_RANGE4)#, levels=np.linspace(-MAX_F, MAX_F, 9))

		# pendulum_phase_portrait()
		# plt.xlim(-2*np.pi, 2*np.pi)

		plt.show()



	# contour_plot(policy, incl_f=False, bounds=P_RANGE4, NX=25, NY=25)#, levels=np.linspace(-MAX_F, MAX_F, 9)
	# plt.show()

	contour_plot(policy, incl_f=False, bounds=P_RANGE4, NX=100, NY=100)#, levels=np.linspace(-MAX_F, MAX_F, 9)
	plt.show()

	for i in range(20):
		# print("running sim")
		# state = rand_state4(P_RANGE4*1.0) + [0,0,0,0]#rand_states4()
		# while get_tot_pole_energy(state) < 3:
			# state = rand_state4(P_RANGE4*1.0)	

		state = IC_gen_fn()

		states, actions, L = policy_simulation(state, E_loss_fn, single_action, policy, 20, 
												ret_success=False, stop_early=False)

		# print(L)

		if False:
			states = np.array([remapped_angle(x) for x in states])
			plt.plot(states[:, 2], states[:,3], "k")#L)
			continue

		if False:
			cmap = cm.get_cmap("autumn_r", 256)


			if L > 0.5:		
				plt.plot(states[:, 2], states[:,3], c=cmap(L), ls="", marker=".", mew=0, alpha=1)#L)
			else:
				plt.plot(states[:, 2], states[:,3], c=cmap(L), ls="", marker=".", mew=0, alpha=1)#L)
			continue

		es = [get_tot_pole_energy(x) for x in states]

		plot_states(states, actions, show_vars=(2,3), show_F=False, standalone_fig=True, markers=True)
		plt.plot(0.2*np.arange(len(states)), es, label="Energy")
		plt.legend()
		plt.title("Initial guess nonlinear policy simulation")
		plt.show()
	
	plt.ylim(-P_RANGE4[3], P_RANGE4[3])
	plt.show()


	exit()

