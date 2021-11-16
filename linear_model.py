from globals import *
from utils import *
from cartpole import *


def target_training_data(N=512, t_step=0.2, sobol=True, incl_f=True, range_factor=1.0):

	N_X_cmpts = 5 if incl_f else 4

	Y = np.zeros((N, 4))
	X = np.zeros((N, N_X_cmpts))

	if sobol:
		sobol_init()

	for i in range(N):
		if sobol:
			X[i,:] = range_factor*sobol_rand_state5(TRAIN_P_RANGE5)[:N_X_cmpts]
		else:
			X[i,:] = range_factor*rand_state5(TRAIN_P_RANGE5)[:N_X_cmpts]

		Y[i,:] = fast_target(X[i, :], t_step)

	return X, Y


def corrupt_msmt(x, obs_noise_fraction=None):
	if obs_noise_fraction is None:
		return x
	else:
		return x + np.random.normal(scale=obs_noise_fraction*P_RANGE4, size=len(x))

def corrupt_msmts(X, obs_noise_fraction=None):
	if obs_noise_fraction is None:
		return x
	else:
		return X + np.random.normal(scale=obs_noise_fraction*P_RANGE4, size=X.shape)

def corrupt_force(f, obs_noise_fraction=None):
	if obs_noise_fraction is None:
		return f
	else:
		return f + np.random.normal(scale=obs_noise_fraction*P_RANGE5[4])


# def corrupt_single(X, Y, obs_noise=None):

# 	return corrupt(X, Y, obs_noise, len(X))



def corrupt(X, Y, obs_noise=None, obs_bias=None, N_X_cmpts=None):

	if N_X_cmpts is None:
		N_X_cmpts = X.shape[1]

	if obs_noise is None:
		obs_noise = np.zeros(N_X_cmpts)
	if obs_bias is None:
		obs_bias = np.zeros(N_X_cmpts)

	if obs_bias is not None or obs_noise is not None:

		obs_noise = np.array(obs_noise)
		obs_bias = np.array(obs_bias)

		noisy_X = X + np.random.normal(loc=obs_bias, scale=obs_noise, size=X.shape)
		noisy_Y = Y + np.random.normal(loc=obs_bias[:4], scale=obs_noise[:4], size=Y.shape)

	return noisy_X, noisy_Y



# enforces lack of dependence on theta or x
def linear_fit(N=1024, t_step=0.2, sobol=True, return_data=False, incl_f=True, enforce_constraints=True, obs_noise=None, dyn_noise=None):
	
	set_dynamic_noise(dyn_noise)
	X, Y = target_training_data(N, t_step, sobol, incl_f)
	set_dynamic_noise(0.)

	C = np.zeros((4, X.shape[1]))

	nX, nY = corrupt(X, Y, obs_noise)

	nonzero_cols = (0,1,3)
	if incl_f:
		nonzero_cols += (4,)
	if not enforce_constraints:
		nonzero_cols = np.arange(X.shape[1])

	CT, res, rank, s = np.linalg.lstsq(nX, nY, rcond=None)
	C[:,nonzero_cols] = CT.T

	if return_data:
		return (X, Y, nX, nY), C
	else:
		return C






# Noiseless case
def get_good_linear_fit(return_data=False, enforce_constraints=True, incl_f=True):
	if return_data:
		data, C = linear_fit(N=2**12, return_data=True, enforce_constraints=enforce_constraints, incl_f=incl_f)
	else:
		C = linear_fit(N=2**12, return_data=False, enforce_constraints=enforce_constraints, incl_f=incl_f)

	def fn(x):
		return C @ x

	if return_data:
		return fn, data[0], data[1]
	else:
		return fn



def get_good_noisy_linear_fit(obs_noise_f=None, dyn_noise=None, N=2**12, incl_f=True):

	if obs_noise_f is None:
		obs_noise = None
	else:
		obs_noise = obs_noise_f*P_RANGE5

	C = linear_fit(N=N, return_data=False, sobol=True, incl_f=incl_f, enforce_constraints=False,
					obs_noise=obs_noise, dyn_noise=dyn_noise)

	def fn(x):
		return C @ x

	return fn



def lin_model_convergence(**kwargs):
	
	X = np.power(10, np.linspace(1, 3.5, 100))
	C = np.zeros((len(X), 16))
	y = np.zeros(len(X))

	print(X)

	for i, n in enumerate(X):
		C = linear_fit(int(n), **kwargs)
		# print(C)
		y[i] = np.linalg.norm(C, "fro")
		print(i, y[i])

	print(X, y)

	plt.semilogx()
	# plt.gca().set_xticks(X)
	plt.scatter(X, y, marker="x", s=10)
	plt.plot([500, 500], [0, max(y)], "r--")
	plt.title("Frobenius norm of C for N\nrandomly drawn training states")
	plt.ylabel("Norm")
	plt.xlabel("N", labelpad=2.0)


# seems kind of unchanged?
def convergences():
	plt.figure()
	lin_model_convergence(return_data=False, sobol=False, incl_f=True)
	lin_model_convergence(return_data=False, sobol=False, incl_f=True,
							obs_noise=0.01*P_RANGE5, dyn_noise=None)
	lin_model_convergence(return_data=False, sobol=False, incl_f=True,
							obs_noise=0.05*P_RANGE5, dyn_noise=None)
	plt.show()

	ax = plot_scan_



if __name__ == "__main__":

	while True:
		ax = plot_scan_matrix(single_action5)
		# plot_scan_matrix(get_good_linear_fit(enforce_constraints=False), axs_in=ax)
		plt.suptitle("Scans across single variables for single action")
		plt.show()

	data, C1 = linear_fit(N=2048, return_data=True, sobol=True, incl_f=True)
	print("------")
	print(C1)

	data, C2 = linear_fit(N=2048, return_data=True, sobol=True, incl_f=True, enforce_constraints=False,
							obs_noise=0.1*P_RANGE5, dyn_noise=None)
	print(C2)
	print("------")


	X, Y, Xn, Yn = data

	# convergences()




	for k in range(4):
		plt.figure()
		contour_plot(lambda x: (fast_target(x))[k], incl_f=True, NX=50, xi=2, yi=4)
		plt.title(VAR_STR[k] + " before fit")

		print("------")
		print(Xn[:,k])
		print("------")
		print(Yn[:,k])


		plt.figure()
		contour_plot(lambda x: (fast_target(x) - C1 @ x)[k], incl_f=True, NX=50, xi=2, yi=4)
		plt.title(VAR_STR[k] + " after fit")

		plt.figure()
		contour_plot(lambda x: (fast_target(x) - C2 @ x)[k], incl_f=True, NX=50, xi=2, yi=4)
		plt.title(VAR_STR[k] + " after fit")
		plt.show()