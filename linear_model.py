from globals import *
from utils import *
from cartpole import *


def linear_training_data(N=512, t_step=0.2, sobol=True, incl_f=True, obs_noise=None, obs_bias=None, dyn_noise=None, dyn_bias=None):

	N_X_cmpts = 5 if incl_f else 4
	if obs_noise is None:
		obs_noise = np.zeros(N_X_cmpts)
	if obs_bias is None:
		obs_bias = np.zeros(N_X_cmpts)
	Y = np.zeros((N, 4))
	X = np.zeros((N, N_X_cmpts))

	if sobol:
		sobol_init()

	for i in range(N):
		if sobol:
			X[i,:] = 1.*sobol_rand_state5(TRAIN_P_RANGE5)[:N_X_cmpts]
		else:
			X[i,:] = 1.*rand_state5(TRAIN_P_RANGE5)[:N_X_cmpts]

		Y[i,:] = target(X[i, :], t_step)

	obs_noise = [.1, .2, .3, .4, .5]
	obs_bias = [5, 4, 3, 2, 1]

	# add observation noise/bias
	obs_noise = np.array(obs_noise)
	obs_bias = np.array(obs_bias)

	X += np.random.normal(loc=obs_bias, scale=obs_noise, size=(N, N_X_cmpts))
	Y += np.random.normal(loc=obs_bias[:4], scale=obs_noise[:4], size=(N, 4))

	return X, Y



print(linear_training_data(N=3, incl_f=True)[0])

exit()

# enforces lack of dependence on theta or x
def linear_fit(N=1024, t_step=0.2, sobol=True, get_saved=False, save=False, fname="lin_train", return_data=False, incl_f=True, enforce_constraints=True):
	
	if get_saved and not save:
		from_save = np.load("../" + fname + ".npy", allow_pickle=True)
		X, Y, C = from_save

		if len(X) == N and X.shape[1]//5 == incl_f: # AND MATCHING INCLUSION OF FORCE
			print("Retrieving saved data.")
			return X, Y, C
		else:
			print("Saved data does not have requested size!")

	X, Y = linear_training_data(N, t_step, sobol, incl_f=incl_f)
	C = np.zeros((4, X.shape[1]))

	nonzero_cols = (1,3)
	if incl_f:
		nonzero_cols += (4,)
	if not enforce_constraints:
		nonzero_cols = np.arange(X.shape[1])

	CT, res, rank, s = np.linalg.lstsq(X[:,nonzero_cols], Y, rcond=None)
	C[:,nonzero_cols] = CT.T

	if save:
		print("Saving data.")
		to_save = (X, Y, C)
		np.save("../saved/" + fname + ".npy", to_save)

	if return_data:
		return X, Y, C
	else:
		return C



# def linear_fit_2(N=512, t_step=0.2, sobol=True, get_saved=True, save=False, fname="lin_train", return_data=False, enforce_constraints=True):
	
# 	if get_saved and not save:
# 		from_save = np.load("../" + fname + ".npy", allow_pickle=True)
# 		X, Y, C = from_save

# 		if len(X) == N:
# 			print("Retrieving saved data.")
# 			return X, Y, C
# 		else:
# 			print("Saved data does not have requested size!")

# 	X, Y = linear_training_data(N, t_step, sobol)
# 	CT, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
# 	C = CT.T

# 	if save:
# 		print("Saving data.")
# 		to_save = (X, Y, C)
# 		np.save("../" + fname + ".npy", to_save)

# 	if return_data:
# 		return X, Y, C
# 	else:
# 		return C



def get_good_linear_fit(return_data=False, enforce_constraints=True, incl_f=True):
	if return_data:
		X, Y, C = linear_fit(N=2**12, return_data=True, enforce_constraints=enforce_constraints, incl_f=incl_f)
	else:
		C = linear_fit(N=2**12, return_data=False, enforce_constraints=enforce_constraints, incl_f=incl_f)

	def fn(x):
		return C @ x

	if return_data:
		return fn, X, Y
	else:
		return fn




def lin_model_convergence():
	
	X = np.power(10, np.linspace(1, 3.5, 100))
	C = np.zeros((len(X), 16))
	y = np.zeros(len(X))

	print(X)

	for i, n in enumerate(X):
		C = linear_fit(int(n))
		# print(C)
		y[i] = np.linalg.norm(C, "fro")
		print(i, y[i])

	print(X, y)

	plt.semilogx()
	# plt.gca().set_xticks(X)
	plt.scatter(X, y, marker="x")
	plt.plot([500, 500], [0, max(y)], "r--")
	plt.title("Frobenius norm of C for N\nrandomly drawn training states")
	plt.ylabel("Norm")
	plt.xlabel("N", labelpad=2.0)
	plt.show()


if __name__ == "__main__":

	X, Y, C = linear_fit(N=2048, return_data=True, get_saved=False, sobol=False, incl_f=True)
	print(C)

	for k in range(4):
		contour_plot(lambda x: (target(x))[k], incl_f=True, NX=50, xi=2, yi=4)
		plt.title(VAR_STR[k] + " before fit")
		plt.show()
		contour_plot(lambda x: (target(x) - C @ x)[k], incl_f=True, NX=50, xi=2, yi=4)
		plt.title(VAR_STR[k] + " after fit")
		plt.show()
