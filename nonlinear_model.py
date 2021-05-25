from globals import *
from utils import *
from linear_model import *
from cartpole import *


SQRT2 = np.sqrt(2)
INV_SQRT2 = 1/np.sqrt(2)

# Scalar version of kernel function
kfn_s4 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[3])*INV_SQRT2/s[3])
))
kfn_s5 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[3])*INV_SQRT2/s[3]) +
	np.square((x[4]-y[4])*INV_SQRT2/s[4])
))

# Vector version of kernel function
kfn4 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3])
))

kfn5 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3]) +
	np.square((x[4]-y[:,4])*INV_SQRT2/s[4])
))

# kfn1 = lambda x, y, s: np.exp(-(
# 	np.square((x[:,0]-y[:,0])*INV_SQRT2/s[0]) +
# 	np.square((x[:,1]-y[:,1])*INV_SQRT2/s[1]) +
# 	np.square(np.sin(.5*(x[:,2]-y[:,2]))*INV_SQRT2/s[2]) +
# 	np.square((x[:,3]-y[:,3])*INV_SQRT2/s[3])
# ))




def nonlinear_training_data(N=512, t_step=0.2, sobol=True, get_saved=True, save=False, fname="lin_train", incl_f=True):

	X, Y, C = linear_fit(N, t_step, sobol, get_saved, save, fname, return_data=True, incl_f=incl_f)

	# err = (Y.T - C @ X.T).T
	err = (Y - X @ C.T)

	# print(Y)
	# print(err)
	# print(np.linalg.norm(Y, "fro"))
	# print(np.linalg.norm(err, "fro"))

	return X, err, C


def nonlinear_fit(X, Y, reg, scales, M=256, t_step=0.2):

	N = len(X)

	λ = reg
	σ = scales

	K_MM = np.zeros((M, M))
	K_MN = np.zeros((M, N))

	if X.shape[1] == 4:
		kfn_s = kfn_s4
		kfn = kfn4
	else:
		kfn_s = kfn_s5
		kfn = kfn5

	for i in range(M):
		a = kfn(X[i,:], X[i+1:,:], σ)
		K_MN[i,i+1:] = a
	
	K_MN[:,:M] += K_MN[:,:M].T
	K_MN[:,:M] += np.identity(M)
	K_MM = K_MN[:,:M]

	b = K_MN @ Y
	A = K_MN @ K_MN.T + λ * K_MM

	αt, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
	α = αt.T

	def fit_fn(x):
		return (α @ kfn(x, X[:M,:], σ))

	return fit_fn


def fit_errors(X, Y, fit_fn, M):

	Y_est = np.array([fit_fn(x) for x in X])

	# print("est:\n", Y_est)
	# print("actual:\n", Y)
	# print("error:\n", Y - Y_est)
	errors = np.sum(np.abs(Y - Y_est), axis=1)
	# errors = (Y - Y_est)[:, 3]


	return np.std(errors), np.std(errors[:M]), np.std(errors[M:])

	print("total MSE:",  np.std(errors))
	print("training MSE:", np.std(errors[:M]))
	
	if len(X)>M:
		print("test MSE:", np.std(errors[M:]))
 



def get_good_nonlinear_fit(ret_full_fit=True, return_all=False, incl_f=True):
	print("fitting...")
	
	N = 2048
	M = 512
	X, Y, C = nonlinear_training_data(N, sobol=True, get_saved=False, incl_f=incl_f)

	perm = np.random.permutation(np.arange(N))
	centre_idxs = perm[:M]
	other_idxs = perm[M:]
	X = X[perm, :]
	Y = Y[perm, :]

	REG = 0.001
	SCALES =  [100, 100, 0.5, 20, 20]

	residual_fit = nonlinear_fit(X, Y, REG, SCALES, M)

	print("fitting complete.")

	if ret_full_fit:
		def full_fit(x):
			return residual_fit(x) + C @ x

		if return_all:
			return full_fit, X, Y, C
		else:
			return full_fit

	else:
		if return_all:
			return residual_fit, X, Y, C
		else:
			return residual_fit, C


def nonlinear_model_slice():

	# X, Y = linear_training_data(256, sobol=True)
	# print(X)

	# fig1 = plt.figure()
	# plt.tricontourf(X[:,2],X[:,3], Y[:,3])
	# plt.scatter(X[:,2], X[:,3], c="k", marker="x", s=10, linewidth=0.5)
	# plt.colorbar()
	# plt.show()

	bounds = P_RANGE.copy()
	bounds[2] = 2*bounds[2]
	fn1, X, Y, C = get_good_nonlinear_fit(ret_full_fit=False, return_all=True)

	# line_plot(lambda x: (target(x))[2], bounds=bounds, NX=100, xi=2)
	# line_plot(lambda x: (modified_lin(C, x))[2], bounds=bounds, NX=100, xi=2)
	# line_plot(lambda x: (target(x) - modified_lin(C, x))[2], bounds=bounds, NX=100, xi=2)
	# plt.show()
	# exit()

	fig2 = plt.figure()
	contour_plot(lambda x: fn1(x)[3], bounds=bounds, NX=100)
	plt.scatter(X[:,2], X[:,3], c="k", marker="x", s=10, linewidth=0.5)



	fig3 = plt.figure()
	contour_plot(lambda x: (target(x) - C @ x)[3], bounds=bounds, NX=100)
	plt.show()

# nonlinear_model_slice()


# fn1, X, Y, C = get_good_nonlinear_fit(ret_full_fit=False, return_all=True)


if __name__ == "__main__":
	#SYSTEMATIC reduction of error?

	# GOOD REGULARISATION VALUE: ABOUT 1e-2, 1e-3
	REG = 0.001

	idx = 2

	X, Y, C = nonlinear_training_data(1024, sobol=True, get_saved=False, incl_f=True)
	print(C)
	# X, Y, C = linear_training_data(512, get_saved=False, sobol=False, t_step=0.05)

	# print(np.std(X, axis=0))
	M = 50
	# print(M)

	scales = 0.05*P_RANGE5
	SCALES =  [100, 100, 0.5, 20, 20]

	SCAN_N = 50
	# scan = np.arange(10, 300)
	scan = np.power(10, np.linspace(-6, -1, SCAN_N))
	# scan = np.power(10, np.linspace(-2, 2, SCAN_N))
	# y = np.zeros((50, 3))
	y = np.zeros(SCAN_N)
	y1 = np.zeros(SCAN_N)
	y2 = np.zeros(SCAN_N)
	for i, s in enumerate(scan):
		REG = s
		# SCALES[idx] = s
		# M = int(s)
		print(s)
		fit_fn = nonlinear_fit(X, Y, REG, SCALES, M)
		tot, train, test = fit_errors(X, Y, fit_fn, M)
		# y[i, :] = [tot, test, train]
		y[i] = tot
		y1[i] = test
		y2[i] = train

	plt.plot(scan, y, "r", label=f"tot MSE vs component {idx}")
	plt.plot(scan, y1, "b", label=f"test MSE vs component {idx}")
	plt.plot(scan, y2, "g", label=f"train MSE vs component {idx}")
	plt.semilogx()
	plt.legend()
	plt.show()

	exit()


	# Qs... 
	# Regularisation.
	# - becomes more 'unstable' with smaller lambda. Numerically.. how does this affect actual answer?
	# Scale optimum point reduces with increasing N?

	# res = scipy.optimize.minimize_scalar(f, bounds=[-, 2], method="bounded")
	# print(res)

	# hyperparams = np.linspace()


	# for M in [10, 20, 40, 80, 160, 320, 480]:
	# 	nonlinear_fit(X, Y, reg, scales, M)



	# exit()
