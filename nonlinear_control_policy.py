from globals import *
from utils import *
# only angle variables cared about.

MAX_F = 30


def get_nonlin_policy(param_obj):

	basis_fn_centres, W_elems, w_vector = param_obj

	W_matrix = [[W_elems[0], W_elems[4], W_elems[7], W_elems[9]],
				[W_elems[4], W_elems[1], W_elems[5], W_elems[8]],
				[W_elems[7], W_elems[5], W_elems[2], W_elems[6]],
				[W_elems[9], W_elems[8], W_elems[6], W_elems[3]]]
	W_matrix = np.array([np.array(x) for x in W_matrix])

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	# print("W:")
	# print(W_matrix)

	# print("weights:")
	# print(w_vector)

	# print("Xi:")
	# print(basis_fn_centres)	

		# print("X-X_i")
		# print(NONLIN_TEMP)
		# print("MAT @ (X-X_i)")
		# print((W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0])
		# print("quad form", np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1))
		# print("overall", np.dot(w_vector, np.exp(-0.5 * np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1))))

	NONLIN_TEMP = np.zeros((len(w_vector), 4))
	SQRT_PI = 1.77245385091

	remap = True

	if remap:
		def p(x, it=None):
			NONLIN_TEMP[:,:] = remapped_angle(x) - basis_fn_centres 
			return np.dot(w_vector, np.exp(-0.5 * np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1)))
	else:
		def p(x, it=None):
			NONLIN_TEMP[:,:] = x - basis_fn_centres 
			NONLIN_TEMP[:,2] = SQRT_PI*np.sin(.5*NONLIN_TEMP[:,2])
			return np.dot(w_vector, np.exp(-0.5 * np.sum(NONLIN_TEMP * (W_matrix @ NONLIN_TEMP[:, :, np.newaxis])[:,:,0], axis=1)))	
	return p




# getting proper vectors to replicate linear near origin

P_v = np.array([20, 2.1])
n = np.linalg.norm(P_v)
p_v = P_v/n
q_v = np.array([-p_v[1], p_v[0]])

q_vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, p_v[0], p_v[1]], [0, 0, q_v[0], q_v[1]]]
global_Q = np.vstack(q_vectors)
def transform_Q(x):
	return global_Q.T @ x
def transform_Qinv(x):
	return global_Q.T @ x


def get_M(P_scale, Q_scale):
	mat = np.array([[1/(P_scale**2), 0], [0, 1/(Q_scale**2)]])
	V = np.vstack([p_v, q_v])
	return V.T @ mat @ V



def get_nonlin_policy2(param_obj):

	basis_fn_centres, eig_scales, q_vectors, w_vector = param_obj

	Q_matrix = np.vstack(q_vectors)

	eig_scale_vec = 1/(np.array(eig_scales)**2)

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])

	NONLIN_TEMP = np.zeros((len(w_vector), 4))
	SQRT_PI = 1.77245385091
	def p(x, it=None):
		NONLIN_TEMP[:,:] = x - basis_fn_centres 
		NONLIN_TEMP[:,:] = (Q_matrix @ NONLIN_TEMP[:,:, np.newaxis])[:,:,0]
		NONLIN_TEMP[:,2] = SQRT_PI*np.sin(.5*NONLIN_TEMP[:,2])
		return np.dot(w_vector, np.exp(-0.5 *  np.sum(NONLIN_TEMP * (eig_scale_vec[np.newaxis,:] * NONLIN_TEMP), axis=1)))

	return p




def get_nonlin_policy3(param_obj):

	eig_scales, basis_fn_centres, w_vector = param_obj

	Q_matrix = np.vstack(q_vectors)


	eig_scale_vec = 1/(np.array(eig_scales)**2)

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])

	NONLIN_TEMP = np.zeros((len(w_vector), 4))
	SQRT_PI = 1.77245385091
	def p(x, it=None):
		NONLIN_TEMP[:,:] = x - basis_fn_centres 
		NONLIN_TEMP[:,:] = (Q_matrix @ NONLIN_TEMP[:,:, np.newaxis])[:,:,0]
		NONLIN_TEMP[:,2] = SQRT_PI*np.sin(.5*NONLIN_TEMP[:,2])
		return np.dot(w_vector, np.exp(-0.5 *  np.sum(NONLIN_TEMP * (eig_scale_vec[np.newaxis,:] * NONLIN_TEMP), axis=1)))

	return p


M = get_M(2, 10)

basis_fn_centres = [[0,0,2,0], [0,0,-2,0]]
W_elems = [0 ,0, M[0,0], M[1,1], 0, 0, M[0,1], 0, 0, 0]
w_vector = [1,-1]

def p1(x1, x2, x3):
	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))

def p2(x1, x2, x3):
	return get_nonlin_policy2((basis_fn_centres, [1, 1, 2, 10], [np.zeros(4), np.zeros(4), [0, 0, p_v[0], p_v[1]], [0, 0, q_v[0], q_v[1]]], [1, -1]))




SQRT_PI = 1.77245385091
def periodicitise(x):
	return np.array(x[0], x[1], SQRT_PI*np.sin(0.5*x[2]), x[3])



def reparametrise(policy, parametrisation):
	return lambda x: policy(parametrisation(x))



def parametrisation_A(overall_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w):

	basis_fn_centres = [[0,0,0, -a_pos], [0,0,0, a_pos], [0,0, np.pi, -b_pos], [0,0, np.pi, b_pos]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [a_w, -a_w, b_w, -b_w]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = overall_weight*MAX_F*np.array(w_vector)

	return basis_fn_centres, W_elems, w_vector


def get_nonlin_policy_A(*x):
	return get_nonlin_policy(parametrisation_A(1., *x))



A_optimal = [ 1.61655299,  6.63370857, 14.20361603,  7.11556684,  -1.0131851,  1.02112302]

def type_A_terms(A_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w):

	basis_fn_centres = [[0,0,0, -a_pos], [0,0,0, a_pos], [0,0, np.pi, -b_pos], [0,0, np.pi, b_pos]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [a_w, -a_w, b_w, -b_w]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = A_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))



def type_B_term(B_weight, theta_scale, theta_d_scale, B_fraction):
	
	basis_fn_centres = [[0,0, np.pi, 0]]
	W_elems = [0 ,0, 1/((theta_scale*B_fraction)**2), 1/((theta_d_scale*B_fraction)**2), 0, 0, 0, 0, 0, 0]
	w_vector = [B_weight]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = B_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))




def type_C_terms(C_weight, C_pos, C_perp_scale):
		
	M = get_M(1, C_perp_scale)


	temp = C_pos*p_v
	basis_fn_centres = [[0,0, temp[0], temp[1]], [0,0,-temp[0], -temp[1]]]
	# W_elems = [0 ,0, .1, .1, 0, 0, 0, 0, 0, 0]
	# W_elems = [0 ,0, 1/(pos**2), 1/(pos**2), 0, 0, 0, 0, 0, 0]
	W_elems = [0 ,0, M[0,0]/C_pos, M[1,1]/C_pos, 0, 0, M[0,1]/C_pos, 0, 0, 0]
	# W_elems = [0 ,0, 1/((theta_scale*C_fraction)**2), 1/((theta_d_scale*C_fraction)**2), 0, 0, 0, 0, 0, 0]
	w_vector = [C_weight*MAX_F, -C_weight*MAX_F]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))


def add_policies(*ps):
	return lambda x,it=None: np.sum([p(x,it) for p in ps])



def get_nonlin_policy_B(A_weight, B_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w, B_fraction):

	p1 = type_A_terms(A_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w)
	p2 = type_B_term(B_weight, theta_scale, theta_d_scale, B_fraction)

	return lambda x, it=None: p1(x, it) + p2(x, it)

def get_nonlin_policy_B1(theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w):
	return get_nonlin_policy_B(1., 5, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w, 0.1)


def get_nonlin_policy_C(A_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w, C_weight, C_pos, C_perp_scale):

	pa = type_A_terms(A_weight, theta_scale, theta_d_scale, a_pos, b_pos, a_w, b_w)
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	return lambda x, it=None: pa(x, it) + pc(x, it)



def get_nonlin_policy_C1(C_weight, C_pos, C_perp_scale):
	return get_nonlin_policy_C(*([1.] + A_optimal + [C_weight, C_pos, C_perp_scale]))


def get_nonlin_policy_C2(C_weight, C_pos, C_perp_scale):
	pb = type_B_term(1, 1, 7, 0.4)
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	return lambda x, it=None: pb(x, it) + pc(x, it)


def get_nonlin_policy_C3(C_weight, C_pos, C_perp_scale):
	pb = type_B_term(1, 1, 7, 0.4)
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	pd = type_D_terms(1, .5, 3.5, 14, 20)
	return lambda x, it=None: pb(x, it) + pc(x, it) + pd(x, it)


def get_nonlin_policy_C4(C_weight, C_pos, C_perp_scale):
	pb = type_B_term(1, 1, 7, 0.4)
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	pd = type_D_terms(1, .5, 7, 15, 50)
	# pd = type_F_terms(1, .5, 3.5, np.pi/2, 15)
	return lambda x, it=None: pb(x, it) + pc(x, it) + pd(x, it)


def get_nonlin_policy_C5(C_weight, C_pos, C_perp_scale, B_weight, theta_scale, theta_d_scale, H_weight, H_scale1, H_scale2, H_pos):
	pb = type_B_term(B_weight, theta_scale, theta_d_scale, 0.2)
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	# pg = type_G_terms(G_weight, .5*theta_scale, theta_d_scale, G_pos)
	ph = type_H_terms(H_weight, H_scale1, H_scale2, H_pos)
	# pi = type_I_terms(1, 0, 1, .2*theta_scale, theta_d_scale, 15, 20, 20)

	return add_policies(pb, pc, ph)








def type_D_terms(D_weight, theta_scale, theta_d_scale, D_pos1, D_pos2):

	basis_fn_centres = [[0,0,0, -D_pos1], [0,0,0, D_pos1], 
						[0,0, np.pi, -D_pos2], [0,0, np.pi, D_pos2]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [-1, 1, 1, -1]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = D_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))






def get_nonlin_policy_D1(C_weight, C_pos, C_perp_scale, D_weight, theta_scale, theta_d_scale, D_pos):
	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	pd = type_D_terms(D_weight, theta_scale, theta_d_scale, D_pos, D_pos)

	return lambda x, it=None: pc(x, it) + pd(x, it)




def type_E_terms(E_weight, theta_scale, theta_d_scale, E_pos1, E_pos2):

	f1 = 1/np.tan(np.pi/6)

	basis_fn_centres = [[0,0,  f1*np.pi, E_pos1], [0,0,  f1*np.pi, -E_pos1], 
						[0,0, -f1*np.pi, E_pos1], [0,0, -f1*np.pi, -E_pos1],
						[0,0,       0, np.linalg.norm([f1*np.pi, E_pos1])], [0,0,     0,   -np.linalg.norm([f1*np.pi, E_pos1])],
						[0,0,     np.pi, E_pos2], [0,0,     np.pi, -E_pos2]]

	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [-1, 1, -1, 1, 1, -1, 1, -1]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = E_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))


def get_nonlin_policy_E1(D_weight, D_pos1, D_pos2, E_weight, E_pos1, E_pos2, theta_scale, theta_d_scale):
	# theta_scale = .5
	# theta_d_scale = 2.5

	pd = type_D_terms(D_weight, theta_scale, theta_d_scale, D_pos1, D_pos2)
	pe = type_E_terms(E_weight, theta_scale, theta_d_scale, E_pos1, E_pos2)

	return lambda x, it=None: pd(x, it) + pe(x, it)


def get_nonlin_policy_E2(C_weight, C_pos, C_perp_scale, D_weight, D_pos1, D_pos2, E_weight, E_pos1, E_pos2):
	theta_scale = .5
	theta_d_scale = 7

	pc = type_C_terms(C_weight, C_pos, C_perp_scale)
	pd = type_D_terms(D_weight, theta_scale, theta_d_scale, D_pos1, D_pos2)
	pe = type_E_terms(E_weight, theta_scale, theta_d_scale, E_pos1, E_pos2)

	return lambda x, it=None: pc(x, it) + pd(x, it) + pe(x, it)

def get_nonlin_policy_E3(C_weight, C_pos, C_perp_scale):
	return get_nonlin_policy_E2(*[C_weight, C_pos, C_perp_scale] +  [1.2, 17, 20, 1.2, 1.5, 7] )



def type_F_terms(F_weight, theta_scale, theta_d_scale, F_pos1, F_pos2):

	basis_fn_centres = [[0,0 ,F_pos1, F_pos2], [0,0, F_pos1, -F_pos2], 
						[0,0,-F_pos1, F_pos2], [0,0,-F_pos1, -F_pos2]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [1, -1, 1, -1]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = F_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))







def type_G_terms(G_weight, theta_scale, theta_d_scale, G_pos):

	basis_fn_centres = [[0,0,0, G_pos], [0,0,0, -G_pos]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [1, -1]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = G_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))



def get_bfc(p_coeff, q_coeff):
	vec = p_coeff*p_v + q_coeff*q_v
	return np.array([0,0, vec[0], vec[1]])



H_OPT = [5.51, 0.61, 3.64, 0.63]
def type_H_terms(H_weight, theta_scale, theta_d_scale, H_pos):

	basis_fn_centres = [get_bfc(0, H_pos) + [0,0,np.pi,0],
						get_bfc(0, -H_pos) + [0,0,np.pi,0]]
	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]
	w_vector = [1, -1]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = H_weight*MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))






def type_I_terms(I_weight1, I_weight2, I_weight3, theta_scale, theta_d_scale, I_h1, I_h2, I_d1):

	basis_fn_centres = [[0, 0, 0, I_h1],     [0, 0, 0, -I_h1],
						[0, 0, I_d1, I_h2],  [0, 0, -I_d1, -I_h2],
						[0, 0, -I_d1, I_h2], [0, 0, I_d1, -I_h2]]


	W_elems = [0 ,0, 1/(theta_scale**2), 1/(theta_d_scale**2), 0, 0, 0, 0, 0, 0]

	w_vector = [I_weight1, -I_weight1, I_weight2, -I_weight2, I_weight3, -I_weight3]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = MAX_F*np.array(w_vector)

	return get_nonlin_policy((basis_fn_centres, W_elems, w_vector))



def pad_zeros1(x):
	return [0, 0, x[0], x[1]]



def policy_from_min1(P_scale, Q_scale, d1, w1, w2=0, d3=0, w3=0):


	basis_fn_centres = [[0, 0, d1, 0], [0, 0, -d1, 0],
						transform_Qinv([0, 0, np.pi, 0])]
	w_vector = [w1*d1, -w1*d1, w2, w3, -w3]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = MAX_F*np.array(w_vector)

	return get_nonlin_policy3(([1, 1, P_scale, Q_scale], basis_fn_centres, w_vector))



def full_policy1(P_scale, Q_scale, w1, dp2, dq2, w2, w3):#, d3=0, w3=0):

	basis_fn_centres = [transform_Q([0, 0, w1, 0]), transform_Q([0, 0, -w1, 0]),
						transform_Q([0, 0, dp2, dq2]) + [0,0,np.pi,0], transform_Q([0, 0, -dp2, -dq2]) + [0,0,np.pi,0]]#, transform_Q([0, 0, dp2, -dq2]), transform_Q([0, 0, -dp2, dq2])]
						# transform_Qinv([0, 0, np.pi, 0])],
						# transform_Qinv([0, 0, np.pi, d3]), transform_Qinv([0, 0, np.pi, -d3])]
	w_vector = [.2, -.2, w2, -w2]#, -w3]

	basis_fn_centres = np.array([np.array(x) for x in basis_fn_centres])
	w_vector = MAX_F*np.array(w_vector)


	return get_nonlin_policy3(([1, 1, P_scale, Q_scale], basis_fn_centres, w_vector))
