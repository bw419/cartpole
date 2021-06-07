
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float state0 = 0.0;
float state1 = 0.0;
float state2 = 0.0;
float state3 = 0.0;

float noise_std_dev = 0.0;


void seed_random_generator() {
/*    generator.seed(std::random_device{}()); */
}

void set_noise(float sigma) {
	noise_std_dev = sigma;
}


float randn() {
	if (noise_std_dev == 0.0) {
		return 0.0;
	}

	float U1, U2, W, mult;
	static float X1, X2;
	static int call = 0;

	if (call == 1)
	{
	  call = !call;
	  return noise_std_dev * (float) X2;
	}

	do
	{
	  U1 = -1 + ((float) rand () / RAND_MAX) * 2;
	  U2 = -1 + ((float) rand () / RAND_MAX) * 2;
	  W = pow (U1, 2) + pow (U2, 2);
	}
	while (W >= 1 || W == 0);

	mult = sqrt ((-2 * log (W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return noise_std_dev * (float) X1;
}


void do_updates(float in_state0, float in_state1, float in_state2, float in_state3, float action) {


	float force = 20.*tanh(action/20.);
	float s, c, m, cart_accel, pole_accel;

	// printf("args: %f %f %f %f %f \n", in_state0, in_state1, in_state2, in_state3, action);

	state0 = in_state0;
	state1 = in_state1;
	state2 = in_state2;
	state3 = in_state3;

	for (int it = 0; it < 50; ++it) {
		s = sin(state2);
		c = cos(state2);
		m = 4.0-1.5*c*c;
			
		cart_accel = (.5*(state3*state3)*s+4*(force-.001*state1)-14.7*c*s+.012*state3*c)/m + 16.0 * randn();
		pole_accel = (-1.5*c*s*(state3*state3)-12*force*c+.012*state1*c-.096*state3+117.6*s)/m + 41.6 * randn();

		state1 += .004 * cart_accel;
		state3 += .004 * pole_accel;
		state2 += .004 * state3;
		state0 += .004 * state1;
	}
}

float get_state0() {
	return state0;
}

float get_state1() {
	return state1;
}

float get_state2() {
	return state2;
}

float get_state3() {
	return state3;
}






///////////////////////////////////////////////////////////////////
// TESTED THIS, AND IT'S ONLY ABOUT 3x FASTER THAN NUMPY
// SO NOT WORTH THE DEBUGGING EFFORT TO USE IT
///////////////////////////////////////////////////////////////////
// float nonlin_model_sigmas[5] = { 1, 1, 1, 1, 1 };
// float nonlin_model_centres[50000] = {};
// float nonlin_model_weights[40000] = {1};

// void set_nonlin_model_sigmas(float sigma1, float sigma2, float sigma3, float sigma4, float sigma5) {
//     nonlin_model_sigmas[0] = sigma1;
//     nonlin_model_sigmas[1] = sigma2;
//     nonlin_model_sigmas[2] = sigma3;
//     nonlin_model_sigmas[3] = sigma4;
//     nonlin_model_sigmas[4] = sigma5;
// }


// void set_nonlin_model_basis_fn(int i, float x1, float x2, float x3, float x4, float x5, float weight1, float weight2, float weight3, float weight4) {
//     nonlin_model_centres[5*i + 0] = x1;
//     nonlin_model_centres[5*i + 1] = x2;
//     nonlin_model_centres[5*i + 2] = x3;
//     nonlin_model_centres[5*i + 3] = x4;
//     nonlin_model_centres[5*i + 4] = x5;
//     nonlin_model_weights[4*i + 0] = weight1;
//     nonlin_model_weights[4*i + 1] = weight2;
//     nonlin_model_weights[4*i + 2] = weight3;
//     nonlin_model_weights[4*i + 3] = weight4;
// }



// float evaluate_kernel_fn(int i, float x1, float x2, float x3, float x4, float x5) {
//     return  exp((x1 - nonlin_model_centres[5*i + 0])/(2*nonlin_model_sigmas[0])
//      + (x2 - nonlin_model_centres[5*i + 1])/(2*nonlin_model_sigmas[1])
//      + (x3 - nonlin_model_centres[5*i + 2])/(2*nonlin_model_sigmas[2])
//      + (x4 - nonlin_model_centres[5*i + 3])/(2*nonlin_model_sigmas[3])
//      + (x5 - nonlin_model_centres[5*i + 4])/(2*nonlin_model_sigmas[4]));

// }


// float y[4] = {};

// void calculate_nonlin_model(int N, float x1, float x2, float x3, float x4, float x5) {
//     for (int i = 0; i < 4; ++i) {
//         y[i] = 0.0;
//     }

//     float kfn_val = 0.0;
//     for (int i = 0; i < N; ++i) {
//         kfn_val = evaluate_kernel_fn(i, x1, x2, x3, x4, x5);
//         for (int j = 0; j < 4; ++j) {
//            y[i] += nonlin_model_weights[4*i + j] * kfn_val;
//         }
//     }
// }



// float evaluate_nonlin_state1() {
//     return y[0];
// }

// float evaluate_nonlin_state2() {
//     return y[1];
// }

// float evaluate_nonlin_state3() {
//     return y[2];
// }

// float evaluate_nonlin_state4() {
//     return y[3];
// }