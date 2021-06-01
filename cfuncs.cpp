
#include <stdio.h>
#include <math.h>

double state0 = 0.0;
double state1 = 0.0;
double state2 = 0.0;
double state3 = 0.0;

double do_updates(double in_state0, double in_state1, double in_state2, double in_state3, double action) {

    double force = 20.*tanh(action/20.);
    double s, c, m, cart_accel, pole_accel;

    // printf("args: %f %f %f %f %f \n", in_state0, in_state1, in_state2, in_state3, action);

    state0 = in_state0;
    state1 = in_state1;
    state2 = in_state2;
    state3 = in_state3;

    for (int it = 0; it < 50; ++it) {
        s = sin(state2);
        c = cos(state2);
        m = 4.0-1.5*c*c;
            
        cart_accel = (.5*(state3*state3)*s+4*(force-.001*state1)-14.7*c*s+.012*state3*c)/m;
        pole_accel = (-1.5*c*s*(state3*state3)-12*force*c+.012*state1*c-.096*state3+117.6*s)/m;

        state1 += .004 * cart_accel;
        state3 += .004 * pole_accel;
        state2 += .004 * state3;
        state0 += .004 * state1;
    }
}

double get_state0() {
    return state0;
}

double get_state1() {
    return state1;
}

double get_state2() {
    return state2;
}

double get_state3() {
    return state3;
}