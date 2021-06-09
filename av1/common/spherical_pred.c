#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "av1/common/spherical_pred.h"
#include "av1/common/common.h"

void equ_to_plane(double phi, double theta, double* x, double* y, int width, int height){
    // The 0.01 is for issues when comparing floating point numbers
    // Will find a better way to do it.
    assert(phi >= -PI/2 - 0.01 && phi <= PI/2 + 0.01 && 
        theta >= -PI - 0.01 && theta <= PI + 0.01);

    // This should actually be a range related to 1/cos(phi), will adjust later
    *x = theta/PI * (double)width/2 + width/2; 
    *y = -phi/PI/2 * (double)height/2 + height/2;
}

void plane_to_equ(int x, int y, double *phi, double *theta, int width, int height){
    assert(x <= width && y <= height);
    
    *theta = (x - width/2) / (double)width * 2 * PI;
    *phi = (y - height/2) / (double)height * PI * -1;
}