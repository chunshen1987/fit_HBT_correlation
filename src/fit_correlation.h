#ifndef fit_correlation_h
#define fit_correlation_h

#include <gsl/gsl_vector.h>         // gsl vector and matrix definitions
#include <gsl/gsl_blas.h>           // gsl linear algebra stuff
#include <gsl/gsl_multifit_nlin.h>  // gsl multidimensional fitting

#include "parameters.h"
#include "arsenal.h"
#include "ParameterReader.h"

using namespace std;

class fit_correlation
{
    private:
       string filename;
       ParameterReader* paraRdr;
      
       int qnpts;
       int flag_1D;
       double q_max;
       double* q_out;
       double* q_side;
       double* q_long;

       double fit_tolarence;
       int fit_max_iterations;

       double* Correl_1D_out;
       double* Correl_1D_out_err;
       double* Correl_1D_side;
       double* Correl_1D_side_err;
       double* Correl_1D_long;
       double* Correl_1D_long_err;
       double*** Correl_3D;
       double*** Correl_3D_err;

    public:
       fit_correlation(string filename_in, ParameterReader* paraRdr);
       ~fit_correlation();

       void fit();
       void read_in_correlation_functions();

       void find_minimum_chisq_correlationfunction_1D();
       void find_minimum_chisq_correlationfunction_3D();

};

#endif
