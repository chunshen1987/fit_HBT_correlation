#ifndef fit_correlation_h
#define fit_correlation_h

#include <gsl/gsl_rng.h>            // gsl random number generators
#include <gsl/gsl_randist.h>        // gsl random number distributions
#include <gsl/gsl_vector.h>         // gsl vector and matrix definitions
#include <gsl/gsl_blas.h>           // gsl linear algebra stuff
#include <gsl/gsl_multifit_nlin.h>  // gsl multidimensional fitting

#include "parameters.h"
#include "arsenal.h"
#include "ParameterReader.h"

using namespace std;

struct Correlationfunction3D_data
{
  size_t data_length;
  double *q_o;
  double *q_s;
  double *q_l;
  double *y;
  double *sigma;
};

struct Correlationfunction1D_data
{
  size_t data_length;
  double *q;
  double *y;
  double *sigma;
};

int Fittarget_correlfun1D_f (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr);
int Fittarget_correlfun1D_df (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr);
int Fittarget_correlfun1D_fdf (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr);
int Fittarget_correlfun3D_f (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr);
int Fittarget_correlfun3D_df (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr);
int Fittarget_correlfun3D_fdf (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr);
int Fittarget_correlfun3D_f_withlambda (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr);
int Fittarget_correlfun3D_df_withlambda (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr);
int Fittarget_correlfun3D_fdf_withlambda (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr);

class fit_correlation
{
    private:
       string filename;
       ParameterReader* paraRdr;
      
       int flag_1D;
       int flag_gsl_fit;

       int qnpts;
       double q_max;
       double *q_out, *q_side, *q_long;

       double fit_tolarence;
       int fit_max_iterations;

       double *Correlfun, *Correlfun_err;
      
       //HBT radii calculated from fitting correlation functions
       double lambda_Correl;
       double R_out_Correl;
       double R_side_Correl;
       double R_long_Correl;
       double R_os_Correl;
       double lambda_Correl_err;
       double R_out_Correl_err;
       double R_side_Correl_err;
       double R_long_Correl_err;
       double R_os_Correl_err;

    public:
       fit_correlation(string filename_in, ParameterReader* paraRdr);
       ~fit_correlation();

       void fit();
       void read_in_correlation_functions();

       void find_minimum_chisq_correlationfunction_o_s_l();
       void find_minimum_chisq_correlationfunction_o_s_and_l();
       void find_minimum_chisq_correlationfunction_o_s_os_and_l();
       void find_minimum_chisq_correlationfunction_full();
       void find_minimum_chisq_correlationfunction_o_s_l_os();

       // multi-dimensional fit with gsl
       void fit_Correlationfunction3D_withlambda_gsl();

       int print_fit_state_3D_withlambda_gsl (size_t iteration, gsl_multifit_fdfsolver * solver_ptr);
       inline double get_fit_results(int i, gsl_multifit_fdfsolver * solver_ptr);
       inline double get_fit_err (int i, gsl_matrix * covariance_ptr);

};

#endif
