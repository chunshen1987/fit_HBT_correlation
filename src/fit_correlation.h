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

// for gsl fitting
struct Correlationfunction3D_data
{
  size_t data_length;
  double *q_o;
  double *q_s;
  double *q_l;
  double *y;
  double *sigma;
};

int Fittarget_correlfun3D_f_withlambda (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr);
int Fittarget_correlfun3D_df_withlambda (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr);
int Fittarget_correlfun3D_fdf_withlambda (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr);

/*****************************************************************************/
/*****************************************************************************/

class fit_correlation
{
    private:
       string filename;
       ParameterReader* paraRdr;
      
       int fit_mode;

       int qnpts;
       int nq_max;
       double q_fit_min;
       double q_max_1, q_max_2, dq_max;
       double *q_out, *q_side, *q_long;

       double fit_tolarence;
       int fit_max_iterations;

       double *Correlfun, *Correlfun_err;
      
       //HBT radii calculated from fitting correlation functions
       double lambda_Correl;
       double R_out_Correl, R_side_Correl, R_long_Correl;
       double R_os_Correl, R_sl_Correl, R_ol_Correl;
       double chi_sq_per_dof;

       double lambda_Correl_err;
       double R_out_Correl_err, R_side_Correl_err, R_long_Correl_err;
       double R_os_Correl_err, R_sl_Correl_err, R_ol_Correl_err;

       ofstream outputfile;

    public:
       fit_correlation(string filename_in, ParameterReader* paraRdr);
       ~fit_correlation();

       void fit();
       void read_in_correlation_functions();
       void output_fit_results(double q_fit);

       void find_minimum_chisq_correlationfunction_o_s_l(double q_fit);
       void find_minimum_chisq_correlationfunction_o_s_and_l_lambda_fixed(double q_fit);
       void find_minimum_chisq_correlationfunction_o_s_and_l(double q_fit);
       void find_minimum_chisq_correlationfunction_o_s_os_and_l(double q_fit);
       void find_minimum_chisq_correlationfunction_full(double q_fit);
       void find_minimum_chisq_correlationfunction_o_s_l_os(double q_fit);
       void find_minimum_chisq_correlationfunction_o_s_l_ol(double q_fit);

       // multi-dimensional fit with gsl
       void fit_Correlationfunction3D_withlambda_gsl(double q_fit);

       int print_fit_state_3D_withlambda_gsl (size_t iteration, gsl_multifit_fdfsolver * solver_ptr);
       inline double get_fit_results(int i, gsl_multifit_fdfsolver * solver_ptr);
       inline double get_fit_err (int i, gsl_matrix * covariance_ptr);

};

#endif
