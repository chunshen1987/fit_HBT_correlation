#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<cmath>
#include<iomanip>
#include <gsl/gsl_vector.h>         // gsl vector and matrix definitions
#include <gsl/gsl_blas.h>           // gsl linear algebra stuff
#include <gsl/gsl_multifit_nlin.h>  // gsl multidimensional fitting

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

#include "arsenal.h"
#include "ParameterReader.h"
#include "fit_correlation.h"

using namespace std;

fit_correlation::fit_correlation(string filename_in, ParameterReader* paraRdr_in)
{
    filename = filename_in;
    paraRdr = paraRdr_in;

    qnpts = paraRdr->getVal("qnpts");
    q_max = paraRdr->getVal("q_max");
    flag_1D = paraRdr->getVal("flag_1D");

    q_out = new double [qnpts];
    q_side = new double [qnpts];
    q_long = new double [qnpts];

    Correl_1D_out = new double [qnpts];
    Correl_1D_out_err = new double [qnpts];
    Correl_1D_side = new double [qnpts];
    Correl_1D_side_err = new double [qnpts];
    Correl_1D_long = new double [qnpts];
    Correl_1D_long_err = new double [qnpts];

    Correl_3D = new double ** [qnpts];
    Correl_3D_err = new double ** [qnpts];
    for(int i = 0; i < qnpts; i++)
    {
        Correl_3D[i] = new double * [qnpts];
        Correl_3D_err[i] = new double * [qnpts];
        for(int j = 0; j < qnpts; j++)
        {
            Correl_3D[i][j] = new double [qnpts];
            Correl_3D_err[i][j] = new double [qnpts];
        }
    }
}

fit_correlation::~fit_correlation()
{
    delete [] q_out;
    delete [] q_side;
    delete [] q_long;

    delete [] Correl_1D_out;
    delete [] Correl_1D_out_err;
    delete [] Correl_1D_side;
    delete [] Correl_1D_side_err;
    delete [] Correl_1D_long;
    delete [] Correl_1D_long_err;

    for(int i = 0; i < qnpts; i++)
    {
        for(int j = 0; j < qnpts; j++)
        {
            delete [] Correl_3D[i][j];
            delete [] Correl_3D_err[i][j];
        }
        delete [] Correl_3D[i];
        delete [] Correl_3D_err[i];
    }
    delete [] Correl_3D;
    delete [] Correl_3D_err;
}

void fit_correlation::fit()
{
    read_in_correlation_functions();
    if(flag_1D == 1)
        find_minimum_chisq_correlationfunction_1D();
    else
        find_minimum_chisq_correlationfunction_3D();
}

void fit_correlation::read_in_correlation_functions()
{
    ifstream data(filename);
    string temp;
    if(flag_1D == 1)
    {
        for(int i = 0; i < qnpts; i++)
        {
            getline(data, temp);
            stringstream temp_stream(temp);
            temp_stream >> q_out[i] >> Correl_1D_out[i] >> Correl_1D_out_err[i]
                        >> q_side[i] >> Correl_1D_side[i] >> Correl_1D_side_err[i]
                        >> q_long[i] >> Correl_1D_long[i] >> Correl_1D_long_err[i];
        }
    }
    else
    {
        for(int i = 0; i < qnpts; i++)
        {
            for(int j = 0; j < qnpts; j++)
            {
                for(int k = 0; k < qnpts; k++)
                {
                    getline(data, temp);
                    stringstream temp_stream(temp);
                    temp_stream >> q_out[i] >> q_side[j] >> q_long[k] >> Correl_3D[i][j][k] >> Correl_3D_err[i][j][k];
                }
            }
        }
    }
}

void fit_correlation::find_minimum_chisq_correlationfunction_1D()
{
    double lambda, R_HBT;
    double q_local;
    double Correl_local;
    double sigma_k_prime;
    double chi_sq;
    // 0 for out, 1 for side, and 2 for long
    for(int idir = 0; idir < 3; idir++)
    {
        double X0_coeff = 0.0;
        double X2_coeff = 0.0;
        double Y0_coeff = 0.0;
        double Y2_coeff = 0.0;
        double Y4_coeff = 0.0;
        for(int iq = 0; iq < qnpts; iq++)
        {
           if(idir == 0)
           {
               q_local = q_out[iq];
               Correl_local = Correl_1D_out[iq];
               sigma_k_prime = Correl_1D_out_err[iq]/Correl_local;
           }
           else if (idir == 1)
           {
               q_local = q_side[iq];
               Correl_local = Correl_1D_side[iq];
               sigma_k_prime = Correl_1D_side_err[iq]/Correl_local;
           }
           else if (idir == 2)
           {
               q_local = q_long[iq];
               Correl_local = Correl_1D_long[iq];
               sigma_k_prime = Correl_1D_long_err[iq]/Correl_local;
           }
           double denorm = sigma_k_prime*sigma_k_prime;
           double q_sq = q_local*q_local;
           X0_coeff += log(Correl_local)/denorm;
           X2_coeff += q_sq*log(Correl_local)/denorm;
           Y0_coeff += 1./denorm;
           Y2_coeff += q_sq/denorm;
           Y4_coeff += q_sq*q_sq/denorm;
        }
        lambda = exp((X2_coeff*Y2_coeff - X0_coeff*Y4_coeff)/(Y2_coeff*Y2_coeff - Y0_coeff*Y4_coeff));
        R_HBT = sqrt((X2_coeff*Y0_coeff - X0_coeff*Y2_coeff)/(Y2_coeff*Y2_coeff - Y0_coeff*Y4_coeff))*hbarC;
        
        // compute chi square
        chi_sq = 0.0;
        for(int iq = 0; iq < qnpts; iq++)
        {
           if(idir == 0)
           {
               q_local = q_out[iq];
               Correl_local = Correl_1D_out[iq];
               sigma_k_prime = Correl_1D_out_err[iq]/Correl_local;
           }
           else if (idir == 1)
           {
               q_local = q_side[iq];
               Correl_local = Correl_1D_side[iq];
               sigma_k_prime = Correl_1D_side_err[iq]/Correl_local;
           }
           else if (idir == 2)
           {
               q_local = q_long[iq];
               Correl_local = Correl_1D_long[iq];
               sigma_k_prime = Correl_1D_long_err[iq]/Correl_local;
           }
           chi_sq += pow((log(Correl_local) - log(lambda) + R_HBT*R_HBT/hbarC/hbarC*q_local*q_local), 2)/sigma_k_prime/sigma_k_prime;
        }
        cout << "lambda = " << lambda << ", R = " << R_HBT << " fm." << endl;
        cout << "chi_sq/d.o.f = " << chi_sq/qnpts << endl;
    }
}

void fit_correlation::find_minimum_chisq_correlationfunction_3D()
{
    double lambda, R_o, R_s, R_l, R_os, R_ol, R_sl;
    int dim = 7;
    int s_gsl;

    double *V = new double [dim];
    double *qweight = new double [dim];
    double **T = new double* [dim];
    for(int i = 0; i < dim; i++)
    {
        V[i] = 0.0;
        T[i] = new double [dim];
        for(int j = 0; j < dim; j++)
            T[i][j] = 0.0;
    }

    gsl_matrix * T_gsl = gsl_matrix_alloc (dim, dim);
    gsl_matrix * T_inverse_gsl = gsl_matrix_alloc (dim, dim);
    gsl_permutation * perm = gsl_permutation_alloc (dim);

    for(int iqout = 0; iqout < qnpts; iqout++)
    {
        double q_out_local = q_out[iqout];
        for(int iqside = 0; iqside < qnpts; iqside++)
        {
            double q_side_local = q_side[iqside];
            for(int iqlong = 0; iqlong < qnpts; iqlong++)
            {
                double q_long_local = q_long[iqlong];
                double correl_local = Correl_3D[iqout][iqside][iqlong];
                double sigma_k_prime = Correl_3D_err[iqout][iqside][iqlong]/correl_local;
                
                double inv_sigma_k_prime_sq = 1./sigma_k_prime*sigma_k_prime;
                double log_correl_over_sigma_sq = log(correl_local)*inv_sigma_k_prime_sq;

                qweight[0] = - 1.0;
                qweight[1] = q_out_local*q_out_local;
                qweight[2] = q_side_local*q_side_local;
                qweight[3] = q_long_local*q_long_local;
                qweight[4] = q_out_local*q_side_local;
                qweight[5] = q_out_local*q_long_local;
                qweight[6] = q_side_local*q_long_local;

                for(int ij = 0; ij < 7; ij++)
                {
                    V[ij] += qweight[ij]*log_correl_over_sigma_sq;
                    T[0][ij] += qweight[ij]*inv_sigma_k_prime_sq;
                    T[ij][0] += qweight[ij]*inv_sigma_k_prime_sq;
                }

                for(int ij = 1; ij < 7; ij++)
                {
                    for(int lm = 1; lm < 7; lm++)
                        T[ij][lm] += -qweight[ij]*qweight[lm]*inv_sigma_k_prime_sq;
                }
            }
        }
    }
    for(int i = 0; i < dim; i++)
        for(int j = 0; j < dim; j++)
            gsl_matrix_set(T_gsl, i, j, T[i][j]);

    // Make LU decomposition of matrix T_gsl
    gsl_linalg_LU_decomp (T_gsl, perm, &s_gsl);
    // Invert the matrix m
    gsl_linalg_LU_invert (T_gsl, perm, T_inverse_gsl);

    double **T_inverse = new double* [dim];
    for(int i = 0; i < dim; i++)
    {
        T_inverse[i] = new double [dim];
        for(int j = 0; j < dim; j++)
            T_inverse[i][j] = gsl_matrix_get(T_inverse_gsl, i, j);
    }
    double *results = new double [dim];
    for(int i = 0; i < dim; i++)
    {
        results[i] = 0.0;
        for(int j = 0; j < dim; j++)
            results[i] += T_inverse[i][j]*V[j];
    }

    lambda = exp(results[0]);
    R_o = sqrt(results[1])*hbarC;
    R_s = sqrt(results[2])*hbarC;
    R_l = sqrt(results[3])*hbarC;
    R_os = sqrt(results[4])*hbarC;
    R_ol = sqrt(results[5])*hbarC;
    R_sl = sqrt(results[6])*hbarC;
    cout << "lambda = " << lambda << endl;
    cout << "R_o = " << R_o << " fm, R_s = " << R_s << " fm, R_l = " << R_l << " fm" << endl;
    cout << "R_os = " << R_os << " fm, R_ol = " << R_ol << " fm, R_sl = " << R_sl << " fm." << endl;

    double chi_sq = 0.0;
    for(int iqout = 0; iqout < qnpts; iqout++)
    {
        double q_out_local = q_out[iqout];
        for(int iqside = 0; iqside < qnpts; iqside++)
        {
            double q_side_local = q_side[iqside];
            for(int iqlong = 0; iqlong < qnpts; iqlong++)
            {
                double q_long_local = q_long[iqlong];
                double correl_local = Correl_3D[iqout][iqside][iqlong];
                double sigma_k_prime = Correl_3D_err[iqout][iqside][iqlong]/correl_local;

                chi_sq += pow((log(correl_local) - results[0] 
                               + results[1]*q_out_local*q_out_local 
                               + results[2]*q_side_local*q_side_local
                               + results[3]*q_long_local*q_long_local
                               + results[4]*q_out_local*q_side_local
                               + results[5]*q_out_local*q_long_local
                               + results[6]*q_side_local*q_long_local), 2)
                          /sigma_k_prime/sigma_k_prime;
            }
        }
    }
    cout << "chi_sq/d.o.f = " << chi_sq/pow(qnpts, 3) << endl;

    gsl_matrix_free (T_gsl);
    gsl_matrix_free (T_inverse_gsl);
    gsl_permutation_free (perm);

    delete [] qweight;
    delete [] V;
    for(int i = 0; i < dim; i++)
    {
        delete [] T[i];
        delete [] T_inverse[i];
    }
    delete [] T;
    delete [] T_inverse;
    delete [] results;
}

//*********************************************************************
// Functions used for multi-dimensional fit
void fit_correlation::fit_Correlationfunction1D_gsl()
{
  const int data_length = qnpts;  // # of points
  const size_t n_para = 2;  // # of parameters

  // allocate space for a covariance matrix of size p by p
  gsl_matrix *covariance_ptr = gsl_matrix_alloc (n_para, n_para);

  // allocate and setup for generating gaussian distibuted random numbers
  gsl_rng_env_setup ();
  const gsl_rng_type *type = gsl_rng_default;
  gsl_rng *rng_ptr = gsl_rng_alloc (type);

  //set up test data
  struct Correlationfunction1D_data Correlfun1D_data;
  Correlfun1D_data.data_length = data_length;
  Correlfun1D_data.q = new double [data_length];
  Correlfun1D_data.y = new double [data_length];
  Correlfun1D_data.sigma = new double [data_length];

  for(int i=0; i<data_length; i++)
  {
     Correlfun1D_data.q[i] = q_out[i];
     // This sets up the data to be fitted, with gaussian noise added
     //Correlfun1D_data.y[i] = 1.0*exp(-10*q_out[i]*q_out[i]);
     Correlfun1D_data.y[i] = Correl_1D_out[i];
     Correlfun1D_data.sigma[i] = Correl_1D_out_err[i];
     //cout << Correlfun1D_data.q[i] << "  " << Correlfun1D_data.y[i] << "  " << Correlfun1D_data.sigma[i] << endl;
  }

  double para_init[n_para] = {1.0, 1.0};  // initial guesse of parameters

  gsl_vector_view xvec_ptr = gsl_vector_view_array (para_init, n_para);
  
  // set up the function to be fit 
  gsl_multifit_function_fdf target_func;
  target_func.f = &Fittarget_correlfun1D_f;        // the function of residuals
  target_func.df = &Fittarget_correlfun1D_df;      // the gradient of this function
  target_func.fdf = &Fittarget_correlfun1D_fdf;    // combined function and gradient
  target_func.n = data_length;              // number of points in the data set
  target_func.p = n_para;              // number of parameters in the fit function
  target_func.params = &Correlfun1D_data;  // structure with the data and error bars

  const gsl_multifit_fdfsolver_type *type_ptr = gsl_multifit_fdfsolver_lmsder;
  gsl_multifit_fdfsolver *solver_ptr 
       = gsl_multifit_fdfsolver_alloc (type_ptr, data_length, n_para);
  gsl_multifit_fdfsolver_set (solver_ptr, &target_func, &xvec_ptr.vector);

  size_t iteration = 0;         // initialize iteration counter
  print_fit_state_1D_gsl (iteration, solver_ptr);
  int status;  		// return value from gsl function calls (e.g., error)
  do
  {
      iteration++;
      
      // perform a single iteration of the fitting routine
      status = gsl_multifit_fdfsolver_iterate (solver_ptr);

      // print out the status of the fit
      cout << "status = " << gsl_strerror (status) << endl;

      // customized routine to print out current parameters
      print_fit_state_1D_gsl (iteration, solver_ptr);

      if (status)    // check for a nonzero status code
      {
          break;  // this should only happen if an error code is returned 
      }

      // test for convergence with an absolute and relative error (see manual)
      status = gsl_multifit_test_delta (solver_ptr->dx, solver_ptr->x, 
                                        fit_tolarence, fit_tolarence);
  }
  while (status == GSL_CONTINUE && iteration < fit_max_iterations);

  // calculate the covariance matrix of the best-fit parameters
  gsl_multifit_covar (solver_ptr->J, 0.0, covariance_ptr);

  // print out the covariance matrix using the gsl function (not elegant!)
  cout << endl << "Covariance matrix: " << endl;
  gsl_matrix_fprintf (stdout, covariance_ptr, "%g");

  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		                // # of digits in doubles

  int width = 7;		// setw width for output
  cout << endl << "Best fit results:" << endl;
  cout << "lambda = " << setw (width) << get_fit_results (0, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (0, covariance_ptr) << endl;

  cout << "R = " << setw (width) << get_fit_results (1, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (1, covariance_ptr) << endl;

  cout << "status = " << gsl_strerror (status) << endl;
  cout << "------------------------------------------------------------------" << endl;

  double chi = gsl_blas_dnrm2(solver_ptr->f);
  double dof = data_length - n_para;
  double c = GSL_MAX_DBL(1, chi/sqrt(dof));

  lambda_Correl = get_fit_results(0, solver_ptr);
  R_out_Correl = get_fit_results(1, solver_ptr)*hbarC;
  lambda_Correl_err = c*get_fit_err(0, covariance_ptr);
  R_out_Correl_err = c*get_fit_err(1, covariance_ptr)*hbarC;

  cout << "final results: " << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << "chisq/dof = " << chi*chi/dof << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << " lambda = " << lambda_Correl << " +/- " << lambda_Correl_err << endl;
  cout << " R_out = " << R_out_Correl << " +/- " << R_out_Correl_err << endl;

  //clean up
  gsl_matrix_free (covariance_ptr);
  gsl_rng_free (rng_ptr);

  delete[] Correlfun1D_data.q;
  delete[] Correlfun1D_data.y;
  delete[] Correlfun1D_data.sigma;

  gsl_multifit_fdfsolver_free (solver_ptr);  // free up the solver

  return;
}

void fit_correlation::fit_Correlationfunction3D_gsl()
{
  const size_t data_length = qnpts*qnpts*qnpts;  // # of points
  const size_t n_para = 4;  // # of parameters

  // allocate space for a covariance matrix of size p by p
  gsl_matrix *covariance_ptr = gsl_matrix_alloc (n_para, n_para);

  // allocate and setup for generating gaussian distibuted random numbers
  gsl_rng_env_setup ();
  const gsl_rng_type *type = gsl_rng_default;
  gsl_rng *rng_ptr = gsl_rng_alloc (type);

  //set up test data
  struct Correlationfunction3D_data Correlfun3D_data;
  Correlfun3D_data.data_length = data_length;
  Correlfun3D_data.q_o = new double [data_length];
  Correlfun3D_data.q_s = new double [data_length];
  Correlfun3D_data.q_l = new double [data_length];
  Correlfun3D_data.y = new double [data_length];
  Correlfun3D_data.sigma = new double [data_length];

  int idx = 0;
  for(int i=0; i<qnpts; i++)
  {
    for(int j=0; j<qnpts; j++)
    {
      for(int k=0; k<qnpts; k++)
      {
         Correlfun3D_data.q_o[idx] = q_out[i];
         Correlfun3D_data.q_s[idx] = q_side[j];
         Correlfun3D_data.q_l[idx] = q_long[k];
         // This sets up the data to be fitted, with gaussian noise added
         // Correlfun3D_data.y[idx] = 1.0*exp( - 0.81*q_out[i]*q_out[i] - 1.21*q_side[j]*q_side[j] - 4.0*q_long[k]*q_long[k] - 0.25*q_out[i]*q_side[j]) + gsl_ran_gaussian(rng_ptr, error);
         Correlfun3D_data.y[idx] = Correl_3D[i][j][k];
         Correlfun3D_data.sigma[idx] = Correl_3D_err[i][j][k];
         //Correlfun3D_data.sigma[idx] = 1e-2;
         idx++;
      }
    }
  }

  double para_init[n_para] = { 1.0, 1.0, 1.0, 1.0 };  // initial guesse of parameters

  gsl_vector_view xvec_ptr = gsl_vector_view_array (para_init, n_para);
  
  // set up the function to be fit 
  gsl_multifit_function_fdf target_func;
  target_func.f = &Fittarget_correlfun3D_f;        // the function of residuals
  target_func.df = &Fittarget_correlfun3D_df;      // the gradient of this function
  target_func.fdf = &Fittarget_correlfun3D_fdf;    // combined function and gradient
  target_func.n = data_length;              // number of points in the data set
  target_func.p = n_para;              // number of parameters in the fit function
  target_func.params = &Correlfun3D_data;  // structure with the data and error bars

  const gsl_multifit_fdfsolver_type *type_ptr = gsl_multifit_fdfsolver_lmsder;
  gsl_multifit_fdfsolver *solver_ptr 
       = gsl_multifit_fdfsolver_alloc (type_ptr, data_length, n_para);
  gsl_multifit_fdfsolver_set (solver_ptr, &target_func, &xvec_ptr.vector);

  size_t iteration = 0;         // initialize iteration counter
  print_fit_state_3D_gsl (iteration, solver_ptr);
  int status;  		// return value from gsl function calls (e.g., error)
  do
  {
      iteration++;
      
      // perform a single iteration of the fitting routine
      status = gsl_multifit_fdfsolver_iterate (solver_ptr);

      // print out the status of the fit
      cout << "status = " << gsl_strerror (status) << endl;

      // customized routine to print out current parameters
      print_fit_state_3D_gsl (iteration, solver_ptr);

      if (status)    // check for a nonzero status code
      {
          break;  // this should only happen if an error code is returned 
      }

      // test for convergence with an absolute and relative error (see manual)
      status = gsl_multifit_test_delta (solver_ptr->dx, solver_ptr->x, 
                                        fit_tolarence, fit_tolarence);
  }
  while (status == GSL_CONTINUE && iteration < fit_max_iterations);

  // calculate the covariance matrix of the best-fit parameters
  gsl_multifit_covar (solver_ptr->J, 0.0, covariance_ptr);

  // print out the covariance matrix using the gsl function (not elegant!)
  cout << endl << "Covariance matrix: " << endl;
  gsl_matrix_fprintf (stdout, covariance_ptr, "%g");

  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		                // # of digits in doubles

  int width = 7;		// setw width for output
  cout << endl << "Best fit results:" << endl;
  cout << "Ro = " << setw (width) << get_fit_results (0, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (0, covariance_ptr) << endl;

  cout << "Rs      = " << setw (width) << get_fit_results (1, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (1, covariance_ptr) << endl;

  cout << "Rl      = " << setw (width) << get_fit_results (2, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (2, covariance_ptr) << endl;
  
  cout << "Ros      = " << setw (width) << get_fit_results (3, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (3, covariance_ptr) << endl;
    
  cout << "status = " << gsl_strerror (status) << endl;
  cout << "--------------------------------------------------------------------" << endl;

  double chi = gsl_blas_dnrm2(solver_ptr->f);
  double dof = data_length - n_para;
  double c = GSL_MAX_DBL(1, chi/sqrt(dof));

  lambda_Correl = 1.0;
  lambda_Correl_err = 0.0;
  R_out_Correl = fabs(get_fit_results(0, solver_ptr))*hbarC;
  R_side_Correl = fabs(get_fit_results(1, solver_ptr))*hbarC;
  R_long_Correl = fabs(get_fit_results(2, solver_ptr))*hbarC;
  R_os_Correl = fabs(get_fit_results(3, solver_ptr))*hbarC;
  R_out_Correl_err = c*get_fit_err(0, covariance_ptr)*hbarC;
  R_side_Correl_err = c*get_fit_err(1, covariance_ptr)*hbarC;
  R_long_Correl_err = c*get_fit_err(2, covariance_ptr)*hbarC;
  R_os_Correl_err = c*get_fit_err(3, covariance_ptr)*hbarC;

  cout << "final results: " << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << "chisq/dof = " << chi*chi/dof << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << " lambda = " << lambda_Correl << " +/- " << lambda_Correl_err << endl;
  cout << " R_out = " << R_out_Correl << " +/- " << R_out_Correl_err << endl;
  cout << " R_side = " << R_side_Correl << " +/- " << R_side_Correl_err << endl;
  cout << " R_long = " << R_long_Correl << " +/- " << R_long_Correl_err << endl;
  cout << " R_os = " << R_os_Correl << " +/- " << R_os_Correl_err << endl;

  //clean up
  gsl_matrix_free (covariance_ptr);
  gsl_rng_free (rng_ptr);

  delete[] Correlfun3D_data.q_o;
  delete[] Correlfun3D_data.q_s;
  delete[] Correlfun3D_data.q_l;
  delete[] Correlfun3D_data.y;
  delete[] Correlfun3D_data.sigma;

  gsl_multifit_fdfsolver_free (solver_ptr);  // free up the solver

  return;
}

void fit_correlation::fit_Correlationfunction3D_withlambda_gsl()
{
  const size_t data_length = qnpts*qnpts*qnpts;  // # of points
  const size_t n_para = 5;  // # of parameters

  // allocate space for a covariance matrix of size p by p
  gsl_matrix *covariance_ptr = gsl_matrix_alloc (n_para, n_para);

  // allocate and setup for generating gaussian distibuted random numbers
  gsl_rng_env_setup ();
  const gsl_rng_type *type = gsl_rng_default;
  gsl_rng *rng_ptr = gsl_rng_alloc (type);

  //set up test data
  struct Correlationfunction3D_data Correlfun3D_data;
  Correlfun3D_data.data_length = data_length;
  Correlfun3D_data.q_o = new double [data_length];
  Correlfun3D_data.q_s = new double [data_length];
  Correlfun3D_data.q_l = new double [data_length];
  Correlfun3D_data.y = new double [data_length];
  Correlfun3D_data.sigma = new double [data_length];

  int idx = 0;
  for(int i=0; i<qnpts; i++)
  {
    for(int j=0; j<qnpts; j++)
    {
      for(int k=0; k<qnpts; k++)
      {
         Correlfun3D_data.q_o[idx] = q_out[i];
         Correlfun3D_data.q_s[idx] = q_side[j];
         Correlfun3D_data.q_l[idx] = q_long[k];
         // This sets up the data to be fitted, with gaussian noise added
         // Correlfun3D_data.y[idx] = 1.0*exp( - 0.81*q_out[i]*q_out[i] - 1.21*q_side[j]*q_side[j] - 4.0*q_long[k]*q_long[k] - 0.25*q_out[i]*q_side[j]) + gsl_ran_gaussian(rng_ptr, error);
         Correlfun3D_data.y[idx] = Correl_3D[i][j][k];
         Correlfun3D_data.sigma[idx] = Correl_3D_err[i][j][k];
         idx++;
      }
    }
  }

  double para_init[n_para] = { 1.0, 1.0, 1.0, 1.0, 1.0 };  // initial guesse of parameters

  gsl_vector_view xvec_ptr = gsl_vector_view_array (para_init, n_para);
  
  // set up the function to be fit 
  gsl_multifit_function_fdf target_func;
  target_func.f = &Fittarget_correlfun3D_f_withlambda;        // the function of residuals
  target_func.df = &Fittarget_correlfun3D_df_withlambda;      // the gradient of this function
  target_func.fdf = &Fittarget_correlfun3D_fdf_withlambda;    // combined function and gradient
  target_func.n = data_length;              // number of points in the data set
  target_func.p = n_para;              // number of parameters in the fit function
  target_func.params = &Correlfun3D_data;  // structure with the data and error bars

  const gsl_multifit_fdfsolver_type *type_ptr = gsl_multifit_fdfsolver_lmsder;
  gsl_multifit_fdfsolver *solver_ptr 
       = gsl_multifit_fdfsolver_alloc (type_ptr, data_length, n_para);
  gsl_multifit_fdfsolver_set (solver_ptr, &target_func, &xvec_ptr.vector);

  size_t iteration = 0;         // initialize iteration counter
  print_fit_state_3D_gsl (iteration, solver_ptr);
  int status;  		// return value from gsl function calls (e.g., error)
  do
  {
      iteration++;
      
      // perform a single iteration of the fitting routine
      status = gsl_multifit_fdfsolver_iterate (solver_ptr);

      // print out the status of the fit
      cout << "status = " << gsl_strerror (status) << endl;

      // customized routine to print out current parameters
      print_fit_state_3D_gsl (iteration, solver_ptr);

      if (status)    // check for a nonzero status code
      {
          break;  // this should only happen if an error code is returned 
      }

      // test for convergence with an absolute and relative error (see manual)
      status = gsl_multifit_test_delta (solver_ptr->dx, solver_ptr->x, 
                                        fit_tolarence, fit_tolarence);
  }
  while (status == GSL_CONTINUE && iteration < fit_max_iterations);

  // calculate the covariance matrix of the best-fit parameters
  gsl_multifit_covar (solver_ptr->J, 0.0, covariance_ptr);

  // print out the covariance matrix using the gsl function (not elegant!)
  cout << endl << "Covariance matrix: " << endl;
  gsl_matrix_fprintf (stdout, covariance_ptr, "%g");

  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		                // # of digits in doubles

  int width = 7;		// setw width for output
  cout << endl << "Best fit results:" << endl;
  cout << "lambda      = " << setw (width) << get_fit_results (0, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (0, covariance_ptr) << endl;

  cout << "Ro = " << setw (width) << get_fit_results (1, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (1, covariance_ptr) << endl;

  cout << "Rs      = " << setw (width) << get_fit_results (2, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (2, covariance_ptr) << endl;

  cout << "Rl      = " << setw (width) << get_fit_results (3, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (3, covariance_ptr) << endl;
  
  cout << "Ros      = " << setw (width) << get_fit_results (4, solver_ptr)
    << " +/- " << setw (width) << get_fit_err (4, covariance_ptr) << endl;
    
  cout << "status = " << gsl_strerror (status) << endl;
  cout << "--------------------------------------------------------------------" << endl;

  double chi = gsl_blas_dnrm2(solver_ptr->f);
  double dof = data_length - n_para;
  double c = GSL_MAX_DBL(1, chi/sqrt(dof));

  lambda_Correl = get_fit_results(0, solver_ptr);
  R_out_Correl = fabs(get_fit_results(1, solver_ptr))*hbarC;
  R_side_Correl = fabs(get_fit_results(2, solver_ptr))*hbarC;
  R_long_Correl = fabs(get_fit_results(3, solver_ptr))*hbarC;
  R_os_Correl = fabs(get_fit_results(4, solver_ptr))*hbarC;
  lambda_Correl_err = c*get_fit_err(0, covariance_ptr);
  R_out_Correl_err = c*get_fit_err(1, covariance_ptr)*hbarC;
  R_side_Correl_err = c*get_fit_err(2, covariance_ptr)*hbarC;
  R_long_Correl_err = c*get_fit_err(3, covariance_ptr)*hbarC;
  R_os_Correl_err = c*get_fit_err(4, covariance_ptr)*hbarC;

  cout << "final results: " << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << "chisq/dof = " << chi*chi/dof << endl;
  cout << scientific << setw(10) << setprecision(5) 
       << " lambda = " << lambda_Correl << " +/- " << lambda_Correl_err << endl;
  cout << " R_out = " << R_out_Correl << " +/- " << R_out_Correl_err << endl;
  cout << " R_side = " << R_side_Correl << " +/- " << R_side_Correl_err << endl;
  cout << " R_long = " << R_long_Correl << " +/- " << R_long_Correl_err << endl;
  cout << " R_os = " << R_os_Correl << " +/- " << R_os_Correl_err << endl;

  //clean up
  gsl_matrix_free (covariance_ptr);
  gsl_rng_free (rng_ptr);

  delete[] Correlfun3D_data.q_o;
  delete[] Correlfun3D_data.q_s;
  delete[] Correlfun3D_data.q_l;
  delete[] Correlfun3D_data.y;
  delete[] Correlfun3D_data.sigma;

  gsl_multifit_fdfsolver_free (solver_ptr);  // free up the solver

  return;
}

//*********************************************************************
// 1D case
//*********************************************************************
//  Simple function to print results of each iteration in nice format
int fit_correlation::print_fit_state_1D_gsl (size_t iteration, gsl_multifit_fdfsolver * solver_ptr)
{
  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		// digits in doubles

  int width = 15;		// setw width for output
  cout << scientific
    << "iteration " << iteration << ": "
    << "x = {" << setw (width) << gsl_vector_get (solver_ptr->x, 0)
    << setw (width) << gsl_vector_get (solver_ptr->x, 1)
    << "}, |f(x)| = " << scientific << gsl_blas_dnrm2 (solver_ptr->f) 
    << endl << endl;

  return 0;
}
//*********************************************************************
// 3D case
//*********************************************************************
//  Simple function to print results of each iteration in nice format
int fit_correlation::print_fit_state_3D_gsl (size_t iteration, gsl_multifit_fdfsolver * solver_ptr)
{
  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		// digits in doubles

  int width = 15;		// setw width for output
  cout << scientific
    << "iteration " << iteration << ": "
    << "  x = {" << setw (width) << gsl_vector_get (solver_ptr->x, 0)
    << setw (width) << gsl_vector_get (solver_ptr->x, 1)
    << setw (width) << gsl_vector_get (solver_ptr->x, 2)
    << setw (width) << gsl_vector_get (solver_ptr->x, 3)
    << "}, |f(x)| = " << scientific << gsl_blas_dnrm2 (solver_ptr->f) 
    << endl << endl;

  return 0;
}
//  Simple function to print results of each iteration in nice format
int fit_correlation::print_fit_state_3D_withlambda_gsl (size_t iteration, gsl_multifit_fdfsolver * solver_ptr)
{
  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (5);		// digits in doubles

  int width = 15;		// setw width for output
  cout << scientific
    << "iteration " << iteration << ": "
    << "  x = {" << setw (width) << gsl_vector_get (solver_ptr->x, 0)
    << setw (width) << gsl_vector_get (solver_ptr->x, 1)
    << setw (width) << gsl_vector_get (solver_ptr->x, 2)
    << setw (width) << gsl_vector_get (solver_ptr->x, 3)
    << setw (width) << gsl_vector_get (solver_ptr->x, 4)
    << "}, |f(x)| = " << scientific << gsl_blas_dnrm2 (solver_ptr->f) 
    << endl << endl;

  return 0;
}
//*********************************************************************
//  Function to return the i'th best-fit parameter
inline double fit_correlation::get_fit_results(int i, gsl_multifit_fdfsolver * solver_ptr)
{
  return gsl_vector_get (solver_ptr->x, i);
}

//*********************************************************************
//  Function to retrieve the square root of the diagonal elements of
//   the covariance matrix.
inline double fit_correlation::get_fit_err (int i, gsl_matrix * covariance_ptr)
{
  return sqrt (gsl_matrix_get (covariance_ptr, i, i));
}





//*********************************************************************
//  Functions return the residuals for each point; that is, the 
//  difference of the fit function using the current parameters
//  and the data to be fit.
int Fittarget_correlfun1D_f (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr)
{
  size_t n = ((struct Correlationfunction1D_data *) params_ptr)->data_length;
  double *q = ((struct Correlationfunction1D_data *) params_ptr)->q;
  double *y = ((struct Correlationfunction1D_data *) params_ptr)->y;
  double *sigma = ((struct Correlationfunction1D_data *) params_ptr)->sigma;

  //fit parameters
  double lambda = gsl_vector_get (xvec_ptr, 0);
  double R = gsl_vector_get (xvec_ptr, 1);

  size_t i;

  for (i = 0; i < n; i++)
  {
      double Yi = lambda*exp(- q[i]*q[i]*R*R);
      gsl_vector_set (f_ptr, i, (Yi - y[i]) / sigma[i]);
  }

  return GSL_SUCCESS;
}

//*********************************************************************
//  Function returning the Jacobian of the residual function
int Fittarget_correlfun1D_df (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr)
{
  size_t n = ((struct Correlationfunction1D_data *) params_ptr)->data_length;
  double *q = ((struct Correlationfunction1D_data *) params_ptr)->q;
  double *sigma = ((struct Correlationfunction1D_data *) params_ptr)->sigma;

  //fit parameters
  double lambda = gsl_vector_get (xvec_ptr, 0);
  double R = gsl_vector_get (xvec_ptr, 1);

  size_t i;

  for (i = 0; i < n; i++)
  {
      // Jacobian matrix J(i,j) = dfi / dxj, 
      // where fi = (Yi - yi)/sigma[i],      
      //       Yi = A * exp(-lambda * i) + b 
      // and the xj are the parameters (A,lambda,b) 
      double sig = sigma[i];

      //derivatives
      double common_elemt = exp(- q[i]*q[i]*R*R);
      
      gsl_matrix_set (Jacobian_ptr, i, 0, common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 1, - lambda*q[i]*q[i]*2.*R*common_elemt/sig);
  }
  return GSL_SUCCESS;
}

//*********************************************************************
//  Function combining the residual function and its Jacobian
int Fittarget_correlfun1D_fdf (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr)
{
  Fittarget_correlfun1D_f(xvec_ptr, params_ptr, f_ptr);
  Fittarget_correlfun1D_df(xvec_ptr, params_ptr, Jacobian_ptr);

  return GSL_SUCCESS;
}


//*********************************************************************
// 3D case
//*********************************************************************
//*********************************************************************
//  Function returning the residuals for each point; that is, the 
//  difference of the fit function using the current parameters
//  and the data to be fit.
int Fittarget_correlfun3D_f (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr)
{
  size_t n = ((struct Correlationfunction3D_data *) params_ptr)->data_length;
  double *q_o = ((struct Correlationfunction3D_data *) params_ptr)->q_o;
  double *q_s = ((struct Correlationfunction3D_data *) params_ptr)->q_s;
  double *q_l = ((struct Correlationfunction3D_data *) params_ptr)->q_l;
  double *y = ((struct Correlationfunction3D_data *) params_ptr)->y;
  double *sigma = ((struct Correlationfunction3D_data *) params_ptr)->sigma;

  //fit parameters
  double R_o = gsl_vector_get (xvec_ptr, 0);
  double R_s = gsl_vector_get (xvec_ptr, 1);
  double R_l = gsl_vector_get (xvec_ptr, 2);
  double R_os = gsl_vector_get (xvec_ptr, 3);

  size_t i;

  for (i = 0; i < n; i++)
  {
      double Yi = exp(- q_l[i]*q_l[i]*R_l*R_l - q_s[i]*q_s[i]*R_s*R_s
                   - q_o[i]*q_o[i]*R_o*R_o - q_o[i]*q_s[i]*R_os*R_os);
      gsl_vector_set (f_ptr, i, (Yi - y[i]) / sigma[i]);
  }

  return GSL_SUCCESS;
}

int Fittarget_correlfun3D_f_withlambda (const gsl_vector *xvec_ptr, void *params_ptr, gsl_vector *f_ptr)
{
  size_t n = ((struct Correlationfunction3D_data *) params_ptr)->data_length;
  double *q_o = ((struct Correlationfunction3D_data *) params_ptr)->q_o;
  double *q_s = ((struct Correlationfunction3D_data *) params_ptr)->q_s;
  double *q_l = ((struct Correlationfunction3D_data *) params_ptr)->q_l;
  double *y = ((struct Correlationfunction3D_data *) params_ptr)->y;
  double *sigma = ((struct Correlationfunction3D_data *) params_ptr)->sigma;

  //fit parameters
  double lambda = gsl_vector_get (xvec_ptr, 0);
  double R_o = gsl_vector_get (xvec_ptr, 1);
  double R_s = gsl_vector_get (xvec_ptr, 2);
  double R_l = gsl_vector_get (xvec_ptr, 3);
  double R_os = gsl_vector_get (xvec_ptr, 4);

  size_t i;

  for (i = 0; i < n; i++)
  {
      double Yi = lambda*exp(- q_l[i]*q_l[i]*R_l*R_l - q_s[i]*q_s[i]*R_s*R_s
                   - q_o[i]*q_o[i]*R_o*R_o - q_o[i]*q_s[i]*R_os*R_os);
      gsl_vector_set (f_ptr, i, (Yi - y[i]) / sigma[i]);
  }

  return GSL_SUCCESS;
}

//*********************************************************************
//  Function returning the Jacobian of the residual function
int Fittarget_correlfun3D_df (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr)
{
  size_t n = ((struct Correlationfunction3D_data *) params_ptr)->data_length;
  double *q_o = ((struct Correlationfunction3D_data *) params_ptr)->q_o;
  double *q_s = ((struct Correlationfunction3D_data *) params_ptr)->q_s;
  double *q_l = ((struct Correlationfunction3D_data *) params_ptr)->q_l;
  double *sigma = ((struct Correlationfunction3D_data *) params_ptr)->sigma;

  //fit parameters
  double R_o = gsl_vector_get (xvec_ptr, 0);
  double R_s = gsl_vector_get (xvec_ptr, 1);
  double R_l = gsl_vector_get (xvec_ptr, 2);
  double R_os = gsl_vector_get (xvec_ptr, 3);

  size_t i;

  for (i = 0; i < n; i++)
  {
      // Jacobian matrix J(i,j) = dfi / dxj, 
      // where fi = (Yi - yi)/sigma[i],      
      //       Yi = A * exp(-lambda * i) + b 
      // and the xj are the parameters (A,lambda,b) 
      double sig = sigma[i];

      //derivatives
      double common_elemt = exp(- q_l[i]*q_l[i]*R_l*R_l - q_s[i]*q_s[i]*R_s*R_s
                   - q_o[i]*q_o[i]*R_o*R_o - q_o[i]*q_s[i]*R_os*R_os);
      
      gsl_matrix_set (Jacobian_ptr, i, 0, - q_o[i]*q_o[i]*2.0*R_o*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 1, - q_s[i]*q_s[i]*2.0*R_s*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 2, - q_l[i]*q_l[i]*2.0*R_l*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 3, - q_o[i]*q_s[i]*2.0*R_os*common_elemt/sig);
  }
  return GSL_SUCCESS;
}

int Fittarget_correlfun3D_df_withlambda (const gsl_vector *xvec_ptr, void *params_ptr,  gsl_matrix *Jacobian_ptr)
{
  size_t n = ((struct Correlationfunction3D_data *) params_ptr)->data_length;
  double *q_o = ((struct Correlationfunction3D_data *) params_ptr)->q_o;
  double *q_s = ((struct Correlationfunction3D_data *) params_ptr)->q_s;
  double *q_l = ((struct Correlationfunction3D_data *) params_ptr)->q_l;
  double *sigma = ((struct Correlationfunction3D_data *) params_ptr)->sigma;

  //fit parameters
  double lambda = gsl_vector_get (xvec_ptr, 0);
  double R_o = gsl_vector_get (xvec_ptr, 1);
  double R_s = gsl_vector_get (xvec_ptr, 2);
  double R_l = gsl_vector_get (xvec_ptr, 3);
  double R_os = gsl_vector_get (xvec_ptr, 4);

  size_t i;

  for (i = 0; i < n; i++)
  {
      // Jacobian matrix J(i,j) = dfi / dxj, 
      // where fi = (Yi - yi)/sigma[i],      
      //       Yi = A * exp(-lambda * i) + b 
      // and the xj are the parameters (A,lambda,b) 
      double sig = sigma[i];

      //derivatives
      double common_elemt = exp(- q_l[i]*q_l[i]*R_l*R_l - q_s[i]*q_s[i]*R_s*R_s
                   - q_o[i]*q_o[i]*R_o*R_o - q_o[i]*q_s[i]*R_os*R_os);
      
      gsl_matrix_set (Jacobian_ptr, i, 0, common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 1, - lambda*q_o[i]*q_o[i]*2.0*R_o*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 2, - lambda*q_s[i]*q_s[i]*2.0*R_s*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 3, - lambda*q_l[i]*q_l[i]*2.0*R_l*common_elemt/sig);
      gsl_matrix_set (Jacobian_ptr, i, 4, - lambda*q_o[i]*q_s[i]*2.0*R_os*common_elemt/sig);
  }
  return GSL_SUCCESS;
}

//*********************************************************************
//  Function combining the residual function and its Jacobian
int Fittarget_correlfun3D_fdf (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr)
{
  Fittarget_correlfun3D_f(xvec_ptr, params_ptr, f_ptr);
  Fittarget_correlfun3D_df(xvec_ptr, params_ptr, Jacobian_ptr);

  return GSL_SUCCESS;
}
int Fittarget_correlfun3D_fdf_withlambda (const gsl_vector* xvec_ptr, void *params_ptr, gsl_vector* f_ptr, gsl_matrix* Jacobian_ptr)
{
  Fittarget_correlfun3D_f_withlambda(xvec_ptr, params_ptr, f_ptr);
  Fittarget_correlfun3D_df_withlambda(xvec_ptr, params_ptr, Jacobian_ptr);

  return GSL_SUCCESS;
}

