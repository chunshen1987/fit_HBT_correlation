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

