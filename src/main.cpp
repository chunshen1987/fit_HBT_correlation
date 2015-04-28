//===============================================================================
//  fit the HBT correlation function to get HBT radii
//
//
//  Programmer: Chun Shen
//       Email: chunshen@physics.mcgill.ca
//
//
//=============================================================================

#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<math.h>
#include<sys/time.h>

#include<gsl/gsl_sf_bessel.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>

#include "Stopwatch.h"
#include "parameters.h"
#include "arsenal.h"
#include "ParameterReader.h"
#include "fit_correlation.h"

using namespace std;

int main(int argc, char *argv[])
{
   cout << endl
        << "                  iHBT_fit                   " << endl
        << endl
        << "  Ver 1.2   ----- Chun Shen, 10/2014   " << endl;
   cout << endl << "**********************************************************" << endl;
   display_logo(2); // Hail to the king~
   cout << endl << "**********************************************************" << endl << endl;
   
   // Read-in parameters
   ParameterReader *paraRdr = new ParameterReader;
   paraRdr->readFromFile("parameters.dat");
   paraRdr->readFromArguments(argc, argv, "#", 2);
   paraRdr->echo();
   
   string filename = argv[1];

   Stopwatch sw_total;
   sw_total.tic();

   fit_correlation test(filename, paraRdr);
   test.fit();

   sw_total.toc();
   cout << "Program totally finished in " << sw_total.takeTime() << " sec." << endl;
   return 0;
}
