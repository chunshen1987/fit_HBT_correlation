#!/usr/bin/env bash

path=$1

for ii in `ls $path`
do 
    ./fit_HBT.e $path/$ii
    awk {'print $1, $2, $4, $6, $8, $10, $16'} HBT_radii_fit_mode_1.dat > HBT_radii_kt_`echo $ii | cut -f 4 -d _ | cut -f 1 -d d`dat
done
