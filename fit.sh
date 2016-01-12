#!/usr/bin/env bash

path=$1
mode=$2

if [ -z "$mode" ]; then
    mode=2
fi

for ii in `ls $path | grep "correlation_function"`
do 
    ./fit_HBT.e $path/$ii fit_mode=$mode
    awk {'print $1, $2, $4, $6, $8, $10, $16'} HBT_radii_fit_mode_$mode.dat > HBT_radii_kt_`echo $ii | cut -f 5,6 -d _ | cut -f 1 -d d`dat
    mv HBT_radii_kt_* $path
done
