g.region s=0 w=0 e=800.00 n=550.00 res=2.5000
g.region -s

r.in.gdal --overwrite input=../../../topo.asc output=elev
r.in.gdal --overwrite input=../../../mass.asc output=minit

start_time=`date +%s`
r.avaflow -e elevation=elev hrelease=minit controls=0,0,0,0,0,0 thresholds=1.00E-03,1.00E-03,1.00E-03,1.00E-03 cfl=2.50E-01,0.0001 time=0.500,40.00 phases=s friction=15.00,15.00,0.00 prefix=delta1_15p00
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_15p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt


start_time=`date +%s`
r.avaflow -e elevation=elev hrelease=minit controls=0,0,0,0,0,0 thresholds=1.00E-03,1.00E-03,1.00E-03,1.00E-03 cfl=2.50E-01,0.0001 time=0.500,40.00 phases=s friction=20.00,20.00,0.00 prefix=delta1_20p00
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_20p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt


start_time=`date +%s`
r.avaflow -e elevation=elev hrelease=minit controls=0,0,0,0,0,0 thresholds=1.00E-03,1.00E-03,1.00E-03,1.00E-03 cfl=2.50E-01,0.0001 time=0.500,40.00 phases=s friction=25.00,25.00,0.00 prefix=delta1_25p00
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_25p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt

