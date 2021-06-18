start_time=`date +%s`
shaltop "" delta1_15p00.txt
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_15p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt

start_time=`date +%s`
shaltop "" delta1_20p00.txt
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_20p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt

start_time=`date +%s`
shaltop "" delta1_25p00.txt
end_time=`date +%s`
elapsed_time=$(($end_time - $start_time))
string_time="${start_time} delta1_25p00 ${elapsed_time}"
echo ${string_time} >> simulation_duration.txt

