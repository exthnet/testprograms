export PGI_ACC_NOTIFY=0
export PGI_ACC_TIME=1

for G in 0 60 120 180 300
do
	for V in 0 64 128 256 512
	do
		for N in 100 500 1000 2000 4000
		do
			echo -n ${G} ${V} ${N}
			grep -m 1 "device time" log_${G}_${V}_${N}_nv_acc.txt | sed -e "s/=/ /g" | sed -e "s/,//g"
		done
	done
done
