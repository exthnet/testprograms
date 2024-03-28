for f in \
run_bcast_nvc.sh \
run_bcast_nvf.sh \
run_bcast_intelc.sh \
run_bcast_intelf.sh \
run_ibcast1_intelc.sh \
run_ibcast1_intelf.sh \
run_ibcast1_nvc.sh \
run_ibcast1_nvf.sh \
run_ibcast2_intelc.sh \
run_ibcast2_intelf.sh
do
	pjsub $f
done
