grep performance ./log_bench_mic_allsize.txt | grep  "128^2" | awk '{print $3}' | sed "s/,//" | grep -v novec > ./log_bench_mic_id.txt
grep performance ./log_bench_mic_allsize.txt | grep  "128^2" | awk '{print $18}' | grep -v novec > ./log_bench_mic_128.txt
grep performance ./log_bench_mic_allsize.txt | grep  "256^2" | awk '{print $18}' | grep -v novec > ./log_bench_mic_256.txt
grep performance ./log_bench_mic_allsize.txt | grep  "512^2" | awk '{print $18}' | grep -v novec > ./log_bench_mic_512.txt
grep performance ./log_bench_mic_allsize.txt | grep "1024^2" | awk '{print $18}' | grep -v novec > ./log_bench_mic_1024.txt
grep performance ./log_bench_mic_allsize.txt | grep "2048^2" | awk '{print $18}' | grep -v novec > ./log_bench_mic_2048.txt
echo ",128,256,512,1024,2048" >  ./log_bench_mic_allsize.csv
paste -d "," ./log_bench_mic_id.txt ./log_bench_mic_128.txt ./log_bench_mic_256.txt ./log_bench_mic_512.txt ./log_bench_mic_1024.txt ./log_bench_mic_2048.txt >> ./log_bench_mic_allsize.csv
