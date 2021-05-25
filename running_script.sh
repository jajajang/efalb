for A in  bloful bmoracle bltwostage-bm-sp_simple2 bltwostage-sp_simple2  
do
python3  run_expr01.py ${A} -d 8 -R 0.01 -r 1 -T 10000 -tg 1 --nTry=60 |& tee log_0207_${A}.txt;
done
