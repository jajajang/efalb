for A in LowESTR #bmoracle LowPopArt bloful bltwostage-sp_simple2
do
python3 run_expr01_230927.py ${A} -d 6 -R 1 -r 1 -T 100000 -tg 1 --nTry=60 
2>&1 | tee log_231016_${A}.txt;
done
