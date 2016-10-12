make
time ./align -train_text ../data/wiki2016/train_text_ab -train_kg ../data/wiki2016/train_kg -train_anchor ../data/wiki2016/train_text_ab -output_path ../etc/wiki2016ab/ -min-count 5 -cw 0 -size 200 -negative 5 -sample 1e-4 -threads 20 -binary 1 -iter 1 -window 5
