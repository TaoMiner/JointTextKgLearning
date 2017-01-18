ulimit -s unlimited
ulimit -c unlimited
make
time ./align -train_text ../data/wiki2016/train_text_mention -train_kg ../data/wiki2016/train_kg -train_anchor ../data/wiki2016/train_text_mention -read_mention_vocab ../data/wiki2016/vocab_mention -output_path ../etc/wiki2016ab/ -min-count 5 -cw 1 -sg 0 -size 200 -negative 5 -sample 1e-4 -threads 20 -binary 1 -iter 1 -window 5