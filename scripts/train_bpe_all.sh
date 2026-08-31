source .venv/bin/activate

echo "owt 10k"
python cs336_basics/train_bpe.py /data/raw/owt_train.txt --vocab-size 10000 --out-dir /data/models/bpe_10k_owt

echo "owt 32k"
python cs336_basics/train_bpe.py /data/raw/owt_train.txt --vocab-size 32000 --out-dir /data/models/bpe_32k_owt