for file in data/*.txt
do 
  uv run cs336_basics/bpe_tokenizer.py out/bpe_params_ts-train/vocab.pkl out/bpe_params_ts-train/merges.pkl $file ${file}-ts.npy
  uv run cs336_basics/bpe_tokenizer.py out/bpe_params_owt-train-long/vocab.pkl out/bpe_params_owt-train-long/merges.pkl $file ${file}-owt.npy
done