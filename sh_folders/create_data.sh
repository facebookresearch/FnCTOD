cd ..

for pd_size in 100 200 300 400
python create_finetunedata.py --configfile ./data/finetunedata_configs/sft-llama2.yaml \
                      --outputfile ./data/finetunedata/sft-llama2-pd$pd_size.json \
                      --domain_size $pd_size \
                      --max_len 4096