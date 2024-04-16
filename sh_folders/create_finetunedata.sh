cd ..

for pd_size in 100 200 300 400
do
    python create_finetunedata.py --configfile ./data/finetunedata/sft-llama2.yml \
                        --outputfile ./data/finetunedata/sft-llama2-pd$pd_size.json \
                        --domain_size $pd_size \
                        --max_len 4096
done