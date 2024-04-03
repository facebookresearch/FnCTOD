# mse2e

cd ..

for split in train
do
    for template in llama2
    do  
        for all_turn in False True
        do
            CUDA_VISIBLE_DEVICES=$devices python -m src.mse2e.prompting \
                                                    --split $split \
                                                    --template $template \
                                                    --all_turn $all_turn
        done
    done
done
