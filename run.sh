#!/bin/bash  

Horizon=100
Hidden_dim=400
Rep_num_feature_update=64
for random_seed in 1 12 123 1234 12345; do
    for Rep_num_feature_update in 64 128; do  	
        for Episodes in 1010000; do
        	Pth=temp/seed_${random_seed}_horizon_${Horizon}_episodes_${Episodes}_hidden_dim_${Hidden_dim}_updates_${Rep_num_feature_update}/
        	python main.py --temp_path $Pth --horizon ${Horizon} --num_episodes ${Episodes} --seed ${random_seed} --rep_num_feature_update ${Rep_num_feature_update} >> result.txt
	done
    done
done
