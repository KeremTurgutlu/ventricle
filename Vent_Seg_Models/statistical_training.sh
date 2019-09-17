####################################################
### REPEATING EXPERIMENTS FOR STASTICAL TESTING ###
###################################################

NUM_REPS=1

# Brain NOTL MR Baseline11
# Brain NOTL CT Baseline11
# Brain TL MR Baseline11
# Brain TL CT Baseline11

# Decide which is better
# Ventricle  NOTL MR Baseline9
#             or
# Ventricle  NOTL MR Baseline11

# Ventricle NOTL CT Baseline11

# Ventricle  TL MR Baseline9
#             or
# Ventricle  TL MR Baseline11

# Ventricle  TL CT Baseline11

# Pick Best models 9 or 11 
# Sample size 5,10,20,40,60,full TL vs NOTL MR and CT


# ###############################
# ### NOTL_Brain_MR_Baseline ###
# ##############################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#     --MODEL_NAME=NOTL_Brain_MR_Baseline_11 \
#     --model_dir=notl_brain_mr_models \
#     --data_name=notl_brain_mr \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=100 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# ###############################
# ### NOTL_Brain_CT_Baseline ###
# ##############################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#     --MODEL_NAME=NOTL_Brain_CT_Baseline_11 \
#     --model_dir=notl_brain_ct_models \
#     --data_name=notl_brain_ct \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=100 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# #############################
# ### TL_Brain_MR_Baseline ###
# ############################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#     --MODEL_NAME=TL_Brain_MR_Baseline_11 \
#     --model_dir=tl_brain_mr_models \
#     --tl_model_dict=tl_brain_mr_model_dict \
#     --data_name=notl_brain_mr \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=50 \
#     --one_cycle=1 \
#     --early_stop=0 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# #############################
# ### TL_Brain_CT_Baseline ###
# ############################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#     --MODEL_NAME=TL_Brain_CT_Baseline_11 \
#     --model_dir=tl_brain_ct_models \
#     --tl_model_dict=tl_brain_ct_model_dict \
#     --data_name=notl_brain_ct \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=50 \
#     --one_cycle=1 \
#     --early_stop=0 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# ##################################
# ### NOTL_Ventricle_MR_Baseline ###
# #################################

# # Baseline - 9
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#     --MODEL_NAME=NOTL_Ventricle_MR_Baseline_9 \
#     --model_dir=notl_ventricle_mr_models \
#     --data_name=notl_ventricle_mr \
#     --bs=1 \
#     --model_name=baseline9 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=50 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#     --MODEL_NAME=NOTL_Ventricle_MR_Baseline_11 \
#     --model_dir=notl_ventricle_mr_models \
#     --data_name=notl_ventricle_mr \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=50 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# ##################################
# ### NOTL_Ventricle_CT_Baseline ###
# #################################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#     --MODEL_NAME=NOTL_Ventricle_CT_Baseline_11 \
#     --model_dir=notl_ventricle_ct_models \
#     --data_name=notl_ventricle_ct \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=50 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# ################################
# ### TL_Ventricle_MR_Baseline ###
# ################################

# # Baseline - 9
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#     --MODEL_NAME=TL_Ventricle_MR_Baseline_9 \
#     --model_dir=tl_ventricle_mr_models \
#     --tl_model_dict=tl_ventricle_mr_model_dict \
#     --data_name=notl_ventricle_mr \
#     --load_dir=atlas_ventricle_mr_models \
#     --bs=1 \
#     --model_name=baseline9 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=25 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#     --MODEL_NAME=TL_Ventricle_MR_Baseline_11 \
#     --model_dir=tl_ventricle_mr_models \
#     --tl_model_dict=tl_ventricle_mr_model_dict \
#     --data_name=notl_ventricle_mr \
#     --load_dir=atlas_ventricle_mr_models \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=25 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

# ################################
# ### TL_Ventricle_CT_Baseline ###
# ################################

# # Baseline - 11
# for i in $(seq 1 $NUM_REPS)
# do
# echo experiment $i
# python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#     --MODEL_NAME=TL_Ventricle_CT_Baseline_11 \
#     --model_dir=tl_ventricle_ct_models \
#     --tl_model_dict=tl_ventricle_ct_model_dict \
#     --data_name=notl_ventricle_ct \
#     --load_dir=atlas_ventricle_mr_models \
#     --bs=1 \
#     --model_name=baseline11 \
#     --loss_func=dice \
#     --lr=1e-2 \
#     --epochs=25 \
#     --one_cycle=1 \
#     --early_stop=1 \
#     --clip=0 \
#     --eps=1e-4 \
#     --lsuv=0
# done

###############################################
### SAMPLE SIZE EXPERIMENTS - 5,10,20,40,60 ###
###############################################

##################################
### NOTL_Ventricle_MR_Baseline ###
#################################

# # Baseline - 9
# for j in {5,10,20,40,60}
# do
#     for i in $(seq 1 $NUM_REPS)
#     do
#     echo experiment $i
#     python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#         --MODEL_NAME=NOTL_Ventricle_MR_Baseline_9_${j}_SAMPLES \
#         --model_dir=notl_ventricle_mr_models \
#         --data_name=notl_ventricle_mr \
#         --bs=1 \
#         --model_name=baseline9 \
#         --loss_func=dice \
#         --lr=1e-2 \
#         --epochs=50 \
#         --one_cycle=1 \
#         --early_stop=1 \
#         --clip=0 \
#         --eps=1e-4 \
#         --lsuv=0 \
#         --sample_size=${j}
#     done
# done

# # Baseline - 11
# for j in {5,10,20,40,60}
# do
#     for i in $(seq 1 $NUM_REPS)
#     do
#     echo experiment $i
#     python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#         --MODEL_NAME=NOTL_Ventricle_MR_Baseline_11_${j}_SAMPLES \
#         --model_dir=notl_ventricle_mr_models \
#         --data_name=notl_ventricle_mr \
#         --bs=1 \
#         --model_name=baseline11 \
#         --loss_func=dice \
#         --lr=1e-2 \
#         --epochs=50 \
#         --one_cycle=1 \
#         --early_stop=1 \
#         --clip=0 \
#         --eps=1e-4 \
#         --lsuv=0 \
#         --sample_size=${j}
#     done
# done

################################
### TL_Ventricle_MR_Baseline ###
################################

# # Baseline - 9
# for j in {5,10,20,40,60}
# do
#     for i in $(seq 1 $NUM_REPS)
#     do
#     echo experiment $i
#     python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#         --MODEL_NAME=TL_Ventricle_MR_Baseline_9_${j}_SAMPLES \
#         --model_dir=tl_ventricle_mr_models \
#         --tl_model_dict=tl_ventricle_mr_model_dict \
#         --data_name=notl_ventricle_mr \
#         --load_dir=atlas_ventricle_mr_models \
#         --bs=1 \
#         --model_name=baseline9 \
#         --loss_func=dice \
#         --lr=1e-2 \
#         --epochs=25 \
#         --one_cycle=1 \
#         --early_stop=1 \
#         --clip=0 \
#         --eps=1e-4 \
#         --lsuv=0 \
#         --sample_size=${j}
#     done
# done

# Baseline - 11
for j in {5,}
do
    for i in $(seq 1 $NUM_REPS)
    do
    echo experiment $i
    python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
        --MODEL_NAME=TL_Ventricle_MR_Baseline_11_${j}_SAMPLES \
        --model_dir=tl_ventricle_mr_models \
        --tl_model_dict=tl_ventricle_mr_model_dict \
        --data_name=notl_ventricle_mr \
        --load_dir=atlas_ventricle_mr_models \
        --bs=1 \
        --model_name=baseline11 \
        --loss_func=dice \
        --lr=1e-2 \
        --epochs=3 \
        --one_cycle=1 \
        --early_stop=1 \
        --clip=0 \
        --eps=1e-4 \
        --lsuv=0 \
        --sample_size=${j}
    done
done

# ##################################
# ### NOTL_Ventricle_CT_Baseline ###
# #################################

# # Baseline - 11
# for j in {5,10,20,40,60}
# do
#     for i in $(seq 1 $NUM_REPS)
#     do
#     echo experiment $i
#     python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_training.py \
#         --MODEL_NAME=NOTL_Ventricle_CT_Baseline_11_${j}_SAMPLES \
#         --model_dir=notl_ventricle_ct_models \
#         --data_name=notl_ventricle_ct \
#         --bs=1 \
#         --model_name=baseline11 \
#         --loss_func=dice \
#         --lr=1e-2 \
#         --epochs=50 \
#         --one_cycle=1 \
#         --early_stop=1 \
#         --clip=0 \
#         --eps=1e-4 \
#         --lsuv=0 \
#         --sample_size=${j}
#     done
# done

# ################################
# ### TL_Ventricle_CT_Baseline ###
# ################################

# # Baseline - 11
# for j in {5,10,20,40,60}
# do
#     for i in $(seq 1 $NUM_REPS)
#     do
#     echo experiment $i
#     python /home/turgutluk/fastai/fastai/launch.py --gpus=012345 ./distributed_transfer_learning.py \
#         --MODEL_NAME=TL_Ventricle_CT_Baseline_11_${j}_SAMPLES \
#         --model_dir=tl_ventricle_ct_models \
#         --tl_model_dict=tl_ventricle_ct_model_dict \
#         --data_name=notl_ventricle_ct \
#         --load_dir=atlas_ventricle_mr_models \
#         --bs=1 \
#         --model_name=baseline11 \
#         --loss_func=dice \
#         --lr=1e-2 \
#         --epochs=25 \
#         --one_cycle=1 \
#         --early_stop=1 \
#         --clip=0 \
#         --eps=1e-4 \
#         --lsuv=0 \
#         --sample_size=${j}
#     done
# done









































