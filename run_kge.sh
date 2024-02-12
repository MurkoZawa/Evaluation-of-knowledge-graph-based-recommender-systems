### TransE: lfm1m ###
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
python3 pathlm/models/kge_rec/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin
