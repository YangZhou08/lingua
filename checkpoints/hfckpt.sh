rootpath=$1 
python dcp_to_consolidate.py --ckpt_dir $rootpath 

inputpath="$rootpath/consolidated" 
outputpath="$rootpath/hf" 
cp /fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/Llama-3.2-3B/original/tokenizer.model $inputpath 
echo "Copied tokenizer.model to $inputpath" 

echo "here is the ls" 
ls -l $inputpath 

python convert_consolidate_hf.py --input_dir $inputpath --output_dir $outputpath --llama_version 3.2 --num_shards 1 
echo "HF checkpoint is saved in $outputpath" 
