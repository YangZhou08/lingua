inputpath="/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/debug/checkpoints/0000000620/consolidated 
cp /fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/llama3b0100/consolidated/tokenizer.model $inputpath 
echo "Copied tokenizer.model to $inputpath" 

echo "here is the ls" 
ls -l $inputpath 

python convert_consolidate_hf.py --input_dir $inputpath --llama_version 3.2 --num_shards 1 
