inputpath="/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/thinkingonly/checkpoints/0000010000/consolidated" 
outputpath="/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/thinkingonly/checkpoints/0000010000/hf" 
# cp /fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/llama3b0100/consolidated/tokenizer.model $inputpath 
echo "Copied tokenizer.model to $inputpath" 

echo "here is the ls" 
ls -l $inputpath 

python convert_consolidate_hf.py --input_dir $inputpath --output_dir $outputpath --llama_version 3.2 --num_shards 1 
