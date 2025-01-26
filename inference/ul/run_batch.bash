#!/bin/bash

filename=$1
start=`date +%s`
echo "`date`: Run inference on files listed in $filename"

# Leading/trailing whitespace trimming and skip commented lines.
while read -r line; do 
    [[ $line = \#* ]] && continue
    echo "`date`: $line"
    time python yolo_infer.py --view-debug --video $line --model "./deploy/best_y11l-dv5-e100-train.onnx" --out-path "./run_y11l_e100_results"

done < "$filename"
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
echo "`date`: Run duration $runtime"
