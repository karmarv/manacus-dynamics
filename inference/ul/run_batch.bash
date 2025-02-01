#!/bin/bash

filename=$1
start=`date +%s`
echo "`date`: Run inference on files listed in $filename"

# Leading/trailing whitespace trimming and skip commented lines.
while read -r line; do 
    [[ $line = \#* ]] && continue
    echo "`date`: $line"
    #time python yolo_infer.py --view-debug --video $line --model "./deploy/y11m-dv6-e25-im1280.onnx" --out-path "./run_y11m-dv6-e25-im1280_results"
    time python yolo_infer.py --view-debug --video $line --model "./deploy/y11l-dv6-e25-im1280.onnx" --out-path "./run_y11l-dv6-e25-im1280_results"

done < "$filename"
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
echo "`date`: Run duration $runtime"
