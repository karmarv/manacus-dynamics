#!/bin/bash

filename=$1
start=`date +%s`
echo "`date`: Run inference on files listed in $filename"

# Leading/trailing whitespace trimming and skip commented lines.
while read -r line; do 
    [[ $line = \#* ]] && continue
    echo "`date`: $line"
    time python yolo_infer.py --view-debug --video $line --out-path "./run_y11m_e10_results"

done < "$filename"
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
echo "`date`: Run duration $runtime"
