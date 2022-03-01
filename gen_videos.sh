for scene in room fern fortress leaves orchids flower trex horns
do
	sh llff_eval.sh $scene
done

for scan in 1 4 15 24 32 33 49 110 114 118
do
	sh dtu_eval.sh "scan${scan}"
done