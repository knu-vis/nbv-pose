sequences=(
    '171204_pose1'
    '161029_piano4'
    '170915_office1'
    '170228_haggling_a3'
    '170407_haggling_b3'
    '170224_haggling_a3'
    '170404_haggling_b3'
    '171026_cello3'
    
    '171026_pose2'
    '170221_haggling_b3'
    '161029_tools1'
)

for seq in "${sequences[@]}"
do
    echo ${seq}
    cmd=$(bash scripts/getData_31.sh ${seq} 0 31)
    cmd=$(bash scripts/extractAll_hd.sh ${seq})
    echo $cmd
    # eval $cmd
done
 