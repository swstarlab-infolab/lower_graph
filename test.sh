#STREAMS=(1 2 3 4 5 6)
#BLOCKS=(160 320 640 1280)
STREAMS=(5)
BLOCKS=(1280)
THREADS=(1024)

cd ./build/
#for ((i = 0; i < ${#BLOCKS[@]}; i++)); do

dataname=$1

#for dataname in RMAT{26..28}; do
    for s in "${STREAMS[@]}"; do
        for b in "${BLOCKS[@]}"; do
            for t in "${THREADS[@]}"; do
                cmake -DCUDA_STREAMS=$s -DCUDA_BLOCKS=$b -DCUDA_THREADS=$t .. > /dev/null;
                make -j 40 > /dev/null;
                ./tc /mnt/nvme-raid0/GCSR/$dataname | tee -a "result-tc-$dataname-$s-$b-$t.out";
            done
        done
    done
#done

cd ..