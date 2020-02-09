STREAMS=(2 3 4 5)
#BLOCKS=(1280 640 320 160 80)
BLOCKS=(160 320 640 1280 2560)
THREADS=(1024 512 256 128 64)

cd ./build/

dataname=$1

#for s in "${STREAMS[@]}"; do
#	for b in "${BLOCKS[@]}"; do
#	    for t in "${THREADS[@]}"; do
#		cmake -DFORMAT_GRID_POWER=24 -DCUDA_STREAMS=$s -DCUDA_BLOCKS=$b -DCUDA_THREADS=$t .. > /dev/null;
#		make -j > /dev/null;
#		./query-tc /mnt/nvme-raid0/GCSR/$dataname | tee -a "result-tc-$dataname-$s-$b-$t.out";
#	    done
#	done
#done

for s in "${STREAMS[@]}"; do
	for i in {0..4}; do
		cmake -DFORMAT_GRID_POWER=24 -DCUDA_STREAMS=$s -DCUDA_BLOCKS=${BLOCKS[$i]} -DCUDA_THREADS=${THREADS[$i]} .. > /dev/null;
		make -j > /dev/null;
		./query-tc /mnt/nvme-raid0/GCSR/$dataname | tee -a "result-tc-$dataname-$s-$b-$t.out";
	done
done

cd ..
