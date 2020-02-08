#STREAMS=(1 2 3 4 5 6)
#BLOCKS=(160 320 640 1280)
#THREADS=(1024 512 256 128)

STREAMS=(2)
BLOCKS=(160)
THREADS=(1024)

cd ./build/

dataname=$1

for s in "${STREAMS[@]}"; do
	for b in "${BLOCKS[@]}"; do
	    for t in "${THREADS[@]}"; do
		cmake -DFORMAT_GRID_POWER=24 -DCUDA_STREAMS=$s -DCUDA_BLOCKS=$b -DCUDA_THREADS=$t .. > /dev/null;
		make -j 40 > /dev/null;
		./tc /mnt/nvme-raid0/GCSR/$dataname | tee -a "result-tc-$dataname-$s-$b-$t.out";
	    done
	done
done

cd ..
