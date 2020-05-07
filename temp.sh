for dataset in twitter_rv.net UK yhweb; do
	./converter --in.folder=/mnt/nvme/Real/$dataset --in.type=tsv --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24 --out.sort=no --out.type=directed --out.width=24 &
done

wait

for dataset in RMAT{24..29}; do
	./converter --in.folder=/mnt/nvme/RMAT_ADJ6/$dataset --in.type=adj6 --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24 --out.sort=no --out.type=directed --out.width=24 &
done

wait

for dataset in RMAT{30..31}; do
	./converter --in.folder=/mnt/nvme/RMAT_ADJ6/$dataset --in.type=adj6 --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24 --out.sort=no --out.type=directed --out.width=24 &
done

wait

for dataset in twitter_rv.net UK yhweb; do
	./converter --in.folder=/mnt/nvme/Real/$dataset --in.type=tsv --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24-lt --out.sort=lt --out.type=directed --out.width=24 &
done

wait

for dataset in RMAT{24..29}; do
	./converter --in.folder=/mnt/nvme/RMAT_ADJ6/$dataset --in.type=adj6 --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24-lt --out.sort=lt --out.type=directed --out.width=24 &
done

wait

for dataset in RMAT{30..31}; do
	./converter --in.folder=/mnt/nvme/RMAT_ADJ6/$dataset --in.type=adj6 --out.folder=/mnt/nvme/GCSR/ --out.name=$dataset-24-lt --out.sort=lt --out.type=directed --out.width=24 &
done

wait