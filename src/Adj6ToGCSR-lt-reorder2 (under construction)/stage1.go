package main

import (
	"os"
	"path/filepath"
)

func stage1_mapper(adj6 []uint8, in <-chan sRawDat, gWidth uint64) <-chan []gridEdge {
	out := make(chan []gridEdge, 128)

	go func() {
		defer close(out)
		for dat := range in {
			el := make([]gridEdge, dat.cnt)
			selfloop := uint64(0)

			for i := uint64(0); i < dat.cnt; i++ {
				s := dat.src
				now := dat.dstStart + i*6
				d := be6_le8(adj6[now : now+6])

				if s < d {
					s, d = d, s
				} else if s == d {
					selfloop++
					continue
				}

				el[i-selfloop] = gridEdge{
					gidx: [2]uint32{uint32(s / gWidth), uint32(d / gWidth)},
					edge: [2]uint32{uint32(s % gWidth), uint32(d % gWidth)},
				}
			}

			el = el[:uint64(len(el))-selfloop]
			out <- el
		}
	}()
	return out
}

func stage1_mapper_reorder(adj6 []uint8, in <-chan sRawDat, gWidth uint64, rSlice []uint64) <-chan []gridEdge {
	out := make(chan []gridEdge, 128)

	go func() {
		defer close(out)
		for dat := range in {
			el := make([]gridEdge, dat.cnt)
			selfloop := uint64(0)

			for i := uint64(0); i < dat.cnt; i++ {
				s := rSlice[dat.src]
				now := dat.dstStart + i*6
				d := rSlice[be6_le8(adj6[now:now+6])]

				if s < d {
					s, d = d, s
				} else if s == d {
					selfloop++
					continue
				}

				el[i-selfloop] = gridEdge{
					gidx: [2]uint32{uint32(s / gWidth), uint32(d / gWidth)},
					edge: [2]uint32{uint32(s % gWidth), uint32(d % gWidth)},
				}
			}

			el = el[:uint64(len(el))-selfloop]
			out <- el
		}
	}()
	return out
}

func stage1_shuffler(in <-chan []gridEdge, targetFolder string) {
	temp := make(map[[2]uint32]([][2]uint32))

	for dat := range in {
		for _, ge := range dat {
			if _, ok := temp[ge.gidx]; !ok {
				temp[ge.gidx] = [][2]uint32{}
			}
			temp[ge.gidx] = append(temp[ge.gidx], ge.edge)

			// (1 << 18) x sizeof(edge32) = 1M x 8B = 2MB
			if len(temp[ge.gidx]) > (1 << 18) {
				targetFile := filepath.Join(targetFolder, fileNameEncode(ge.gidx, ".el32"))
				if err := fileSaveAppend(targetFile, u32a2s_bs(temp[ge.gidx])); err != nil {
					panic(err)
				}

				temp[ge.gidx] = temp[ge.gidx][:0]
			}
		}
	}

	for k, v := range temp {
		if len(v) > 0 {
			targetFile := filepath.Join(targetFolder, fileNameEncode(k, ".el32"))
			if err := fileSaveAppend(targetFile, u32a2s_bs(v)); err != nil {
				panic(err)
			}
		}
	}
}

func stage1_reorder(inFolder, outFolder, outName string, gWidth uint64, rSlice []uint64) {
	fListChan := fileList(inFolder, "")

	absOutFolder, err := filepath.Abs(outFolder)
	if err != nil {
		panic(err)
	}

	targetFolder := filepath.Join(absOutFolder, outName)
	if err := os.MkdirAll(targetFolder, 0755); err != nil {
		panic(err)
	}

	parallel_for(4, func(i int) {
		for fPath := range fListChan {
			adj6, err := fileLoad(fPath)
			if err != nil {
				panic(err)
			}

			sRawDatChan := split_be6(adj6)

			parallel_for(32, func(j int) {
				mapped := stage1_mapper_reorder(adj6, sRawDatChan, gWidth, rSlice)
				stage1_shuffler(mapped, targetFolder)
			})
		}
	})
}

func stage1(inFolder, outFolder, outName string, gWidth uint64) {
	fListChan := fileList(inFolder, "")

	absOutFolder, err := filepath.Abs(outFolder)
	if err != nil {
		panic(err)
	}

	targetFolder := filepath.Join(absOutFolder, outName)
	if err := os.MkdirAll(targetFolder, 0755); err != nil {
		panic(err)
	}

	parallel_for(4, func(i int) {
		for fPath := range fListChan {
			adj6, err := fileLoad(fPath)
			if err != nil {
				panic(err)
			}

			sRawDatChan := split_be6(adj6)

			parallel_for(32, func(j int) {
				mapped := stage1_mapper(adj6, sRawDatChan, gWidth)
				stage1_shuffler(mapped, targetFolder)
			})
		}
	})
}
