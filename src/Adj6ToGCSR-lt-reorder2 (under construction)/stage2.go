package main

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/jfcg/sorty"
)

type edgelist [][2]uint32

func (el edgelist) Len() int { return len(el) }
func (el edgelist) Swap(i, j int) {
	el[i], el[j] = el[j], el[i]
}
func (el edgelist) Less(i, j int) bool {
	return el[i][0] < el[j][0] || (el[i][0] == el[j][0] && el[i][1] < el[j][1])
}

func (el edgelist) SequentialSort(i, j int) {
	sort.SliceStable(el, func(i, j int) bool {
		return el[i][0] < el[j][0] || (el[i][0] == el[j][0] && el[i][1] < el[j][1])
	})
}

func stage2_dedup(in [][2]uint32) [][2]uint32 {
	bitVec := make([]uint32, ceil(uint(len(in)), 32))
	mtxVec := make([]sync.Mutex, ceil(uint(len(in)), 32))

	getBit := func(i uint) bool {
		return (atomic.LoadUint32(&(bitVec[i/32])) & (uint32(1) << (i % 32))) != 0
	}

	setBit := func(i uint) {
		mtxVec[i/32].Lock()
		bitVec[i/32] |= (uint32(1) << (i % 32))
		mtxVec[i/32].Unlock()
	}

	workers := runtime.NumCPU()

	// initialzie bitvec
	parallel_for(workers, func(wIdx int) {
		for i := wIdx; i < len(bitVec); i += workers {
			bitVec[i] = 0
		}
	})

	sorty.Mxg = uint32(runtime.NumCPU()) * 2

	// sort input
	sorty.Sort(len(in), func(i, j, r, s int) bool {
		if in[i][0] < in[j][0] || (in[i][0] == in[j][0] && in[i][1] < in[j][1]) {
			if r != s {
				in[r], in[s] = in[s], in[r]
			}
			return true
		}
		return false
	})

	// set bitvec
	ones := uint64(0)
	parallel_for(workers, func(wIdx int) {
		myOnes := uint64(0)
		for i := wIdx; i < len(in)-1; i += workers {
			if in[i][0] != in[i+1][0] || in[i][1] != in[i+1][1] {
				setBit(uint(i))
				myOnes++
			}
		}
		atomic.AddUint64(&ones, myOnes)
	})
	setBit(uint(len(in) - 1))

	// prepare prefix sum result slice
	pSumRes := make([]uint, len(in)+1)
	pSumRes[0] = 0

	// preprocess for prefix sum
	parallel_for(workers, func(wIdx int) {
		for i := wIdx + 1; i < len(pSumRes); i += workers {
			if getBit(uint(i - 1)) {
				pSumRes[i] = 1
			} else {
				pSumRes[i] = 0
			}
		}
	})
	// reduction sum
	//ones := reductionSum(pSumRes)
	out := make([][2]uint32, ones)

	// exclusive prefix sum
	pSumResPart := pSumRes[1:]
	prefixSumInclusive(&pSumResPart)

	// compaction
	parallel_for(workers, func(wIdx int) {
		for i := wIdx; i < len(in); i += workers {
			if getBit(uint(i)) {
				out[pSumRes[i]] = in[i]
			}
		}
	})

	return out
}

func stage2_save(fileStem string, deduped [][2]uint32) {
	workers := runtime.NumCPU()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		col := make([]uint32, len(deduped))
		parallel_for(workers, func(wIdx int) {
			for i := wIdx; i < len(deduped); i += workers {
				col[i] = deduped[i][1]
			}
		})

		if err := fileSave(fileStem+".col", u32s_bs(col)); err != nil {
			panic(err)
		}
	}()

	// fill row and ptr
	wg.Add(1)
	go func() {
		defer wg.Done()

		bitVec := make([]uint32, ceil(uint(len(deduped)), 32))
		mtxVec := make([]sync.Mutex, ceil(uint(len(deduped)), 32))

		getBit := func(i uint) bool {
			if atomic.LoadUint32(&(bitVec[i/32]))&(uint32(1)<<(i%32)) == 0 {
				return false
			} else {
				return true
			}
		}

		setBit := func(i uint) {
			mtxVec[i/32].Lock()
			bitVec[i/32] |= (uint32(1) << (i % 32))
			mtxVec[i/32].Unlock()
		}

		// initialzie bitvec
		parallel_for(workers, func(wIdx int) {
			for i := wIdx; i < len(bitVec); i += workers {
				bitVec[i] = 0
			}
		})

		// set bitvec
		parallel_for(workers, func(wIdx int) {
			for i := wIdx + 1; i < len(deduped); i += workers {
				if deduped[i-1][0] != deduped[i][0] {
					setBit(uint(i))
				}
			}
		})
		setBit(0)

		// prepare prefix sum result slice
		pSumRes := make([]uint, len(deduped)+1)
		pSumRes[0] = 0

		// preprocess for prefix sum
		parallel_for(workers, func(wIdx int) {
			for i := wIdx + 1; i < len(pSumRes); i += workers {
				if getBit(uint(i - 1)) {
					pSumRes[i] = 1
				} else {
					pSumRes[i] = 0
				}
			}
		})

		// reduction sum
		ones := reductionSum(pSumRes)

		// exclusive prefix sum
		pSumResPart := pSumRes[1:]
		prefixSumInclusive(&pSumResPart)

		var wg2 sync.WaitGroup

		wg2.Add(1)
		go func() {
			defer wg2.Done()
			row := make([]uint32, ones)
			parallel_for(workers, func(wIdx int) {
				for i := wIdx; i < len(deduped); i += workers {
					if getBit(uint(i)) {
						row[pSumRes[i]] = deduped[i][0]
					}
				}
			})

			if err := fileSave(fileStem+".row", u32s_bs(row)); err != nil {
				panic(err)
			}
		}()

		wg2.Add(1)
		go func() {
			defer wg2.Done()
			ptr := make([]uint32, ones+1)
			parallel_for(workers, func(wIdx int) {
				for i := wIdx; i < len(deduped); i += workers {
					if getBit(uint(i)) {
						ptr[pSumRes[i]] = uint32(i)
					}
				}
			})
			ptr[len(ptr)-1] = uint32(len(deduped))

			if err := fileSave(fileStem+".ptr", u32s_bs(ptr)); err != nil {
				panic(err)
			}
		}()

		wg2.Wait()
	}()

	wg.Wait()
}

func stage2(outFolder, outName string) {
	absOutFolder, err := filepath.Abs(outFolder)
	if err != nil {
		panic(err)
	}

	targetFolder := filepath.Join(absOutFolder, outName)

	fListChan := fileList(targetFolder, ".el32")

	parallel_for(8, func(i int) {
		for fPath := range fListChan {
			absfPath, err := filepath.Abs(fPath)
			if err != nil {
				panic(err)
			}

			el32, err := fileLoad(absfPath)
			if err != nil {
				panic(err)
			}

			// deduplication of edgelist
			deduped := stage2_dedup(bs_u32a2s(el32))

			folder, filename := filepath.Split(absfPath)
			filename2 := strings.Split(filename, ".")[0]
			stage2_save(filepath.Join(folder, filename2), deduped)
			os.Remove(absfPath)
		}
	})
}
