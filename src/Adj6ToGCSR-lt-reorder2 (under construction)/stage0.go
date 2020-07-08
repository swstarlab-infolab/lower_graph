package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jfcg/sorty"
)

func stage0(inFolder string, reorderType int) []uint64 {

	minVID, maxVID := ^uint64(0), uint64(0)
	mtxMinMaxVID := &sync.Mutex{}

	stopwatch("find min/max VID", func() {
		fListChan := fileList(inFolder, "")

		parallel_for(8, func(i int) {
			for fPath := range fListChan {
				adj6, err := fileLoad(fPath)
				if err != nil {
					panic(err)
				}

				sRawDatChan := split_be6(adj6)

				parallel_for(64, func(j int) {
					myMinVID, myMaxVID := ^uint64(0), uint64(0)

					for dat := range sRawDatChan {
						for i := uint64(0); i < dat.cnt; i++ {
							s := dat.src
							now := dat.dstStart + i*6
							d := be6_le8(adj6[now : now+6])

							// update min/max vertex id
							myMinVID = min(myMinVID, min(s, d))
							myMaxVID = max(myMaxVID, max(s, d))
						}
					}

					mtxMinMaxVID.Lock()
					minVID = min(minVID, myMinVID)
					maxVID = max(maxVID, myMaxVID)
					mtxMinMaxVID.Unlock()
				})
			}
		})
	})

	log.Println("min vertex ID:", minVID)
	log.Println("max vertex ID:", maxVID)

	rSlice := make([]reorder, maxVID+1)
	stopwatch("init rSlice", func() {
		workers := runtime.NumCPU()
		parallel_for(workers, func(wIdx int) {
			for i := wIdx; i < len(rSlice); i += workers {
				rSlice[i].key = uint64(i)
				rSlice[i].val = 0
			}
		})
	})

	stopwatch("count degree", func() {
		fListChan := fileList(inFolder, "")
		parallel_for(8, func(i int) {
			for fPath := range fListChan {
				adj6, err := fileLoad(fPath)
				if err != nil {
					panic(err)
				}

				sRawDatChan := split_be6(adj6)

				parallel_for(64, func(j int) {
					for dat := range sRawDatChan {
						s := dat.src
						atomic.AddUint64(&(rSlice[s].val), dat.cnt)

						for i := uint64(0); i < dat.cnt; i++ {
							now := dat.dstStart + i*6
							d := be6_le8(adj6[now : now+6])

							if s != d {
								atomic.AddUint64(&(rSlice[d].val), 1)
							} else {
								//subtract 1
								atomic.AddUint64(&(rSlice[s].val), ^uint64(1-1))
							}
						}
					}
				})
			}
		})
	})

	stopwatch("reorder vertices by rank", func() {
		var comp func(i, j int) bool

		switch reorderType {
		case 1:
			comp = func(i, j int) bool {
				if rSlice[i].val == 0 && rSlice[j].val != 0 {
					return false
				} else if rSlice[i].val != 0 && rSlice[j].val == 0 {
					return true
				} else if rSlice[i].val == 0 && rSlice[j].val == 0 {
					return false
				} else {
					if rSlice[i].val == rSlice[j].val {
						return rSlice[i].key < rSlice[j].key
					} else {
						return rSlice[i].val < rSlice[j].val
					}
				}
			}
		case 2:
			comp = func(i, j int) bool {
				if rSlice[i].val == rSlice[j].val {
					return rSlice[i].key > rSlice[j].key
				} else {
					return rSlice[i].val > rSlice[j].val
				}
			}
		default:
			panic(fmt.Errorf("Reorder type is wrong"))
		}

		sorty.Mxg = uint32(runtime.NumCPU()) * 2

		sorty.Sort(len(rSlice), func(i, k, r, s int) bool {
			if comp(i, k) {
				if r != s {
					rSlice[r], rSlice[s] = rSlice[s], rSlice[r]
				}
				return true
			}
			return false
		})

		workers := runtime.NumCPU()
		parallel_for(workers, func(idx int) {
			for i := idx; i < len(rSlice); i += workers {
				rSlice[i].val = uint64(i)
			}
		})

		sorty.Sort(len(rSlice), func(i, k, r, s int) bool {
			if rSlice[i].key < rSlice[k].key {
				if r != s {
					rSlice[r], rSlice[s] = rSlice[s], rSlice[r]
				}
				return true
			}
			return false
		})

	})

	out := reorders_u64s(rSlice)
	for i := 0; i < len(out); i++ {
		if i%2 == 1 {
			out[i/2] = out[i]
		}
	}

	out = out[:len(out)/2]

	//log.Println(out)
	return out
}
