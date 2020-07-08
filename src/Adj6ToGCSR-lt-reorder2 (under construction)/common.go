package main

import (
	"runtime"
	"strconv"
)

func fileNameEncode(gidx [2]uint32, extension string) string {
	return strconv.Itoa(int(gidx[0])) + "-" + strconv.Itoa(int(gidx[1])) + extension
}

func reductionSum(in []uint) uint {
	workers := runtime.NumCPU()

	batchLength := uint(32)
	batchCount := ceil(uint(len(in)), batchLength)

	tmp := make([]uint, workers)

	parallel_for(workers, func(wIdx int) {
		myResult := uint(0)

		for b := uint(wIdx); b < batchCount; b += uint(workers) {
			if b != batchCount-1 {
				for off := uint(0); off < batchLength; off++ {
					myResult += uint(in[b*batchLength+off])
				}
			} else {
				// last batch
				for off := uint(0); off < uint(len(in))-b*batchLength; off++ {
					myResult += uint(in[b*batchLength+off])
				}
			}
		}

		tmp[wIdx] = myResult
	})

	totalResult := uint(0)
	for _, v := range tmp {
		totalResult += v
	}

	return totalResult
}

func prefixSumInclusive(in *[]uint) {
	workers := runtime.NumCPU()

	batchLength := uint(32)
	batchCount := ceil(uint(len(*in)), batchLength)

	tmp := make([]uint, batchCount-1)

	parallel_for(workers, func(wIdx int) {
		for b := uint(wIdx); b < batchCount; b += uint(workers) {
			if b != batchCount-1 {
				for off := uint(1); off < batchLength; off++ {
					idx := b*batchLength + off
					(*in)[idx] += (*in)[idx-1]
				}
				tmp[b] = (*in)[b*batchLength+(batchLength-1)]
			} else {
				// last batch
				for off := uint(1); off < uint(len(*in))-b*batchLength; off++ {
					idx := b*batchLength + off
					(*in)[idx] += (*in)[idx-1]
				}
			}
		}
	})

	for i := 1; i < len(tmp); i++ {
		tmp[i] += tmp[i-1]
	}

	parallel_for(workers, func(wIdx int) {
		for b := uint(wIdx) + 1; b < batchCount; b += uint(workers) {
			if b != batchCount-1 {
				for off := uint(0); off < batchLength; off++ {
					idx := b*batchLength + off
					(*in)[idx] += tmp[b-1]
				}
			} else {
				// last batch
				for off := uint(0); off < uint(len(*in))-b*batchLength; off++ {
					idx := b*batchLength + off
					(*in)[idx] += tmp[b-1]
				}
			}
		}
	})
}

/*
func prefixSumUint(in []uint, exclusive bool) []uint {
	workers := runtime.NumCPU()

	batchLength := uint(32)
	batchCount := ceil(uint(len(in)), batchLength)

	tmp := make([]uint, batchCount-1)
	var trueOut, tempOut []uint

	if exclusive {
		trueOut = make([]uint, len(in)+1)
		trueOut[0] = 0
		tempOut = trueOut[1:]
	} else {
		trueOut = make([]uint, len(in))
		tempOut = trueOut
	}

	parallel_for(workers, func(wIdx int) {
		for b := uint(wIdx); b < batchCount; b += uint(workers) {
			tempOut[b*batchLength] = in[b*batchLength]
			if b != batchCount-1 {
				for off := uint(1); off < batchLength; off++ {
					idx := b*batchLength + off
					tempOut[idx] = tempOut[idx-1] + in[idx]
				}
				tmp[b] = tempOut[b*batchLength+(batchLength-1)]
			} else {
				// last batch
				for off := uint(1); off < uint(len(in))-b*batchLength; off++ {
					idx := b*batchLength + off
					tempOut[idx] = tempOut[idx-1] + in[idx]
				}
			}
		}
	})

	for i := 1; i < len(tmp); i++ {
		tmp[i] += tmp[i-1]
	}

	parallel_for(workers, func(wIdx int) {
		for b := uint(wIdx) + 1; b < batchCount; b += uint(workers) {
			if b != batchCount-1 {

				for off := uint(0); off < batchLength; off++ {
					tempOut[b*batchLength+off] += tmp[b-1]
				}
			} else {
				// last batch
				for off := uint(0); off < uint(len(in))-b*batchLength; off++ {
					tempOut[b*batchLength+off] += tmp[b-1]
				}
			}
		}
	})

	return trueOut
}
*/

func split_be6(adj6 []uint8) <-chan sRawDat {
	out := make(chan sRawDat, 128)
	go func() {
		defer close(out)

		for i := uint64(0); i < uint64(len(adj6)); {
			s := be6_le8(adj6[i : i+6])
			i += 6
			c := be6_le8(adj6[i : i+6])
			i += 6

			out <- sRawDat{
				src:      s,
				cnt:      c,
				dstStart: i,
			}

			i += (6 * c)
		}
	}()
	return out
}
