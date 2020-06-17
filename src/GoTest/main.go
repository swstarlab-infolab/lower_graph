package main

import (
	"log"
	"math"
	"sort"
)

func removeDuplicated(in []int) []int {
	// sort first
	sort.Ints(in)

	dup := make([]uint8, int(math.Ceil(float64(len(in))/3.0)))

	// create bit array
	{
		setBit := func(barr []uint8, idx int) {
			barr[idx>>3] |= (0x1 << (idx & 7))
		}

		for i := 0; i < len(in)-1; i++ {
			if in[i] == in[i+1] {
				setBit(dup, i)
			}
		}
	}

	// set
	{
		readBit := func(barr []uint8, idx int) bool {
			if ((0x1 << (idx & 7)) & barr[idx>>3]) == 0 {
				return false
			} else {
				return true
			}
		}

		// shifting function
		leftshift := func(arr []int, start, end, shift int) {
			tmp := make([]int, end-start)
			tmp = arr[start:end]

			for idx, v := range tmp {
				arr[start-shift+idx] = v
			}
		}

		// shift elements of slice
		start, end, shift := 0, 0, 0

		for range in {
			if readBit(dup, end) {
				if end > start {
					leftshift(in, start, end, shift)
					end++
					start = end
					shift++
				} else {
					end++
					start = end
					shift++
				}
			} else {
				end++
			}
		}

		// last element handling
		if end > start {
			leftshift(in, start, end, shift)
		}

		// slice size change
		return in[:len(in)-shift]
	}
}

func main() {
	in := []int{1, 2, 2, 2, 3, 3, 3, 6, 6}
	log.Println(in)
	out := removeDuplicated(in)
	log.Println(out)
}
