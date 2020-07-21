package main

import (
	"log"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

type resultType struct {
	stream, block, thread int
	triangle              uint64
	realtime              float64
}

func main() {
	TOTAL := 1024 * 160
	APP := "./build/TriangleCounting-LowerTriangular-BinarySearch-Manual-no-meta-stream"

	for N := 30; N <= 30; N++ {
		data := "/mnt/nvme/GCSR/RMAT" + strconv.Itoa(N) + "-24-lt/"

		result := []resultType{}

		for stream := 1; stream <= 1; stream++ {
			for block := 160 * 2; block <= 160*(1<<3); block *= 2 {
				thread := 0
				if block < 160 {
					thread = 1024
				} else {
					thread = TOTAL / block
				}

				log.Printf("RMAT=%v,S=%4d,block=%4d,thread=%4d Start\n", N, stream, block, thread)

				cmd := exec.Command(APP, data, strconv.Itoa(stream), strconv.Itoa(block), strconv.Itoa(thread))
				b, _ := cmd.Output()
				splitted := strings.Split(string(b), "\n")

				var myTriangle uint64
				var myRealtime float64
				var err error

				for _, line := range splitted {
					if strings.Contains(line, "total triangles") {
						val := strings.Split(line, ": ")[1]
						myTriangle, err = strconv.ParseUint(val, 10, 64)
						if err != nil {
							panic(err)
						}
					}
					if strings.Contains(line, "REALTIME") {
						val := strings.Split(line, ": ")[1]
						myRealtime, err = strconv.ParseFloat(val, 64)
						if err != nil {
							panic(err)
						}
					}
				}

				log.Printf("RMAT=%v,S=%4d,block=%4d,thread=%4d,Triangle=%v,Time=%.6f\n", N, stream, block, thread, myTriangle, myRealtime)

				result = append(result, resultType{
					stream:   stream,
					block:    block,
					thread:   thread,
					triangle: myTriangle,
					realtime: myRealtime,
				})
			}
		}

		sort.Slice(result, func(i, j int) bool {
			return result[i].realtime < result[j].realtime
		})

		log.Printf("BEST--------------RMAT=%v,S=%4d,block=%4d,thread=%4d,Triangle=%v,Time=%.6f\n", N, result[0].stream, result[0].block, result[0].thread, result[0].triangle, result[0].realtime)
	}
}
