package main

import (
	"log"
	"os/exec"
	"strconv"
	"strings"
)

func main() {
	APP := "./build/TriangleCounting-LowerTriangular-BinarySearch-Manual-no-meta-stream"

	for N := 33; N <= 33; N++ {
		data := "/mnt/nvme/GCSR/RMAT" + strconv.Itoa(N) + "-24-lt/"

		for _, stream := range []int{1, 2, 3} {
			for _, block := range []int{640, 1280} {
				for _, thread := range []int{512, 1024} {
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
				}
			}
		}
	}
}
