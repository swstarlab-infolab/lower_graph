package main

import (
	"log"
	"os/exec"
	"strconv"
	"strings"
)

func main() {
	TOTAL := 1024 * 160
	APP := "./build/TriangleCounting-LowerTriangular-BinarySearch-Manual-no-meta"

	for R := 2; R <= 2; R++ {
		log.Println("Reorder", R)
		for N := 27; N <= 27; N++ {
			for S := 1; S <= 1; S++ {
				for B := 160 * 8; B <= 160*(1<<3); B *= 2 {
					T := TOTAL / B
					log.Printf("RMAT=%v,S=%v,B=%v,T=%v\n", N, S, B, T)
					cmd := exec.Command(APP, "/mnt/nvme/GCSR-Reorder"+strconv.Itoa(R)+"/RMAT"+strconv.Itoa(N)+"-24-lt-reorder"+strconv.Itoa(R)+"/", strconv.Itoa(S), strconv.Itoa(B), strconv.Itoa(T))
					b, _ := cmd.Output()
					splitted := strings.Split(string(b), "\n")

					var triangle uint64
					var realtime float64
					var err error

					for _, line := range splitted {
						if strings.Contains(line, "total triangles") {
							val := strings.Split(line, ": ")[1]
							triangle, err = strconv.ParseUint(val, 10, 64)
							if err != nil {
								panic(err)
							}
						}
						if strings.Contains(line, "REALTIME") {
							val := strings.Split(line, ": ")[1]
							realtime, err = strconv.ParseFloat(val, 64)
							if err != nil {
								panic(err)
							}
						}
					}

					log.Printf("Triangle=%v,Time=%.6f\n", triangle, realtime)

				}
			}
		}
	}
}
