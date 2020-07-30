package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func appendFile(filename string, text string) {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}

	defer f.Close()

	if _, err = f.WriteString(text); err != nil {
		panic(err)
	}
}

func saveFileAppend(path string, in []byte) {
	info, err := os.Stat(path)
	if err != nil {
		log.Panicln(err)
	}

	if info.IsDir() {
		return
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Panicln(err)
	}
	defer file.Close()

	for pos := 0; pos < len(in); {
		b, err := file.Write(in[pos:])
		if err != nil {
			log.Panicln(err)
		}
		pos += b
	}
}

func loadFile(path string) []byte {
	info, err := os.Stat(path)
	if err != nil {
		log.Panicln(err)
	}

	if info.IsDir() {
		return []byte{}
	}

	file, err := os.Open(path)
	if err != nil {
		log.Panicln(err)
	}
	defer file.Close()

	out := make([]uint8, info.Size())

	for pos := int64(0); pos < info.Size(); {
		b, err := file.ReadAt(out[pos:], pos)
		if err != nil {
			log.Panicln(err)
		}
		pos += int64(b)
	}

	return out
}

func fileSize(filename string) int64 {
	info, err := os.Stat(filename)
	if err != nil {
		panic(err)
	}
	return info.Size()
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

type Setting struct {
	RMAT, shard, stream, block, thread int
}

func runCommand(app, inputData string, setting Setting) (uint64, float64) {
	cmd := exec.Command(app,
		inputData,
		strconv.Itoa(setting.stream),
		strconv.Itoa(setting.block),
		strconv.Itoa(setting.thread))

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

	return myTriangle, myRealtime
}

func main() {
	logFile := "log-GCSR-Reorder2-Quad.txt"
	app := "./build/TriangleCounting-LowerTriangular-BitArray-Manual-no-meta-stream-quad"

	if stat, err := os.Stat(logFile); stat.Size() == 0 && err == nil {
	} else if err != nil {
		log.Fatalln(err)
	}

	var fileContent string

	if fileExists(logFile) {
		fileContent = string(loadFile(logFile))
	} else {
		saveFileAppend(logFile, []byte("RMAT,shard,stream,block,thread,triangle,time\n"))
	}

	for N := 20; N <= 30; N++ {
		for shard := 29; shard <= 33; shard++ {
			inputData := "/mnt/nvme/GCSR-Reorder2-Quad-" + strconv.Itoa(shard) + "/RMAT" + strconv.Itoa(N) + "-24-lt-reorder2-quad-" + strconv.Itoa(shard) + "/"

			for stream := 1; stream <= 2; stream++ {
				for _, block := range []int{320, 480, 640, 960, 1280} {
					for _, thread := range []int{128, 256, 512, 768, 1024} {
						// Check
						setting := Setting{
							RMAT:   N,
							shard:  shard,
							stream: stream,
							block:  block,
							thread: thread,
						}

						targetString := fmt.Sprintf("%d,%d,%d,%d,%d,",
							setting.RMAT,
							setting.shard,
							setting.stream,
							setting.block,
							setting.thread)

						if strings.Contains(fileContent, targetString) {
							log.Printf("%v (Already Done)", targetString)
							continue
						}

						triangle, time := runCommand(app, inputData, setting)

						finalOutput := fmt.Sprintf("%v,%d,%.6f\n", targetString, triangle, time)

						log.Printf("%v", finalOutput)
						saveFileAppend(logFile, []byte(finalOutput))
					}
				}
			}

		}
	}
}
