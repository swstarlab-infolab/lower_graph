package main

import (
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"unsafe"
)

func mapper1(adj6 []uint8, in <-chan sRawDat) <-chan []gridEdge {
	out := make(chan []gridEdge, 16)
	go func() {
		defer close(out)
		for dat := range in {
			el := make([]gridEdge, dat.cnt)
			selfloop := uint64(0)

			for i := uint64(0); i < dat.cnt; i++ {
				s := dat.src
				now := dat.dstStart + i*wordByte
				d := convBE6toLE8(adj6[now : now+wordByte])

				if s < d {
					s, d = d, s
				} else if s == d {
					selfloop++
					continue
				}

				el[i-selfloop] = gridEdge{
					gidx: gidx32{uint32(s / gWidth), uint32(d / gWidth)},
					edge: edge32{vertex32(s % gWidth), vertex32(d % gWidth)},
				}
			}

			el = el[:uint64(len(el))-selfloop]
			out <- el
		}
	}()
	return out
}

func shuffler1(rMap reduceMap, in <-chan []gridEdge) {
	for dat := range in {
		temp := make(map[gidx32]([]edge32))
		for _, ge := range dat {
			if _, ok := temp[ge.gidx]; !ok {
				temp[ge.gidx] = []edge32{}
			}
			temp[ge.gidx] = append(temp[ge.gidx], ge.edge)
		}

		for k, v := range temp {
			rMap[k] <- v
		}
	}
}

func filenameEncode(gidx gidx32, ext string) string {
	return strconv.Itoa(int(gidx[0])) + "-" + strconv.Itoa(int(gidx[1])) + ext
}

func edge32SliceToByteSlice(in []edge32) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 8
	header.Cap = len(in) * 8

	return *(*[]byte)(unsafe.Pointer(&header))
}

func writer1(gidx gidx32, in <-chan []edge32, targetFolder string) {
	targetFile := filepath.Join(targetFolder, filenameEncode(gidx, ".el32"))
	file, err := os.OpenFile(targetFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Panic(err)
	}
	defer file.Close()

	for v := range in {
		file.Write(edge32SliceToByteSlice(v))
	}
}

func routine1(rMap reduceMap, in <-chan string) {
	var wgShuffler sync.WaitGroup

	shufflers := 32

	for file := range in {
		data := loader(file)
		splitPos := splitter(data)
		for i := 0; i < shufflers; i++ {
			mapped := mapper1(data, splitPos)
			wgShuffler.Add(1)
			go func(c <-chan []gridEdge) {
				defer wgShuffler.Done()
				shuffler1(rMap, c)
			}(mapped)
		}

		//log.Println("Phase 1 (Adj6->Edgelist)", file, "done")
	}

	wgShuffler.Wait()
}

func phase1(info information) {
	files := walk()

	rMap := make(reduceMap)
	minGidx, maxGidx := info.minvid/gWidth, info.maxvid/gWidth

	if minGidx > uint64(^uint32(0)) {
		log.Panicln("Phase 1 Out of range minGidx", minGidx)
	}

	if maxGidx > uint64(^uint32(0)) {
		log.Panicln("Phase 1 Out of range maxGidx", maxGidx)
	}

	outFolder := ctx.Value("outFolder").(string)
	outName := ctx.Value("outName").(string)

	targetFolder := filepath.Join(outFolder, outName)

	os.MkdirAll(targetFolder, 0755)

	var wgWriter, wgRoutine sync.WaitGroup

	// prepare
	for row := uint32(minGidx); row <= uint32(maxGidx); row++ {
		for col := uint32(minGidx); col <= uint32(maxGidx); col++ {
			key := gidx32{row, col}
			val := make(chan []edge32, 16)

			rMap[key] = val

			wgWriter.Add(1)
			go func(k gidx32, v <-chan []edge32) {
				defer wgWriter.Done()
				writer1(k, v, targetFolder)
			}(key, val)
		}
	}
	// init map

	workers := 16

	for i := 0; i < workers; i++ {
		wgRoutine.Add(1)
		go func() {
			defer wgRoutine.Done()
			routine1(rMap, files)
		}()
	}

	wgRoutine.Wait()

	log.Println("Phase 1 (Adj6->Edgelist)", "Computation thread done")

	for _, v := range rMap {
		close(v)
	}

	wgWriter.Wait()

	filepath.Walk(targetFolder, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}

		if filepath.Ext(path) == ".el32" && info.Size() == 0 {
			os.Remove(path)
			return nil
		}

		return nil
	})

	log.Println("Phase 1 (Adj6->Edgelist)", "Disk Writing done")
}
