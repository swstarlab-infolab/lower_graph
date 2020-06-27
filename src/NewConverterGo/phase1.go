package main

import (
	"log"
	"os"
	"path/filepath"
	"sync"
)

func mapper(adj6 []uint8, in <-chan sRawDat) <-chan []gridEdge {
	out := make(chan []gridEdge, 128)
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

func shuffler(
	in <-chan []gridEdge,
	wgShuffler *sync.WaitGroup,
	targetFolder string,
) {
	go func() {
		defer wgShuffler.Done()

		temp := make(map[gidx32]([]edge32))

		for dat := range in {
			for _, ge := range dat {
				if _, ok := temp[ge.gidx]; !ok {
					temp[ge.gidx] = []edge32{}
				}
				temp[ge.gidx] = append(temp[ge.gidx], ge.edge)
			}

			for k, v := range temp {
				// (1 << 20) x sizeof(edge32) = 1M x 8B = 8MB
				if len(v) > (1 << 20) {
					targetFile := filepath.Join(targetFolder, filenameEncode(k, ".el32"))
					file, err := os.OpenFile(targetFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)

					if err != nil {
						log.Panic(err)
					}
					_, err = file.Write(edge32SliceToByteSlice(v))
					if err != nil {
						log.Panic(err)
					}
					file.Close()

					// shrink size
					temp[k] = v[:0]
				}
			}
		}

		for k, v := range temp {
			if len(v) > 0 {

				targetFile := filepath.Join(targetFolder, filenameEncode(k, ".el32"))
				file, err := os.OpenFile(targetFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)

				if err != nil {
					log.Panic(err)
				}
				_, err = file.Write(edge32SliceToByteSlice(v))
				if err != nil {
					log.Panic(err)
				}
				file.Close()
			}
		}

	}()
}

//func phase1(info information) {
func phase1() {
	files := walk()

	targetFolder := filepath.Join(ctx.Value("outFolder").(string), ctx.Value("outName").(string))
	os.MkdirAll(targetFolder, 0755)

	workers := 4
	shufflers := 32

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(in <-chan string) {
			defer wg.Done()

			for file := range in {
				var wgShuffler sync.WaitGroup

				wgShuffler.Add(shufflers)

				data := loader(file)
				splitPos := splitter(data)
				for i := 0; i < shufflers; i++ {
					mapped := mapper(data, splitPos)
					shuffler(mapped, &wgShuffler, targetFolder)
				}

				wgShuffler.Wait()

				log.Println("Phase 1 (Adj6->Edgelist)", file, "Finished")
			}
		}(files)
	}

	wg.Wait()
}
