package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"time"
	"unsafe"
)

var (
	ctx = context.Background()
)

const (
	wordByte = 6
	gWidth   = 1 << 24
)

type vertex32 uint32
type edge32 [2]vertex32
type gidx32 [2]uint32

type sRawDat struct {
	src, cnt, dstStart uint64
}

type gridEdge struct {
	gidx gidx32
	edge edge32
}

func filenameEncode(gidx gidx32, ext string) string {
	return strconv.Itoa(int(gidx[0])) + "-" + strconv.Itoa(int(gidx[1])) + ext
}

func edge32SliceToByteSlice(in []edge32) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	const edge32Byte = 8
	header.Len = len(in) * edge32Byte
	header.Cap = len(in) * edge32Byte

	return *(*[]byte)(unsafe.Pointer(&header))
}

func convBE6toLE8(in []uint8) uint64 {
	var temp uint64 = 0
	temp |= uint64(in[0])

	const uint64Byte = 8

	for i := 1; i < wordByte; i++ {
		temp <<= uint64Byte
		temp |= uint64(in[i])
	}

	return temp
}

func walk() <-chan string {
	out := make(chan string, 16)
	go func() {
		defer close(out)
		inFolder := ctx.Value("inFolder").(string)
		filepath.Walk(inFolder, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				log.Panicln(err)
				return err
			}
			if info.IsDir() || info.Size() == 0 {
				return nil
			}
			absPath, _ := filepath.Abs(path)
			out <- absPath

			return nil
		})
	}()
	return out
}

func loader(path string) []uint8 {
	info, err := os.Stat(path)
	if err != nil {
		log.Panicln(err)
	}

	file, err := os.Open(path)
	if err != nil {
		log.Panicln(err)
	}
	defer file.Close()

	out := make([]uint8, info.Size())

	// If file is over 1GB, you should iteratively load files
	for pos := int64(0); pos < info.Size(); {
		b, err := file.ReadAt(out[pos:], pos)
		pos += int64(b)
		if err != nil {
			log.Panicln(err)
		}
	}

	return out
}

func splitter(adj6 []uint8) <-chan sRawDat {
	out := make(chan sRawDat, 128)
	go func() {
		defer close(out)
		for i := uint64(0); i < uint64(len(adj6)); {
			s := convBE6toLE8(adj6[i : i+wordByte])
			i += wordByte
			c := convBE6toLE8(adj6[i : i+wordByte])
			i += wordByte

			out <- sRawDat{
				src:      s,
				cnt:      c,
				dstStart: i,
			}

			i += wordByte * c
		}
	}()
	return out
}

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

func routine() {
	files := walk()

	targetFolder := filepath.Join(ctx.Value("outFolder").(string), ctx.Value("outName").(string))
	os.MkdirAll(targetFolder, 0755)

	// Tuned for best runtime
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

func init() {
	inFolder := flag.String("in.folder", "", "Input Folder")
	outFolder := flag.String("out.folder", "", "Output Folder")
	outName := flag.String("out.name", "", "Output Name")
	flag.Parse()

	if len(*inFolder) == 0 || len(*outFolder) == 0 || len(*outName) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	ctx = context.WithValue(ctx, "inFolder", *inFolder)
	ctx = context.WithValue(ctx, "outFolder", *outFolder)
	ctx = context.WithValue(ctx, "outName", *outName)
}

func main() {
	{
		log.Println("Phase 1 (Adj6->Edgelist) Start")
		start := time.Now()

		routine()

		elapsed := time.Since(start).Seconds()
		log.Println("Phase 1 (Adj6->Edgelist) Complete, Elapsed Time:", elapsed, "(sec)")
	}
}
