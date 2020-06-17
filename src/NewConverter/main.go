package main

import (
	"context"
	"flag"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"time"
	"unsafe"
)

var (
	ctx = context.Background()
)

const (
	wordSz   = 6
	gWidth   = 1 << 24
	chanSz   = 64
	mapperSz = 16
	workerSz = 8
)

type Edge64 struct {
	src, dst uint64
}

type GridIndex32 struct {
	row, col uint32
}

type Edge32 struct {
	src, dst uint32
}

type GridEdge struct {
	gridIndex GridIndex32 // key
	localEdge Edge32      // value
}

type FileInfo struct {
	path  string
	bsize int64
}

type Listb struct {
	src uint64
	dst []byte
}

func convBE6toLE8(in []byte) uint64 {
	return (uint64(in[0]) << (8 * 5)) +
		(uint64(in[1]) << (8 * 4)) +
		(uint64(in[2]) << (8 * 3)) +
		(uint64(in[3]) << (8 * 2)) +
		(uint64(in[4]) << (8 * 1)) +
		(uint64(in[5]) << (8 * 0))
}

func filename(gidx GridIndex32) string {
	return strconv.FormatInt(int64(gidx.row), 10) + "-" + strconv.FormatInt(int64(gidx.col), 10)
}

func walkFile(inFolder string) []FileInfo {
	//defer log.Println("Finish walkFile", inFolder)
	paths := []FileInfo{}
	filepath.Walk(inFolder, func(path string, info os.FileInfo, err error) error {
		file, err := os.Open(path)
		if err != nil {
			return nil
		}

		defer file.Close()

		if info.IsDir() || info.Size() == 0 {
			return nil
		}

		absPath, _ := filepath.Abs(path)
		paths = append(paths, FileInfo{
			path:  absPath,
			bsize: info.Size(),
		})

		return nil
	})

	return paths
}

func loader(inFile FileInfo) []byte {

	file, _ := os.Open(inFile.path)
	defer func() {
		file.Close()
		//log.Println("Finish loader", inFile.path)
	}()

	out := make([]byte, inFile.bsize)
	file.Read(out)
	return out
}

func splitter(in []byte) <-chan Listb {
	out := make(chan Listb, chanSz)

	go func() {
		defer func() {
			close(out)
			//log.Println("Finish splitter")
		}()

		for i := 0; i < len(in); {
			src := convBE6toLE8(in[i : i+wordSz])
			i += wordSz

			cnt := convBE6toLE8(in[i : i+wordSz])
			i += wordSz

			out <- Listb{
				src: src,
				dst: in[i : i+(wordSz*int(cnt))],
			}
			i += wordSz * int(cnt)
		}
	}()

	return out
}

func mapper(listChan <-chan Listb) <-chan []GridEdge {
	out := make(chan []GridEdge, chanSz)
	go func() {
		defer func() {
			close(out)
			//log.Println("Finish mapper")
		}()

		for list := range listChan {
			el := []GridEdge{}

			for i := 0; i < len(list.dst); i += wordSz {
				s := list.src
				d := convBE6toLE8(list.dst[i : i+wordSz])

				if s < d {
					s, d = d, s //swap
				} else if s == d {
					continue
				}

				el = append(el, GridEdge{
					gridIndex: GridIndex32{
						row: uint32(s / gWidth),
						col: uint32(d / gWidth),
					},
					localEdge: Edge32{
						src: uint32(s % gWidth),
						dst: uint32(d % gWidth),
					},
				})
			}

			out <- el
		}
	}()

	return out
}

func shuffler(
	wgShuffler *sync.WaitGroup,
	wgWriter *sync.WaitGroup,
	writerEntry map[GridIndex32](chan []Edge32),
	writerEntryMtx *sync.Mutex,
	in <-chan []GridEdge,
) {
	go func() {
		defer func() {
			wgShuffler.Done()
			//log.Println("Finish shuffler")
		}()

		for el := range in {
			kv := make(map[GridIndex32][]Edge32)
			for _, e := range el {
				kv[e.gridIndex] = append(kv[e.gridIndex], e.localEdge)
			}

			for k, v := range kv {
				writerEntryMtx.Lock()
				if _, ok := writerEntry[k]; !ok {
					writerEntry[k] = make(chan []Edge32, chanSz)
					wgWriter.Add(1)
					go writer(wgWriter, k, writerEntry[k])
				}
				writerEntryMtx.Unlock()

				writerEntry[k] <- v
			}
		}
	}()
}

func cleanup(
	wgShuffler *sync.WaitGroup,
	wgWriter *sync.WaitGroup,
	writerEntry map[GridIndex32](chan []Edge32),
) {
	defer func() {
		//log.Println("Finish cleanup")
	}()

	wgShuffler.Wait()

	for k, v := range writerEntry {
		close(v)
		delete(writerEntry, k)
	}

	wgWriter.Wait()
}

func writer(wgWriter *sync.WaitGroup, gidx GridIndex32, in <-chan []Edge32) {

	// Create folder and file
	outFolder := ctxString(ctx, "outFolder") + "/" + ctxString(ctx, "outName") + "/"

	if _, err := os.Stat(outFolder); os.IsNotExist(err) {
		os.MkdirAll(outFolder, 0755)
	}

	outFile := outFolder + filename(gidx) + ".el"

	file, err := os.OpenFile(outFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Panicln(err)
	}

	defer func() {
		file.Close()
		wgWriter.Done()
		//log.Println("Finish writer")
	}()

	// processing
	for el := range in {
		barr := castEdge32ToByte(el)
		// Write
		if _, err := file.Write(barr); err != nil {
			log.Panicln(err)
		}
	}

}

func init() {
	inFolder := flag.String("in.folder", "", "Folder path for input files")
	outFolder := flag.String("out.folder", "", "Folder path for input files")
	outName := flag.String("out.name", "", "Folder path for input files")
	flag.Parse()

	if len(os.Args) == 0 || *inFolder == "" {
		flag.Usage()
		os.Exit(1)
	}

	ctx = context.WithValue(ctx, "inFolder", *inFolder)
	ctx = context.WithValue(ctx, "outFolder", *outFolder)
	ctx = context.WithValue(ctx, "outName", *outName)
}

func phase1() {
	fn := func(finfo FileInfo) {
		input := loader(finfo) // sync

		listb := splitter(input) // async

		var wgShuffler sync.WaitGroup
		var wgWriter sync.WaitGroup
		writerEntry := make(map[GridIndex32](chan []Edge32))
		writerEntryMtx := &sync.Mutex{}

		mapped := make([](<-chan []GridEdge), mapperSz)
		wgShuffler.Add(len(mapped))
		for i := range mapped {
			mapped[i] = mapper(listb)                                                // async
			shuffler(&wgShuffler, &wgWriter, writerEntry, writerEntryMtx, mapped[i]) // async
		}

		cleanup(&wgShuffler, &wgWriter, writerEntry) // sync
		log.Println("Complete", finfo.path)
	}

	// Boss
	jobs := func() <-chan FileInfo {
		out := make(chan FileInfo, chanSz)
		go func() {
			defer close(out)
			for _, f := range walkFile(ctxString(ctx, "inFolder")) {
				out <- f
			}
		}()
		return out
	}()

	// Worker
	var wg sync.WaitGroup
	for worker := 0; worker < workerSz; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				fn(job)
			}
		}()
	}

	wg.Wait()
}

func castByteToEdge32(in []byte) []Edge32 {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len /= (4 * 2)
	header.Cap /= (4 * 2)
	return *(*[]Edge32)(unsafe.Pointer(&header))
}

func castEdge32ToByte(in []Edge32) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len *= (4 * 2)
	header.Cap *= (4 * 2)
	return *(*[]byte)(unsafe.Pointer(&header))
}

type Edgelist32 []Edge32

func (e Edgelist32) Len() int {
	return len(e)
}
func (e Edgelist32) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}
func (e Edgelist32) Less(i, j int) bool {
	return (e[i].src < e[j].src) || (e[i].src == e[j].src && e[i].dst < e[j].dst)
}

func removeDuplicated(in []Edge32) []Edge32 {
	// sort first
	sort.Sort(Edgelist32(in))

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
		leftshift := func(arr []Edge32, start, end, shift int) {
			tmp := make([]Edge32, end-start)
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

func phase2() {
	fn := func(finfo FileInfo) {
		input := castByteToEdge32(loader(finfo))
		cleaned := removeDuplicated(input)
		// 중복찾기 하고

		// 변환
		//log.Println(cleaned)
		/*
			lesswap := func(i, k, r, s int) bool {
				less := func(i, k int) bool {
					return (enput[i].src < enput[k].src) || (enput[i].src == enput[k].src && enput[i].dst < enput[k].dst)
				}

				swap := func(r, s int) {
					enput[r], enput[s] = enput[s], enput[r]
				}

				if less(i, k) {
					if r != s {
						swap(r, s)
					}
					return true
				}
				return false
			}

			sorty.Sort(len(enput), lesswap)
		*/

		os.Remove(finfo.path)

		log.Println("Complete", finfo.path, "->")
	}

	// Boss
	jobs := func() <-chan FileInfo {
		out := make(chan FileInfo, chanSz)
		go func() {
			defer close(out)
			for _, f := range walkFile(ctxString(ctx, "outFolder")) {
				out <- f
			}
		}()
		return out
	}()

	// Worker
	var wg sync.WaitGroup
	for worker := 0; worker < workerSz; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				fn(job)
			}
		}()
	}

	wg.Wait()
}

func main() {
	elapsedPhase1 := func() float64 {
		log.Println("Phase 1 Start (Asynchronous Map-Reduce)")
		start := time.Now()
		phase1()
		duration := time.Since(start)
		log.Println("Phase 1 Done. (Sync all thread)")
		return duration.Seconds()
	}()

	elapsedPhase2 := func() float64 {
		log.Println("Phase 2 Start (Edgelist -> CSR)")
		start := time.Now()
		phase2()
		duration := time.Since(start)
		log.Println("Phase 2 Done.")
		return duration.Seconds()
	}()

	log.Println("Convert Finished; Phase 1 time", elapsedPhase1, "(sec)")
	log.Println("Convert Finished; Phase 2 time", elapsedPhase2, "(sec)")
	log.Println("Convert Finished; Total   time", elapsedPhase1+elapsedPhase2, "(sec)")
}
