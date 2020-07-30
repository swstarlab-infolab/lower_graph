package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jfcg/sorty"
)

var (
	ctx            = context.Background()
	minVID, maxVID = ^uint64(0), uint64(0)
	mtxMinMaxVID   = &sync.Mutex{}
)

const (
	wordByte = 6
	gWidth   = 1 << 24
)

// value can be degree and new vid
type relabel struct {
	key, val uint64
}

type sRawDat struct {
	src, cnt, dstStart uint64
}

func max(a, b uint64) uint64 {
	if a > b {
		return a
	} else {
		return b
	}
}

func min(a, b uint64) uint64 {
	if a < b {
		return a
	} else {
		return b
	}
}

// return : (minimun, maximum)
func minMax(a, b uint64) (uint64, uint64) {
	if a < b {
		return a, b
	} else {
		return b, a
	}
}

func uint64ToByteSlice(in uint64) []byte {
	header := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&in)),
		Len:  4,
		Cap:  4,
	}
	return *(*[]byte)(unsafe.Pointer(&header))
}

func relabelSliceToByteSlice(in []relabel) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	const inByte = 16
	header.Len = len(in) * inByte
	header.Cap = len(in) * inByte

	return *(*[]byte)(unsafe.Pointer(&header))
}

func relabelSliceToUint64Slice(in []relabel) []uint64 {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 2
	header.Cap = len(in) * 2

	return *(*[]uint64)(unsafe.Pointer(&header))
}

func uint64SliceToByteSlice(in []uint64) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	const inByte = 8
	header.Len = len(in) * inByte
	header.Cap = len(in) * inByte

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

func mapperA(adj6 []uint8, wg *sync.WaitGroup, in <-chan sRawDat) {
	wg.Add(1)
	go func() {
		defer wg.Done()

		myMinVID, myMaxVID := ^uint64(0), uint64(0)

		for dat := range in {
			//myMap := make(map[uint64]uint64)

			for i := uint64(0); i < dat.cnt; i++ {
				s := dat.src
				now := dat.dstStart + i*wordByte
				d := convBE6toLE8(adj6[now : now+wordByte])

				// update min/max vertex id
				myMinVID = min(myMinVID, min(s, d))
				myMaxVID = max(myMaxVID, max(s, d))
			}
		}

		// apply min/max vertex id to global memory
		mtxMinMaxVID.Lock()
		minVID = min(minVID, myMinVID)
		maxVID = max(maxVID, myMaxVID)
		mtxMinMaxVID.Unlock()
	}()
}

func mapperB(adj6 []uint8, globalSlice []relabel, wg *sync.WaitGroup, in <-chan sRawDat) {
	wg.Add(1)
	go func() {
		defer wg.Done()

		for dat := range in {
			s := dat.src
			atomic.AddUint64(&(globalSlice[s].val), dat.cnt)

			for i := uint64(0); i < dat.cnt; i++ {
				now := dat.dstStart + i*wordByte
				d := convBE6toLE8(adj6[now : now+wordByte])

				if s != d {
					atomic.AddUint64(&(globalSlice[d].val), 1)
				}
			}
		}
	}()
}

/*
func mapperB(adj6 []uint8, globalSlice []relabel, wg *sync.WaitGroup, in <-chan sRawDat) {
	wg.Add(1)
	go func() {
		defer wg.Done()

		myMap := make(map[uint64]uint64)

		for dat := range in {
			s := dat.src
			if _, ok := myMap[s]; !ok {
				myMap[s] = 0
			}
			myMap[s] += dat.cnt

			for i := uint64(0); i < dat.cnt; i++ {
				now := dat.dstStart + i*wordByte
				d := convBE6toLE8(adj6[now : now+wordByte])

				//log.Println(s, d)

				// exclude self-loop
				if s != d {
					if _, ok := myMap[d]; !ok {
						myMap[d] = 0
					}
					myMap[d]++
				}
			}
		}

		// apply min/max vertex id to global memory
		for k, v := range myMap {
			atomic.AddUint64(&(globalSlice[k].val), v)
		}
	}()
}
*/

func subroutineA() {
	files := walk()

	workers := 8
	mappers := 64

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(in <-chan string) {
			defer wg.Done()

			for file := range in {
				data := loader(file)
				splitPos := splitter(data)
				var wgMapper sync.WaitGroup
				for i := 0; i < mappers; i++ {
					mapperA(data, &wgMapper, splitPos)
				}
				wgMapper.Wait()

			}
		}(files)
	}

	wg.Wait()
}

func subroutineB(rSlice *[]relabel) {
	files := walk()

	workers := 8
	shufflers := 64

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(in <-chan string) {
			defer wg.Done()

			for file := range in {
				data := loader(file)
				splitPos := splitter(data)
				var wgMapper sync.WaitGroup
				for i := 0; i < shufflers; i++ {
					mapperB(data, *rSlice, &wgMapper, splitPos)
				}
				wgMapper.Wait()
			}
		}(files)
	}

	wg.Wait()
}

func subroutineC(rSlice *[]relabel) {
	sorty.Mxg = uint32(runtime.NumCPU())

	var comp func(i, j int) bool

	switch ctx.Value("relabelType").(int) {
	case 1:
		comp = func(i, j int) bool {
			if (*rSlice)[i].val == 0 && (*rSlice)[j].val != 0 {
				return false
			} else if (*rSlice)[i].val != 0 && (*rSlice)[j].val == 0 {
				return true
			} else if (*rSlice)[i].val == 0 && (*rSlice)[j].val == 0 {
				return false
			} else {
				if (*rSlice)[i].val == (*rSlice)[j].val {
					return (*rSlice)[i].key < (*rSlice)[j].key
				} else {
					return (*rSlice)[i].val < (*rSlice)[j].val
				}
			}
		}
	case 2:
		comp = func(i, j int) bool {
			if (*rSlice)[i].val == (*rSlice)[j].val {
				return (*rSlice)[i].key > (*rSlice)[j].key
			} else {
				return (*rSlice)[i].val > (*rSlice)[j].val
			}
		}
	default:
		log.Fatalln("relabel Type is wrong")
	}

	sorty.Sort(len((*rSlice)), func(i, k, r, s int) bool {
		if comp(i, k) {
			if r != s {
				(*rSlice)[r], (*rSlice)[s] = (*rSlice)[s], (*rSlice)[r]
			}
			return true
		}
		return false
	})

	workers := runtime.NumCPU()

	var wg2 sync.WaitGroup
	wg2.Add(workers)

	for i := 0; i < workers; i++ {
		go func(idx int) {
			defer wg2.Done()
			for i := idx; i < len(*rSlice); i += workers {
				(*rSlice)[i].val = uint64(i)
			}
		}(i)
	}

	wg2.Wait()

	sorty.Sort(len((*rSlice)), func(i, k, r, s int) bool {
		if (*rSlice)[i].key < (*rSlice)[k].key {
			if r != s {
				(*rSlice)[r], (*rSlice)[s] = (*rSlice)[s], (*rSlice)[r]
			}
			return true
		}
		return false
	})
}

func subroutineD(rSlice *[]relabel) {
	out2 := make([]uint64, len(*rSlice))
	for i, v := range *rSlice {
		out2[i] = v.val
	}

	//inter = inter[:len(inter)/2]

	out := uint64SliceToByteSlice(out2)

	outFilePath := ctx.Value("outFile").(string)
	outFileFolder, _ := filepath.Split(outFilePath)
	os.MkdirAll(outFileFolder, 0755)

	file, err := os.OpenFile(outFilePath, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Panicln(err)
	}
	defer file.Close()

	for pos := int64(0); pos < int64(len(out)); {
		b, err := file.WriteAt(out[pos:], pos)
		pos += int64(b)
		if err != nil {
			log.Panicln(err)
		}
	}
}

func routine() {
	stopwatch("subroutineA (find min/max vertex ID)", func() {
		subroutineA()
	})
	log.Println("min vertex ID:", minVID)
	log.Println("max vertex ID:", maxVID)

	var rSlice []relabel

	stopwatch("Allocate memory space for relabeling map", func() {
		rSlice = make([]relabel, maxVID+1)
		for i := range rSlice {
			rSlice[i].key = uint64(i)
			rSlice[i].val = 0
		}
	})

	stopwatch("subroutineB (count for the degree of each vertex)", func() {
		subroutineB(&rSlice)
	})

	stopwatch("subroutineC (make relabeling map)", func() {
		subroutineC(&rSlice)
	})

	/*
		for i, v := range rSlice {
			log.Println(i, "->", v.key, v.val)
		}
	*/

	stopwatch("subroutineD (write relabeling map to file; uint64 little endian)", func() {
		subroutineD(&rSlice)
	})
}

func init() {
	inFolder := flag.String("in.folder", "", "Input Folder")
	outFile := flag.String("out.file", "", "Output File")
	relabelType := flag.Int("relabel.type", -1, "Output File")
	flag.Parse()

	if len(*inFolder) == 0 || len(*outFile) == 0 || *relabelType == -1 {
		flag.Usage()
		os.Exit(1)
	}

	*outFile, _ = filepath.Abs(*outFile)

	ctx = context.WithValue(ctx, "inFolder", *inFolder)
	ctx = context.WithValue(ctx, "outFile", *outFile)
	ctx = context.WithValue(ctx, "relabelType", *relabelType)
}

func stopwatch(msg string, f func()) {
	log.Println("[Start        ]", msg)
	start := time.Now()
	f()
	elapsed := time.Since(start).Seconds()
	log.Printf("[End %.6fs] %v\n", elapsed, msg)
}

func main() {
	stopwatch("Phase 0 (Reorder Table)", func() {
		routine()
	})
}
