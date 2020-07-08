package main

import (
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

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

func minMax(a, b uint64) (uint64, uint64) {
	if a < b {
		return a, b
	} else {
		return b, a
	}
}

func ceil(x, y uint) uint {
	if x > 0 {
		return 1 + ((x - 1) / y)
	} else {
		return 0
	}
}

func stopwatch(message string, function func()) {
	log.Println("[Start        ]", message)
	start := time.Now()
	function()
	elapsed := time.Since(start).Seconds()
	log.Printf("[End %.6fs] %v\n", elapsed, message)
}

func fileLoad(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	file, err := os.OpenFile(path, os.O_RDONLY, 0644)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	out := make([]byte, info.Size())

	for pos := int64(0); pos < info.Size(); {
		b, err := file.ReadAt(out[pos:], pos)
		pos += int64(b)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}

func fileSave(path string, in []byte) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	folder, _ := filepath.Split(absPath)
	if err := os.MkdirAll(folder, 0755); err != nil {
		return err
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	defer file.Close()

	for pos := int64(0); pos < int64(len(in)); {
		b, err := file.WriteAt(in[pos:], pos)
		pos += int64(b)
		if err != nil {
			return nil
		}
	}

	return nil
}

func fileSaveAppend(path string, in []byte) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	folder, _ := filepath.Split(absPath)
	if err := os.MkdirAll(folder, 0755); err != nil {
		return err
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	defer file.Close()

	for pos := int64(0); pos < int64(len(in)); {
		b, err := file.Write(in[pos:])
		pos += int64(b)
		if err != nil {
			return err
		}
	}

	return nil
}

// extension must contain single dot only as its prefix, like ".exe", or ".jpg", but not like ".txt.bak"
func fileList(rootFolder, extension string) <-chan string {
	out := make(chan string, 8)
	go func() {
		defer close(out)

		if err := filepath.Walk(rootFolder, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() || info.Size() == 0 {
				return nil
			}

			absPath, _ := filepath.Abs(path)
			if extension != "" {
				if filepath.Ext(absPath) == extension {
					out <- absPath
				}
			} else {
				out <- absPath
			}

			return nil
		}); err != nil {
			panic(err)
		}
	}()
	return out
}

func parallel_for(workers int, function func(workerIdx int)) {
	var wg sync.WaitGroup

	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(wIdx int) {
			defer wg.Done()
			function(wIdx)
		}(w)
	}
	wg.Wait()
}
