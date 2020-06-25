package main

import (
	"log"
	"os"
	"path/filepath"
)

func convBE6toLE8(in []uint8) uint64 {
	var temp uint64 = 0
	temp |= uint64(in[0])

	for i := 1; i <= 5; i++ {
		temp <<= 8
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

	_, err = file.Read(out)
	if err != nil {
		log.Panicln(err)
	}

	return out
}

func splitter(adj6 []uint8) <-chan sRawDat {
	out := make(chan sRawDat, 16)
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