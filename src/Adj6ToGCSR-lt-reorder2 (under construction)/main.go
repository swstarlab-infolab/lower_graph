package main

import (
	"flag"
	"os"
)

func main() {
	inFolder := flag.String("inFolder", "", "Input Folder")
	outFolder := flag.String("outFolder", "", "Output Folder")
	outName := flag.String("outName", "", "Output Name")
	gWidth := flag.Uint64("gridWidth", uint64(1<<24), "Grid Width")
	reorderType := flag.Int("reorderType", -1, "Reorder Type")

	flag.Parse()

	if *inFolder == "" || *outFolder == "" || *outName == "" {
		flag.Usage()
		os.Exit(1)
	}

	switch *reorderType {
	case 1:
		fallthrough
	case 2:
		var rSlice []uint64
		stopwatch("STAGE0 Reorder", func() {
			rSlice = stage0(*inFolder, *reorderType)
		})

		stopwatch("STAGE1 Adj6→EL32", func() {
			stage1_reorder(*inFolder, *outFolder, *outName, *gWidth, rSlice)
		})
	default:
		stopwatch("STAGE1 Adj6→EL32", func() {
			stage1(*inFolder, *outFolder, *outName, *gWidth)
		})
	}

	stopwatch("STAGE2 EL32→GCSR", func() {
		stage2(*outFolder, *outName)
	})
}
