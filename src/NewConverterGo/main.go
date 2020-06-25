package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/exec"
	"time"
)

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
	/*
		var inputInfo information
		{
			log.Println("Phase 0 (Adj6 Scan)", "Start")
			start := time.Now()
			inputInfo = phase0()
			elapsed := time.Since(start).Seconds()
			log.Println("Phase 0 (Adj6 Scan)", "Complete, Elapsed Time:", elapsed, "(sec)")
		}

		log.Println("Phase 0 Result")
		log.Println("    MinVID          :", inputInfo.minvid)
		log.Println("    MaxVID          :", inputInfo.maxvid)
		log.Println("    MaxVID - MinVID :", inputInfo.maxvid-inputInfo.minvid)
		log.Println("    Edges           :", inputInfo.edges)
		log.Println("    Selfloops       :", inputInfo.selfloops)
		log.Println("    Edges-Selfloops :", inputInfo.edges-inputInfo.selfloops)
	*/

	{
		log.Println("Phase 1 (Adj6->Edgelist) Start")
		start := time.Now()
		phase1()
		//phase1(inputInfo)
		elapsed := time.Since(start).Seconds()
		log.Println("Phase 1 (Adj6->Edgelist) Complete, Elapsed Time:", elapsed, "(sec)")
	}

	{
		log.Println("Phase 2 (Edgelist->CSR)", "Start")
		exec.Command("./NewConverterCPP", "inFolder")
	}
}
