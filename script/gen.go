package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"strconv"
	"strings"
	"time"

	"github.com/fatih/color"
)

func log(logType string, message string) {
	t := time.Now().Local()
	s := fmt.Sprintf("[%v] %v", t.Format("2006-01-02 15:04:05.000"), message)

	switch logType {
	case "debug":
		color.HiBlue("%v", s)
	case "success":
		color.HiGreen("%v", s)
	case "error":
		color.HiRed("%v", s)
	case "warning":
		color.HiYellow("%v", s)
	case "info":
		fmt.Printf("%v\n", s)
	}
}

func check(e error) {
	if e != nil {
		log("error", e.Error())
		os.Exit(1)
	}
}

type datasetSetting struct {
	inputpath, outputpath    string
	name, format             string
	minGridSize, maxGridSize int
}

func producer(work chan []string, done chan bool, setting []datasetSetting) {
	for _, data := range setting {
		for gridSize := data.minGridSize; gridSize <= data.maxGridSize; gridSize++ {
			filename := strings.Join([]string{data.name, "lt", strconv.Itoa(gridSize)}, "-")
			work <- []string{data.format, data.inputpath + data.name + "/", strconv.Itoa(gridSize), data.outputpath, filename}
		}
	}

	close(work)
	done <- true
}

func consumer(workerID int, bin string, work chan []string, done chan bool) {
	for w := range work {
		cmd := exec.Command(bin, w...)
		stdout, e := cmd.Output()
		basicLogString := "GRIDSIZE=" + strconv.Itoa(workerID) + " " + strings.Join(w, " ") + " : "
		if e != nil {
			log("error", basicLogString+strings.TrimSuffix(e.Error(), "\n"))
		} else {
			log("success", basicLogString+strings.TrimSuffix(string(stdout), "\n"))
		}
	}

	done <- true
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	log("info", "Engaging Workers")

	workerCount := 7

	work := make(chan []string, 9)
	done := make(chan bool, workerCount+1)

	bin := "/home/heeyong/grid-csr/build/converter"

	datasets := []datasetSetting{}

	func() {
		myinfolder := "/mnt/nvme/TSV/"
		myoutfolder := "/mnt/nvme/GCSR/"
		myformat := "tsv"

		datasets := append(datasets, []datasetSetting{
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "wiki-Talk",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 22,
			},
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "cit-Patents",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 23,
			},
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "soc-LiveJournal1",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 23,
			},
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "com-orkut",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 22,
			},
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "twitter_rv.net",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 26,
			},
			{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        "com-friendster",
				format:      myformat,
				minGridSize: 16,
				maxGridSize: 27,
			},
		}...)
	}()

	func() {
		myinfolder := "/mnt/nvme/Adj6/"
		myoutfolder := "/mnt/nvme/GCSR/"
		myformat := "adj6"

		mydatasets := []datasetSetting{}
		for i := 16; i <= 29; i++ {
			RMATnumber := fmt.Sprintf("RMAT%02s", i)
			mydatasets := append(mydatasets, datasetSetting{
				inputpath:   myinfolder,
				outputpath:  myoutfolder,
				name:        RMATnumber,
				format:      myformat,
				minGridSize: 16,
				maxGridSize: i,
			})
		}

		datasets := append(datasets, mydatasets...)
	}()

	go producer(work, done, datasets)

	for i := 0; i < workerCount; i++ {
		go consumer(i, bin, work, done)
	}

	for i := 0; i < workerCount+1; i++ {
		<-done
	}

	log("info", "Finished")
}
