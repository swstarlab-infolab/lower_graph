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
	"github.com/jaypipes/ghw"
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
	basepath, name           string
	minGridSize, maxGridSize int
}

func producer(work chan []string, done chan bool, setting []datasetSetting) {
	maxThreads := 160 * 1024

	for _, data := range setting {
		for gridSize := data.maxGridSize; gridSize >= data.minGridSize; gridSize-- {

			filename := strings.Join([]string{data.name, "lt", strconv.Itoa(gridSize)}, "-")

			for s := 1; s <= 2; s++ {
				for t := 1024; t >= 32; t /= 2 {
					b := maxThreads / t
					work <- []string{data.basepath + filename, strconv.Itoa(s), strconv.Itoa(b), strconv.Itoa(t)}
				}
			}
		}
	}

	close(work)
	done <- true
}

func consumer(gpuIndex int, bin string, work chan []string, done chan bool) {
	for w := range work {
		env := "CUDA_VISIBLE_DEVICES=" + strconv.Itoa(gpuIndex)

		cmd := exec.Command(bin, w...)
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, env)
		stdout, e := cmd.Output()

		basicLogString := "GPU=" + strconv.Itoa(gpuIndex) + " " + strings.Join(w, " ") + " : "
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
	pci, e := ghw.PCI()
	check(e)
	devices := pci.ListDevices()
	if len(devices) == 0 {
		log("error", e.Error())
		os.Exit(1)
	}

	nvidiaGPUs := 0
	for _, device := range devices {
		if device.Vendor.Name == "NVIDIA Corporation" && !strings.Contains(device.Product.Name, "Audio") {
			log("info", device.Address+", "+device.Vendor.Name+", "+device.Product.Name)
			nvidiaGPUs++
		}
	}

	log("info", "Total GPUs: "+strconv.Itoa(nvidiaGPUs))
	log("info", "Engaging Workers")

	bin := "/home/heeyong/grid-csr/build/triangle-new"
	basePath := "/mnt/nvme-raid0/GCSR-New/"

	work := make(chan []string, 9)
	done := make(chan bool, nvidiaGPUs+1)

	datasets := []datasetSetting{
		{
			basepath:    basePath,
			name:        "as20000102",
			minGridSize: 10,
			maxGridSize: 16,
		},
	}

	go producer(work, done, datasets)

	for i := 0; i < nvidiaGPUs; i++ {
		go consumer(i, bin, work, done)
	}

	for i := 0; i < nvidiaGPUs+1; i++ {
		<-done
	}

	log("info", "Finished")
}
