package main

/*
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

func mapper0(adj6 []uint8, in <-chan sRawDat) <-chan information {
	out := make(chan information, 1)

	go func() {
		defer close(out)

		myInfo := information{
			selfloops: 0,
			edges:     0,
			minvid:    ^uint64(0),
			maxvid:    0,
		}

		for dat := range in {
			myInfo.edges += dat.cnt
			for i := uint64(0); i < dat.cnt; i++ {
				s := dat.src
				now := dat.dstStart + i*wordByte
				d := convBE6toLE8(adj6[now : now+wordByte])

				//log.Println(s, d)

				myInfo.minvid = min(min(s, d), myInfo.minvid)
				myInfo.maxvid = max(max(s, d), myInfo.maxvid)

				if s == d {
					myInfo.selfloops++
				}
			}
		}

		out <- myInfo
	}()

	return out
}

func infomerge(in ...<-chan information) <-chan information {
	out := make(chan information, 16)

	var wg sync.WaitGroup
	wg.Add(len(in))

	for _, c := range in {
		go func(a <-chan information) {
			defer wg.Done()
			for v := range a {
				out <- v
			}
		}(c)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func reducer(in <-chan information) <-chan information {
	out := make(chan information, 1)
	go func() {
		defer close(out)

		myInfo := information{
			minvid:    ^uint64(0),
			maxvid:    0,
			edges:     0,
			selfloops: 0,
		}

		for v := range in {
			myInfo = information{
				minvid:    min(myInfo.minvid, v.minvid),
				maxvid:    max(myInfo.maxvid, v.maxvid),
				edges:     myInfo.edges + v.edges,
				selfloops: myInfo.selfloops + v.selfloops,
			}
		}

		out <- myInfo
	}()
	return out
}

func routine0(in <-chan string) <-chan information {
	out := make(chan information, 16)
	go func() {
		defer close(out)

		for file := range in {
			data := loader(file)
			splitPos := splitter(data)

			workers := 192

			mapped := make([]<-chan information, workers)
			for i := range mapped {
				mapped[i] = mapper0(data, splitPos)
			}

			out <- <-reducer(infomerge(mapped...))
			//log.Println("Phase 0 (Scan)", file, "Done")
		}
	}()
	return out
}

func phase0() information {
	files := walk()

	result := make([]<-chan information, 32)

	for i := range result {
		result[i] = routine0(files)
	}

	return <-reducer(infomerge(result...))
}
*/
