package main

import (
	"bufio"
	"fmt"
	"bytes"
	"os"
	"io"
)

func ADJ6toCSR4(inpaths []string, outpath string) error {
	fileread := func() error {
		var f *os.File
		var part []byte;
		var count int;
		var e error
		for _, p := range inpaths {
			f, e = os.Open(p)
			if e != nil {
				return e;
			}
			defer f.Close();

			reader := bufio.NewReader(f);
			buffer := bytes.NewBuffer(make([]byte, 0));
			part = make([]byte, 512)

			for {
				fmt.Printf("read: %v\n", p);
				if count, e = reader.Read(part); e != nil {
					break;
				}
				buffer.Write(part[:count]);
			}
			if e != io.EOF {
				return e;
			}
		}

		return nil;
	}

	return fileread();
}