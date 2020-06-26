package main

import "context"

var (
	ctx = context.Background()
)

const (
	wordByte = 6
	gWidth   = 1 << 24
)

type vertex32 uint32
type edge32 [2]vertex32
type gidx32 [2]uint32

type sRawDat struct {
	src, cnt, dstStart uint64
}

type gridEdge struct {
	gidx gidx32
	edge edge32
}
