package main

import (
	"reflect"
	"unsafe"
)

// reorder array
type reorder struct {
	key, val uint64
}

// sRawDat array
type sRawDat struct {
	src, cnt, dstStart uint64
}

type gridEdge struct {
	gidx [2]uint32
	edge [2]uint32
}

// view slice of uint32 size 2 array as byte slice
func u32s_bs(in []uint32) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 4
	header.Cap = len(in) * 4

	return *(*[]byte)(unsafe.Pointer(&header))
}

// view slice of uint32 size 2 array as byte slice
func u32a2s_bs(in [][2]uint32) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 8
	header.Cap = len(in) * 8

	return *(*[]byte)(unsafe.Pointer(&header))
}

func bs_u32a2s(in []byte) [][2]uint32 {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) / 8
	header.Cap = len(in) / 8

	return *(*[][2]uint32)(unsafe.Pointer(&header))
}

// view uint64 as byte slice
func u64_bs(in uint64) []byte {
	header := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&in)),
		Len:  4,
		Cap:  4,
	}
	return *(*[]byte)(unsafe.Pointer(&header))
}

// view uint64 slice as byte slice
func u64s_bs(in []uint64) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 8
	header.Cap = len(in) * 8

	return *(*[]byte)(unsafe.Pointer(&header))
}

// view reorder slice as byte slice
func reorders_bs(in []reorder) []byte {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 16
	header.Cap = len(in) * 16

	return *(*[]byte)(unsafe.Pointer(&header))
}

// view reorder slice as uint64 slice
func reorders_u64s(in []reorder) []uint64 {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	header.Len = len(in) * 2
	header.Cap = len(in) * 2

	return *(*[]uint64)(unsafe.Pointer(&header))
}

func be6_le8(in []uint8) uint64 {
	temp := uint64(in[0])
	for i := 1; i <= 5; i++ {
		temp <<= 8
		temp |= uint64(in[i])
	}
	return temp
}
