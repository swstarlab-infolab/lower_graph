package main

import (
	"context"
	"fmt"
)

func ctxInt(ctx context.Context, key string) int {
	var a int
	if v := ctx.Value(key); v != nil {
		if u, ok := v.(int); !ok {
			panic(fmt.Errorf("context type casting error: " + key + " is not int"))
		} else {
			a = u
		}
	}

	return a
}

func ctxBool(ctx context.Context, key string) bool {
	var a bool
	if v := ctx.Value(key); v != nil {
		if u, ok := v.(bool); !ok {
			panic(fmt.Errorf("context type casting error: " + key + " is not bool"))
		} else {
			a = u
		}
	}

	return a
}

func ctxString(ctx context.Context, key string) string {
	var a string
	if v := ctx.Value(key); v != nil {
		if u, ok := v.(string); !ok {
			panic(fmt.Errorf("context type casting error: " + key + " is not string"))
		} else {
			a = u
		}
	}

	return a
}
