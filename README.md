# Grid-CSR

Grid = Blocked Matrix

Grid-divided CSR structure

## CSR Types for Grid-CSR

### CSR Type 1
```
ptr: 0 0 1 1 2 4 4 7
col: 0 1 1 2 0 2 4
```

Result

### CSR Type 2
```
row: 1 3 4 6
ptr: 0 0 1 1 2 4 4 7
col: 0 1 1 2 0 2 4
```

### CSR Type 3
```
row: 1 3 4 6
ptr: 0 1 2 4 7
col: 0 1 1 2 0 2 4
```