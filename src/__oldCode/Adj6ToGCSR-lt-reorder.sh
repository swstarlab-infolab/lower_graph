./build/Adj6ToGCSR-lt-reorder-Phase0 -in.folder $1 -out.file "$2/$3/reorder.bin" -reorder.type $4
./build/Adj6ToGCSR-lt-reorder-Phase1 -in.folder $1 -reorder.file "$2/$3/reorder.bin" -out.folder $2 -out.name $3 &&
rm -rf "$2/$3/reorder.bin" &&
./build/Adj6ToGCSR-lt-reorder-Phase2 "$2/$3" && 
./build/Adj6ToGCSR-lt-reorder-Phase3 "$2/$3"