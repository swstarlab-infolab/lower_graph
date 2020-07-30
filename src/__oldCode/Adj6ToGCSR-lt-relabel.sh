./build/Adj6ToGCSR-lt-relabel-Phase0 -in.folder $1 -out.file "$2/$3/relabel.bin" -relabel.type $4
./build/Adj6ToGCSR-lt-relabel-Phase1 -in.folder $1 -relabel.file "$2/$3/relabel.bin" -out.folder $2 -out.name $3 &&
rm -rf "$2/$3/relabel.bin" &&
./build/Adj6ToGCSR-lt-relabel-Phase2 "$2/$3" && 
./build/Adj6ToGCSR-lt-relabel-Phase3 "$2/$3"