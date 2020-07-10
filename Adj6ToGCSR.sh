./build/Adj6ToGCSR-lt-Phase1 -in.folder $1 -out.folder $2 -out.name $3 &&
./build/Adj6ToGCSR-lt-Phase2 "$2/$3" && 
./build/Adj6ToGCSR-lt-Phase3 "$2/$3"