./build/Adj6ToCSR-Phase1 -in.folder $1 -out.folder $2 -out.name $3 &&
./build/Adj6ToCSR-Phase2 "$2/$3" && 
./build/Adj6ToCSR-Phase3 "$2/$3"