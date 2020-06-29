static void DataManagerInit(Context & ctx, int myID)
{
	using DataChanType = bchan<Tx>;

	auto & myMem = ctx.dataManagerCtx[myID];

	printf("Start to initialize Device: %d\n", myID);
	if (myID > -1) {
		// GPU Memory
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else if (myID == -1) {
		// CPU Memory
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else {
		// Storage
	}
}
