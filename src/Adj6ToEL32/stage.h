#ifndef E50D46DC_7197_4A21_9962_83851F3004D8
#define E50D46DC_7197_4A21_9962_83851F3004D8

#include "type.h"

#include <stdint.h>

void stage1(fs::path const & inFolder,
			fs::path const & outFolder,
			uint32_t const	 gridWidth,
			bool const		 lowerTriangular);

void stage2(fs::path const & inFolder, fs::path const & outFolder);

#endif /* E50D46DC_7197_4A21_9962_83851F3004D8 */
