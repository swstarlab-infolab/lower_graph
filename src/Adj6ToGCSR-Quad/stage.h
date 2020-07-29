#ifndef E50D46DC_7197_4A21_9962_83851F3004D8
#define E50D46DC_7197_4A21_9962_83851F3004D8

#include "type.h"

#include <stdint.h>
sp<std::vector<uint64_t>> stage0(fs::path const & inFolder,
								 fs::path const & outFolder,
								 uint64_t const	  maxVID,
								 uint64_t const	  reorderType);

void stage1(fs::path const &		  inFolder,
			fs::path const &		  outFolder,
			uint32_t const			  gridWidth,
			bool const				  lowerTriangular,
			bool const				  reorder,
			sp<std::vector<uint64_t>> reorderTable);

void stage2(fs::path const & outFolder);
void stage3(fs::path const & outFolder, uint32_t const gridWidth);
void stage4(fs::path const & outFolder);

#endif /* E50D46DC_7197_4A21_9962_83851F3004D8 */
