#ifndef E50D46DC_7197_4A21_9962_83851F3004D8
#define E50D46DC_7197_4A21_9962_83851F3004D8

#include "type.h"

#include <stdint.h>
sp<std::vector<uint64_t>> stage0(fs::path const & inFolder,
								 fs::path const & outFolder,
								 uint64_t const	  maxVID,
								 uint64_t const	  relabelType);

void stage1(fs::path const &		  inFolder,
			fs::path const &		  outFolder,
			uint32_t const			  gridWidth,
			bool const				  lowerTriangular,
			bool const				  relabel,
			sp<std::vector<uint64_t>> relabelTable);

void stage2(fs::path const & outFolder);
void stage3(fs::path const & outFolder, uint32_t const gridWidth, size_t const limitByte);
void stage4(fs::path const & outFolder);

#endif /* E50D46DC_7197_4A21_9962_83851F3004D8 */
