#include <BuddySystem/BuddySystem.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#define _mixx_unused(x) ((void)(x))

namespace _buddy_impl
{

struct buddy_table_size_info {
	level_t tbl_size;
	level_t buddy_lv;
};

/*
 * Root: 200
 * Minimum coefficient: 3
 *
 * Coefficient tree
 *              +---          [232]             ---> 232 [Root]
 *              |               |
 *              |             [116]             ---> 116
 *   Linear  <--+               |
 *              |             [58]              ---> 58
 *              |               |
 *              +---          [29]              ---> 29
 *              |            /    \
 *              |         [15]     [14]         ---> 15, 14
 *   Binary  <--+        /  \       /  \
 *              |      [8]  [7]    [7]  [7]     ---> 8, 7 [A1B3-Pattern]
 *              |      / \  / \    / \   / \
 *              +--- [4][4][4][3] [4][3][4][3]  ---> 4, 3 [A3B1-Pattern]
 *
 * Buddy table
 * - Flag: Unique (U), Frequent (F), Rare (R), A1B3-Pattern, A3B1-Pattern
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * | Idx | Lev |  Cof  | U | F | R | Pattern | Dist | Off |
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * |  0  |  0  |  232  | # |   |   |   N/A   |   0  |  0  |
 * |  1  |  1  |  116  | # |   |   |   N/A   |   1  |  0  |
 * |  2  |  2  |   58  | # |   |   |   N/A   |   1  |  0  |
 * |  3  |  3  |   29  | # |   |   |   N/A   |   1  |  0  |
 * |  4  |  4  |   15  |   |   | # |   A3B1* |   1  |  0  |
 * |  5  |  4  |   14  |   |   | # |   A3B1* |   2  |  1  |
 * |  6  |  5  |    8  |   |   | # |   A1B3  |   2  |  0  |
 * |  7  |  5  |    7  |   | # |   |   A1B3  |   3  |  1  |
 * |  8  |  6  |    4  |   | # |   |   A3B1  |   2  |  0  |
 * |  9  |  6  |    3  |   |   | # |   A3B1  |   3  |  1  |
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * Note that the patterns of a first binary level (4) are assumed to be R-A3B1*
 *
 */

inline unsigned log2u(unsigned v)
{
	// source: https://graphics.stanford.edu/~seander/bithacks.html
	static unsigned const b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	static unsigned const s[] = {1, 2, 4, 8, 16};
	unsigned int		  r	  = 0;
	for (auto i = 4; i >= 0; i--) {
		if (v & b[i]) {
			v >>= s[i];
			r |= s[i];
		}
	}
	return r;
}

inline uint64_t log2u(uint64_t v)
{
	static uint64_t const b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000, 0xFFFFFFFF00000000};
	static uint64_t const s[] = {1, 2, 4, 8, 16, 32};
	uint64_t			  r	  = 0;
	for (auto i = 5; i >= 0; i--) {
		if (v & b[i]) {
			v >>= s[i];
			r |= s[i];
		}
	}
	return r;
}

template <typename T>
T * seek_pointer(T * p, int64_t offset)
{
	auto const p2 = reinterpret_cast<char *>(p);
	return reinterpret_cast<T *>(p2 + offset);
}

buddy_table_size_info get_buddy_table_size_info(cof_type const root, cof_type const min_cof)
{
	// Note that a size of root node is root * alignment
	auto const get_linear_bound = [](cof_type n, cof_type min_cof) -> cof_type {
		do {
			if (n & 0x1u || n <= min_cof)
				return n;
			n /= 2;
		} while (true);
	};

	auto const get_binary_depth = [](cof_type odd, cof_type min_cof) -> unsigned {
		assert(odd & 0x1u);
		unsigned depth = 0;
		do {
			cof_type const q = odd / 2; // quotient
			// select a next odd number
			q & 0x1u ? odd = q : odd = q + 1;
			if (q < min_cof)
				break;
			depth += 1;
		} while (true);
		return depth;
	};

	buddy_table_size_info info;
	cof_type const		  linear_bound = get_linear_bound(root, min_cof);
	unsigned const		  linear_depth =
		static_cast<unsigned>(log2u(static_cast<uint64_t>(root / linear_bound))) + 1;
	if (linear_bound <= min_cof) {
		info.tbl_size = info.buddy_lv = linear_depth;
	} else {
		unsigned const binary_depth = get_binary_depth(linear_bound, min_cof);
		info.tbl_size				= linear_depth + binary_depth * 2;
		info.buddy_lv				= linear_depth + binary_depth;
	}
	return info;
}

bool is_aligned_address(void * addr, uint64_t alignment)
{
	auto const addr2 = reinterpret_cast<uint64_t>(addr);
	return addr2 % alignment == 0;
}

buddy_table::buddy_table()
{
	_align	  = 0;
	_min_cof  = 0;
	_tbl_size = 0;
	_buddy_lv = 0;
	memset(&_attr, 0, sizeof _attr);
}

buddy_table::~buddy_table() noexcept { _free_attr(&_attr); }

buddy_table::buddy_table(cof_type const root_cof, unsigned const align, cof_type const min_cof)
	: buddy_table()
{
	init(root_cof, align, min_cof);
}

void buddy_table::init(cof_type const root_cof, unsigned const align, cof_type const min_cof)
{
	assert(align % 2 == 0);
	assert(min_cof > 0);
	auto const info = get_buddy_table_size_info(root_cof, min_cof);
	_align			= align;
	_min_cof		= min_cof;
	_tbl_size		= info.tbl_size;
	_buddy_lv		= info.buddy_lv;
	_attr			= _init_attr(_tbl_size, root_cof, align, min_cof);
	// TODO: Add exception handling when initialization fails (11/05/2019)
}

void buddy_table::clear() noexcept
{
	_free_attr(&_attr);
	_align	  = 0;
	_min_cof  = 0;
	_tbl_size = 0;
	_buddy_lv = 0;
	memset(&_attr, 0, sizeof _attr);
}

void buddy_table::printout() const
{
	for (level_t i = 0; i < _tbl_size; ++i) {
		printf("%4u | %4u | %" PRId64 "\n", i, _attr.level_v[i], _attr.cof_v[i]);
	}
}

blkidx_t buddy_table::best_fit(uint64_t const block_size) const
{
	assert(block_size <= static_cast<uint64_t>(_attr.cof_v[0] * _align));
	assert(_tbl_size > 0);
	cof_type const cof = static_cast<cof_type>((block_size + _align - 1) / _align);

	for (unsigned i = _tbl_size - 1; i > 0; --i) {
		if (_attr.cof_v[i] >= cof) {
			assert(block_size <= static_cast<uint64_t>(_attr.cof_v[i] * _align));
			return i;
		}
	}
	return 0;

	/*cof_type diff1 = std::numeric_limits<cof_type>::max();
	blkidx_t i = _tbl_size;
	do {
		cof_type const diff2 = std::abs(_attr.cof_v[i - 1] - cof);
		if (diff2 > diff1)
			return i;
		diff1 = diff2;
		i -= 1;
	} while (i > 0);
	return i;*/
}

void buddy_table::_free_attr(tbl_attr * attr)
{
	free(attr->level_v);
	free(attr->cof_v);
	free(attr->prof_v);
	memset(attr, 0, sizeof(tbl_attr));
}

buddy_table::tbl_attr buddy_table::_init_attr(unsigned const tbl_size,
											  cof_type const root,
											  unsigned const align,
											  cof_type const min_cof)
{
	assert(align % 2 == 0);
	assert(min_cof > 0);
	assert(root > 0);

	cof_type n = root;
	tbl_attr attr;
	memset(&attr, 0, sizeof attr);
	attr.level_v = static_cast<level_t *>(calloc(tbl_size, sizeof(level_t)));
	if (attr.level_v == nullptr)
		goto lb_err;
	attr.cof_v = static_cast<cof_type *>(calloc(tbl_size, sizeof(cof_type)));
	if (attr.cof_v == nullptr)
		goto lb_err;
	attr.prof_v = static_cast<blk_prop_t *>(calloc(tbl_size, sizeof(uint8_t)));
	if (attr.prof_v == nullptr)
		goto lb_err;

	// root
	attr.level_v[0] = 0;
	attr.cof_v[0]	= n;
	attr.prof_v[0].flags |= UniqueBuddyBlock;
	attr.prof_v[0].dist	  = 0;
	attr.prof_v[0].offset = 0;

	do {
		// linear
		blkidx_t i = 1;
		while (i < tbl_size && !(n & 0x1u)) {
			attr.level_v[i] = i;
			attr.cof_v[i]	= n / 2;
			attr.prof_v[i].flags |= UniqueBuddyBlock;
			attr.prof_v[i].dist	  = 1;
			attr.prof_v[i].offset = 0;
			n /= 2;
			i += 1;
		}

		if (i >= tbl_size)
			break;

		assert(((tbl_size - i) & 0x1u) == 0);
		// binary
		level_t		   lv			= i;
		blkidx_t const linear_size	= i;
		bool		   a1b3_pattern = false;
		while (i + 1u < tbl_size) {
			assert(n & 0x1u);
			cof_type const r = n / 2;
			cof_type const l = r + 1;
			// left-side child
			attr.level_v[i] = lv;
			attr.cof_v[i]	= l;
			// right-side child
			attr.level_v[i + 1] = lv;
			attr.cof_v[i + 1]	= r;
			// spanning info
			if (a1b3_pattern) {
				attr.prof_v[i].flags	 = RareBuddyBlock | A1B3Pattern;
				attr.prof_v[i + 1].flags = FrequentBuddyBlock | A1B3Pattern;
			} else {
				attr.prof_v[i].flags	 = FrequentBuddyBlock | A3B1Pattern;
				attr.prof_v[i + 1].flags = RareBuddyBlock | A3B1Pattern;
			}
			attr.prof_v[i].dist		  = 2;
			attr.prof_v[i].offset	  = 0;
			attr.prof_v[i + 1].dist	  = 3;
			attr.prof_v[i + 1].offset = 1;
			// update states for next iteration
			a1b3_pattern	 = l & 0x1u;
			a1b3_pattern ? n = l : n = r;
			i += 2;
			lv += 1;
		}
		// fix for a first binary level
		assert(linear_size + 1 < tbl_size);
		attr.prof_v[linear_size].dist	  = 1;
		attr.prof_v[linear_size + 1].dist = 2;
		attr.prof_v[linear_size].flags	  = attr.prof_v[linear_size + 1].flags =
			RareBuddyBlock | A3B1Pattern;
	} while (false);
	goto lb_return;

lb_err:
	_free_attr(&attr);

lb_return:
	return attr;
}

segregated_storage::segregated_storage(void * preallocated, size_t _bufsize, size_t _block_size)
	: buffer(preallocated), bufsize(_bufsize), block_size(_block_size),
	  capacity(bufsize / block_size), _free_list(capacity)
{
	reset();
}

void * segregated_storage::allocate()
{
	void * p;
	if (_free_list.pop(&p)) {
		assert(static_cast<char *>(p) >= static_cast<char *>(buffer) &&
			   static_cast<char *>(p) < static_cast<char *>(buffer) + bufsize);
		return p;
	}
	return nullptr;
}

void segregated_storage::deallocate(void * p)
{
	assert(static_cast<char *>(p) >= static_cast<char *>(buffer) &&
		   static_cast<char *>(p) < static_cast<char *>(buffer) + bufsize);
	_free_list.push(p);
}

void segregated_storage::reset()
{
	void ** p = _free_list._get_buffer();
	for (size_t i = 0; i < capacity; ++i)
		p[i] = seek_pointer(buffer, block_size * i);
	_free_list._config(capacity);
}

double segregated_storage::fill_rate() const
{
	return static_cast<double>(_free_list.size()) / static_cast<double>(_free_list.capacity());
}

template <typename RequestType, typename OffsetType>
inline RequestType get_bit(OffsetType off)
{
	RequestType c = 1;
	return c << off;
}

template <typename RequestType, typename OffsetType>
inline void set_bit(RequestType * bitmap, OffsetType off)
{
	bitmap[off / (sizeof(RequestType) * 8)] |=
		get_bit<RequestType>(off % (sizeof(RequestType) * 8));
}

template <typename RequestType, typename OffsetType>
inline void clear_bit(RequestType * bitmap, OffsetType off)
{
	bitmap[off / (sizeof(RequestType) * 8)] &=
		~get_bit<RequestType>(off % (sizeof(RequestType) * 8));
}

template <typename RequestType, typename OffsetType>
inline bool test_bit(RequestType * bitmap, OffsetType off)
{
	return bitmap[off / (sizeof(RequestType) * 8)] &
		   get_bit<RequestType>(off % (sizeof(RequestType) * 8));
}

static char const __num_to_bits[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

inline uint8_t count_set_bits(uint8_t const n)
{
	if (n == 0)
		return 0;
	uint8_t const nibble = n & 0xf;
	return __num_to_bits[nibble] + __num_to_bits[n >> 4];
}

inline uint16_t count_set_bits(uint16_t const n)
{
	if (n == 0)
		return 0;
	uint16_t const c = count_set_bits(static_cast<uint8_t>(n));
	return c + count_set_bits(static_cast<uint8_t>(n >> 8));
}

inline uint32_t count_set_bits(uint32_t const n)
{
	if (n == 0)
		return 0;
	uint32_t const c = count_set_bits(static_cast<uint16_t>(n));
	return c + count_set_bits(static_cast<uint16_t>(n >> 16));
}

inline uint64_t count_set_bits(uint64_t const n)
{
	if (n == 0)
		return 0;
	uint64_t const c = count_set_bits(static_cast<uint32_t>(n));
	return c + count_set_bits(static_cast<uint32_t>(n >> 32));
}

template <size_t Length>
class bitset
{
public:
	constexpr static size_t length		= Length;
	constexpr static size_t byte_length = length / 8;
	static_assert(length % 8 == 0, "a template argument \'Length\' must be multiple of 8");

	bitset() { memset(data, 0, byte_length); }

	bitset(bitset const & other) { memmove(data, other.data, byte_length); }

	~bitset() noexcept
	{ /* nothing to do. */
	}

	inline bool test(size_t const index) const
	{
		const char byte = data[index / 8];
		return byte & get_bit<char>(index % 8);
	}

	inline void set(size_t const index) { data[index / 8] |= get_bit<char>(index % 8); }

	inline void clear(size_t const index) { data[index / 8] &= ~get_bit<char>(index % 8); }

	inline void set_all() { memset(data, ~0, byte_length); }

	inline void clear_all() { memset(data, 0, byte_length); }

	inline size_t count() const
	{
		size_t			l = byte_length / 8;
		size_t			c = 0;
		uint8_t const * p = reinterpret_cast<uint8_t const *>(data);
		for (size_t i = 0; i < l; ++i) {
			c += count_set_bits(*reinterpret_cast<uint64_t const *>(p));
			p += 8;
		}
		l = byte_length - (l * 8);
		for (size_t i = 0; i < l; ++i) {
			c += count_set_bits(*p);
			p += 1;
		}
		return c;
	}

	inline bool operator==(bitset const & other)
	{
		return memcmp(data, other.data, byte_length) == 0;
	}

	bitset & operator|=(bitset const & other)
	{
		size_t		 l	= byte_length / 8;
		char *		 p1 = data;
		char const * p2 = other.data;
		for (size_t i = 0; i < l; ++i) {
			*reinterpret_cast<int64_t *>(p1) |= *reinterpret_cast<int64_t const *>(p2);
			p1 += 8;
			p2 += 8;
		}
		l = byte_length - (l * 8);
		for (size_t i = 0; i < l; ++i) {
			*p1 |= *p2;
			p1 += 1;
			p2 += 1;
		}
		return *this;
	}

	bitset & operator&=(bitset const & other)
	{
		size_t		 l	= byte_length / 8;
		char *		 p1 = data;
		char const * p2 = other.data;
		for (size_t i = 0; i < l; ++i) {
			*reinterpret_cast<int64_t *>(p1) &= *reinterpret_cast<int64_t const *>(p2);
			p1 += 8;
			p2 += 8;
		}
		l = byte_length - (l * 8);
		for (size_t i = 0; i < l; ++i) {
			*p1 &= *p2;
			p1 += 1;
			p2 += 1;
		}
		return *this;
	}

	bitset & operator^=(bitset const & other)
	{
		size_t		 l	= byte_length / 8;
		char *		 p1 = data;
		char const * p2 = other.data;
		for (size_t i = 0; i < l; ++i) {
			*reinterpret_cast<int64_t *>(p1) ^= *reinterpret_cast<int64_t const *>(p2);
			p1 += 8;
			p2 += 8;
		}
		l = byte_length - (l * 8);
		for (size_t i = 0; i < l; ++i) {
			*p1 ^= *p2;
			p1 += 1;
			p2 += 1;
		}
		return *this;
	}

	bitset operator|(bitset const & rhs) const
	{
		bitset r{*this};
		return r |= rhs;
	}

	bitset operator&(bitset const & rhs) const
	{
		bitset r{*this};
		return r &= rhs;
	}

	bitset operator^(bitset const & rhs) const
	{
		bitset r{*this};
		return r ^= rhs;
	}

	char data[byte_length];
};

//! Byte swap unsigned short
inline uint16_t swap_uint16(uint16_t val) { return (val << 8) | (val >> 8); }

//! Byte swap short
inline int16_t swap_int16(int16_t val) { return (val << 8) | ((val >> 8) & 0xFF); }

//! Byte swap unsigned int
inline uint32_t swap_uint32(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

//! Byte swap int
inline int32_t swap_int32(int32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | ((val >> 16) & 0xFFFF);
}

//! Byte swap 64-bit int
inline int64_t swap_int64(int64_t val)
{
	val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
	val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
	return (val << 32) | ((val >> 32) & 0xFFFFFFFFULL);
}

//! Byte swap 64-bit unsigned int
inline uint64_t swap_uint64(uint64_t val)
{
	val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
	val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
	return (val << 32) | (val >> 32);
}

bit_stack::bit_stack(size_t capacity_)
{
#ifdef _MSC_VER
	_cap = roundup(capacity_, 4llu);
#else
	_cap = roundup(capacity_, 4lu);
#endif
	_top	   = 0;
	_container = static_cast<int *>(malloc(_cap / sizeof(int)));
	if (_container != nullptr)
		memset(_container, 0, _cap / sizeof(int));
}

bit_stack::bit_stack(bit_stack const & other)
{
	_cap	   = other._cap;
	_top	   = other._top;
	_container = static_cast<int *>(malloc(_cap / sizeof(int)));
	if (_container != nullptr)
		memcpy(_container, other._container, _cap / sizeof(int));
}

bit_stack::bit_stack(bit_stack && other) noexcept
{
	_cap			 = other._cap;
	_top			 = other._top;
	_container		 = other._container;
	other._container = nullptr;
}

bit_stack::~bit_stack() noexcept { free(_container); }

bit_stack & bit_stack::operator=(bit_stack const & rhs)
{
	_cap	   = rhs._cap;
	_top	   = rhs._top;
	_container = static_cast<int *>(malloc(_cap / sizeof(int)));
	if (_container != nullptr)
		memcpy(_container, rhs._container, _cap / sizeof(int));
	return *this;
}

bit_stack & bit_stack::operator=(bit_stack && rhs) noexcept
{
	_cap		   = rhs._cap;
	_top		   = rhs._top;
	_container	   = rhs._container;
	rhs._container = nullptr;
	return *this;
}

bool bit_stack::push(bool const value)
{
	if (_top == _cap) {
		if (!reserve(_cap * 2))
			return false;
		_top += 1;
	}
	if (value)
		set_bit(_container, _top);
	else
		clear_bit(_container, _top);
	_top += 1;
	return true;
}

bool bit_stack::pop()
{
	if (empty())
		return false;
	_top -= 1;
	return true;
}

bool bit_stack::peek() const
{
	assert(_top > 0);
	return test_bit(_container, _top - 1);
}

void bit_stack::clear()
{
	_top = 0;
	if (_container != nullptr)
		memset(_container, 0, _cap / sizeof(int));
}

bool bit_stack::reserve(size_t const new_capacity)
{
	if (_cap >= new_capacity)
		return true;
	assert(_cap % 4 == 0);
	assert(_container != nullptr);
	int * new_container = static_cast<int *>(malloc(new_capacity / sizeof(int)));
	if (new_container == nullptr)
		return false;
	memcpy(new_container, _container, size() / sizeof(int));
	free(_container);
	_container = new_container;
	return true;
}

buddy_system::buddy_system()
{
	memset(&_rgn, 0, sizeof _rgn);
	_align		  = 0;
	_max_blk_size = 0;
	_flist_v	  = nullptr;
	memset(&_status, 0, sizeof(buddy_system_status));
}

buddy_system::~buddy_system() { _cleanup(); }

buddy_system::buddy_system(memrgn_t const & rgn, unsigned const align, unsigned const min_cof)
	: buddy_system()
{
	init(rgn, align, min_cof);
}

void buddy_system::init(memrgn_t const & rgn, unsigned const align, unsigned const min_cof)
{
	// TODO: badalloc handling
	assert(is_aligned_address(_rgn.ptr, align));
	cof_type const root_cof = static_cast<cof_type>(rgn.size / align);
	_rgn					= rgn;
	_align					= align;
	_max_blk_size			= root_cof * align;
	_tbl.init(root_cof, align, min_cof);
	auto block	  = _block_pool.allocate();
	block->cof	  = root_cof;
	block->blkidx = 0;
	block->rgn	  = _rgn;
	block->pair	  = nullptr;
	block->parent = nullptr;
	block->in_use = false;
	_flist_v	  = _init_free_list_vec(_tbl.size(), _node_pool);
	_flist_v[0].emplace_front(block);
	_route.reserve(_tbl.max_level());
	_total_allocated_size = 0;
	fprintf(stdout, "Buddy system is online. [%p, %" PRIu64 "]\n", rgn.ptr, rgn.size);
}

void * buddy_system::allocate(uint64_t size)
{
	size += sizeof(buddy_block **);
	buddy_block * block = allocate_block(size);
	if (block == nullptr)
		return nullptr;

	auto const p = static_cast<buddy_block **>(block->rgn.ptr);
	*p			 = block;
	return p + 1;
}

void buddy_system::deallocate(void * p)
{
	if (p == nullptr)
		return;
	deallocate_block(*(static_cast<buddy_block **>(p) - 1));
}

buddy_block * buddy_system::allocate_block(uint64_t const size)
{
	if ((size > _max_blk_size))
		return nullptr;

	assert(_route.empty());
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	assert(_route_dbg.empty());
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	blkidx_t   bf	  = _tbl.best_fit(size);
	auto const result = _create_route(bf);
	if (!result.success) {
		_route.clear();
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		_route_dbg.clear();
#endif					// !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		return nullptr; // bad alloc
	}

	assert(!_route.empty());
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	assert(_route.size() == _route_dbg.size());
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	assert(!_flist_v[result.blkidx].empty());
	auto		  begin = _flist_v[result.blkidx].begin();
	buddy_block * block = *begin;
	_flist_v[result.blkidx].remove_node(begin);

	_route.pop();
	// blkidx_t idx_dbg = result.blkidx;
	buddy_block * child[2];
	while (!_route.empty()) {
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		assert(idx_dbg == _route_dbg.peek());
		_route_dbg.pop();
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		child[0] = _block_pool.allocate();
		child[1] = _block_pool.allocate();
		_split_block(block, child[0], child[1], _tbl);
		block->in_use		  = true;
		buddy_block *& target = child[_route.peek()];
		buddy_block *& spare  = child[!_route.peek()];
		assert(_flist_v[target->blkidx].empty());
		// idx_dbg = target->blkidx;
		spare->inv = _flist_v[spare->blkidx].emplace_front(spare).node();

		// update states
		block = target;
		_route.pop();
	}
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	_route_dbg.pop();
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING

	assert(block->in_use == false);

	block->in_use = true;
	block->inv	  = nullptr;
	_status.total_allocated += 1;

	assert(_route.empty());
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	assert(_route_dbg.empty());
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING

	assert(size <= (uint64_t)(block->cof * _align));
	_total_allocated_size += block->cof * _align;
	return block;
}

void buddy_system::deallocate_block(buddy_block * blk)
{
	_total_allocated_size -= blk->cof * _align;
	_deallocate(blk);
}

free_list_t * buddy_system::_init_free_list_vec(unsigned const size, free_list_t::pool_type & pool)
{
	free_list_t * v = static_cast<free_list_t *>(calloc(size, sizeof(free_list_t)));
	if (v != nullptr) {
		for (unsigned i = 0; i < size; ++i)
			placement_new(&v[i], pool);
	} else {
		fprintf(stderr,
				"Bad alloc occured during initialize a free list vector of the buddy system!\n");
	}
	return v;
}

void buddy_system::_cleanup_free_list_vec(unsigned const size, free_list_t * v)
{
	for (unsigned i = 0; i < size; ++i) {
		// assert(i == 0 || v[i].empty());
		placement_delete(&v[i]);
	}
	delete v;
}

void buddy_system::_split_block(buddy_block *		parent,
								buddy_block *		left,
								buddy_block *		right,
								buddy_table const & tbl)
{
	assert(parent->in_use == false);

	auto const left_block_index = [&tbl](buddy_block const * parent) -> blkidx_t {
		auto const	   prop			  = tbl.property(parent->blkidx);
		blkidx_t const parent_lv_base = parent->blkidx - prop.offset;
		if (prop.check(UniqueBuddyBlock))
			return parent_lv_base + 1;
		if (parent->cof & 0x1u)
			return parent_lv_base + 2;
		return parent_lv_base + 2 + (prop.offset != 0);
	};

	auto const right_block_index = [&tbl](buddy_block const * parent) -> blkidx_t {
		auto const	   prop			  = tbl.property(parent->blkidx);
		blkidx_t const parent_lv_base = parent->blkidx - prop.offset;
		if (prop.check(UniqueBuddyBlock))
			return parent_lv_base + 1 + (parent->cof & 0x1u);
		if (parent->cof & 0x1u)
			return parent_lv_base + 3;
		return parent_lv_base + 2 + (prop.offset != 0);
	};

	// left
	left->cof	   = parent->cof / 2 + parent->cof % 2;
	left->rgn.ptr  = parent->rgn.ptr;
	left->rgn.size = tbl.align() * left->cof;
	left->pair	   = right;
	left->parent   = parent;
	left->in_use   = false;
	left->inv	   = nullptr;
	left->blkidx   = left_block_index(parent);

	// right
	right->cof		= parent->cof - left->cof;
	right->rgn.ptr	= seek_pointer(parent->rgn.ptr, left->rgn.size);
	right->rgn.size = parent->rgn.size - left->rgn.size;
	right->pair		= left;
	right->parent	= parent;
	right->in_use	= false;
	right->inv		= nullptr;
	right->blkidx	= right_block_index(parent);
}

void buddy_system::_deallocate(buddy_block * block)
{
	assert(block->in_use == true);
	block->in_use	   = false;
	buddy_block * pair = block->pair;
	if (block->pair == nullptr || pair->in_use) {
		block->inv = _flist_v[block->blkidx].emplace_back(block).node();
		return;
	}
	buddy_block * parent = block->parent;
	_flist_v[pair->blkidx].remove_node(pair->inv);
	_block_pool.deallocate(block);
	_block_pool.deallocate(pair);
	_deallocate(parent);
	_status.total_deallocated += 1;
}

void buddy_system::_cleanup()
{
	// TODO: Implementation
	if (_flist_v == nullptr)
		return; // system is not initialized yet.
	if (_flist_v[0].empty()) {
		// fprintf(stderr, "Buddy system detects memory leak!\n");
	}
	_cleanup_free_list_vec(_tbl.size(), _flist_v);
	_tbl.clear();
}

buddy_block * buddy_system::_acquire_block(blkidx_t const bidx) const
{
	free_list_t & list = _flist_v[bidx];
	if (list.empty())
		return nullptr;
	auto		  begin = list.begin();
	buddy_block * block = *begin;
	_flist_v[bidx].remove_node(begin);
	return block;
}

/*
 * * Root: 200
 * Minimum coefficient: 3
 *
 * Coefficient tree
 *              +---          [232]             ---> 232 [Root]
 *              |               |
 *              |             [116]             ---> 116
 *   Linear  <--+               |
 *              |             [58]              ---> 58
 *              |               |
 *              +---          [29]              ---> 29
 *              |            /    \
 *              |         [15]     [14]         ---> 15, 14
 *   Binary  <--+        /  \       /  \
 *              |      [8]  [7]    [7]  [7]     ---> 8, 7 [A1B3-Pattern]
 *              |      / \  / \    / \   / \
 *              +--- [4][4][4][3] [4][3][4][3]  ---> 4, 3 [A3B1-Pattern]
 *
 * Buddy table
 * - Flag: Unique (U), Frequent (F), Rare (R), A1B3-Pattern, A3B1-Pattern
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * | Idx | Lev |  Cof  | U | F | R | Pattern | Dist | Off |
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * |  0  |  0  |  232  | # |   |   |   N/A   |   0  |  0  | => 0: U, Root
 * |  1  |  1  |  116  | # |   |   |   N/A   |   1  |  0  | => 1: U
 * |  2  |  2  |   58  | # |   |   |   N/A   |   1  |  0  | => 2: U
 * |  3  |  3  |   29  | # |   |   |   N/A   |   1  |  0  | => 3: U
 * |  4  |  4  |   15  |   |   | # |   A3B1  |   1  |  0  | => 4: R-A3B1*
 * |  5  |  4  |   14  |   |   | # |   A3B1  |   2  |  1  | => 5: R-A3B1*
 * |  6  |  5  |    8  |   |   | # |   A1B3  |   2  |  0  | => 6: R-A1B3
 * |  7  |  5  |    7  |   | # |   |   A1B3  |   3  |  1  | => 7: F-A1B3
 * |  8  |  6  |    4  |   | # |   |   A3B1  |   2  |  0  | => 8: F-A3B1
 * |  9  |  6  |    3  |   |   | # |   A3B1  |   3  |  1  | => 9: R-A3B1
 * +-----+-----+-------+---+---+---+---------+------+-----+
 * Note that the patterns of a first binary level (4) are assumed to be R-A3B1*
 *
 * Initial state of:
 * Free-list vector         | Allocation tree (* means free node)
 * +-----+-----------+      |               [232:0x00]*
 * | Idx | Free-list |      |
 * +-----+-----------+      |
 * |  0  |    0x00   |      |
 * |  1  |    NULL   |      |
 * |  2  |    NULL   |      |
 * |  3  |    NULL   |      |
 * |  4  |    NULL   |      |
 * |  5  |    NULL   |      |
 * |  6  |    NULL   |      |
 * |  7  |    NULL   |      |
 * |  8  |    NULL   |      |
 * |  9  |    NULL   |      |
 * +-----+-----------+      |
 *
 * Create a route of seed 9 for a first allocation:
 * +------+-----+--------+------------+------+--------+---------------------+
 * | Step | Idx | Lookup | Properties | Cand | Parent |        Route        |
 * +------+-----+--------+------------+------+--------+---------------------+
 * |   0  |  9  |  MISS  |   R-A3B1   | 8, 9 |  6, 7  | 8                   |
 * |  1-1 |  6  |  MISS  |   R-A1B3   |   6  |    4   |                     |
 * |  1-2 |  7  |  MISS  |   F-A1B3   | 6, 7 |  4, 5  | 8->7                |
 * |  2-1 |  5  |  MISS  |   R-A3B1*  |   5  |    3   |                     |
 * |  2-2 |  4  |  MISS  |   R-A3B1*  |   4  |    3   | 8->7->4             |
 * |   3  |  3  |  MISS  |      U     |   3  |    2   | 8->7->4->3          |
 * |   4  |  2  |  MISS  |      U     |   2  |    1   | 8->7->4->3->2       |
 * |   5  |  1  |  MISS  |      U     |   1  |    0   | 8->7->4->3->2->1    |
 * |   6  |  0  |   HIT  |      U     |   0  |   NULL | 8->7->4->3->2->1->0 |
 * +------+-----+--------+------------+------+--------+---------------------+
 *
 * States after the first allocation:
 * Free-list vector         | Allocation tree (* means free node)
 * +-----+-----------+      |                          [232:0x00]
 * | Idx | Free-list |      |                          /         \
 * +-----+-----------+      |                  [116:0x10]        [116:0x11]*
 * |  0  |    NULL   |      |                  /         \
 * |  1  |    0x11   |      |          [58:0x20]         [58:0x21]*
 * |  2  |    0x21   |      |           |       \
 * |  3  |    0x31   |      |        [29:0x30]  [29:0x31]*
 * |  4  |    0x41   |      |           |     \
 * |  5  |    NULL   |      |      [15:0x40]  [14:0x41]*
 * |  6  |    NULL   |      |        |      \
 * |  7  |    0x50   |      |    [8:0x50]*  [7:0x51]
 * |  8  |    NULL   |      |                /     \
 * |  9  |    0x61   |      |           [4:0x60]  [3:0x61]*
 * +-----+-----------+      |              ^
 *                          |              |
 *                          |              +-- return this (request: 3, result 4)
 *
 * Create a route of seed 9 for a second allocation:
 * +------+-----+--------+------------+------+--------+---------------------+
 * | Step | Idx | Lookup | Properties | Cand | Parent |        Route        |
 * +------+-----+--------+------------+------+--------+---------------------+
 * |   0  |  9  |   HIT  |   R-A3B1   | 8, 9 |  6, 7  | 9 (Cache hit)       |
 * +------+-----+--------+------------+------+--------+---------------------+
 *
 * States after the second allocation:
 * Free-list vector         | Allocation tree (* means free node)
 * +-----+-----------+      |                          [232:0x00]
 * | Idx | Free-list |      |                          /         \
 * +-----+-----------+      |                  [116:0x10]        [116:0x11]*
 * |  0  |    NULL   |      |                  /         \
 * |  1  |    0x11   |      |          [58:0x20]         [58:0x21]*
 * |  2  |    0x21   |      |           |       \
 * |  3  |    0x31   |      |        [29:0x30]  [29:0x31]*
 * |  4  |    0x41   |      |           |     \
 * |  5  |    NULL   |      |      [15:0x40]  [14:0x41]*
 * |  6  |    NULL   |      |        |      \
 * |  7  |    0x50   |      |    [8:0x50]*  [7:0x51]
 * |  8  |    NULL   |      |                /     \
 * |  9  |    NULL   |      |           [4:0x60]  [3:0x61]*
 * +-----+-----------+      |                        ^
 *                          |                        |
 *                          |                        +-- here (request: 3, result 3)
 *
 */

buddy_system::routing_result buddy_system::_create_route(blkidx_t bidx)
{
	auto const append_idx_to_route_if_cached = [this](blkidx_t const idx) -> bool {
		auto const prop = _tbl.property(idx);
		if (!_flist_v[idx].empty()) {
			_route.push(prop.offset);
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
			_route_dbg.push(idx);
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
			return true;
		}
		return false;
	};

#ifdef MIXX_BUDDY_SYSTEM_PREVENT_ROOT_ALLOC
	// Terminate if the target block is a root node
	if (MIXX_UNLIKELY(blkidx == 0))
		return routing_result{false, 0}; // bad alloc
#endif

	// Lookup caches of requested block index
	if (append_idx_to_route_if_cached(bidx))
		return routing_result{true, bidx};

	// If the requested block is R-A3B1,
	// restart the routine to allocate a neighbor block
	auto prop = _tbl.property(bidx);
	if (prop.check(RareBuddyBlock | A3B1Pattern))
		return _create_route(bidx - 1);

	// Append a current index to the route
	_route.push(prop.offset);
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
	_route_dbg.push(bidx);
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING

	// If the requested block is R-A1B3,
	// restart the routine to allocate its parent block
	if (prop.check(RareBuddyBlock | A1B3Pattern))
		return _create_route(bidx - prop.dist);

	do {
		// Move the index to the beginning of the previous level
		bidx -= prop.dist;
		prop = _tbl.property(bidx);
		if (prop.check(UniqueBuddyBlock)) {
			if (append_idx_to_route_if_cached(bidx))
				return routing_result{true, bidx};
			// Append the current index to the route (Unique: 0)
			_route.push(0);
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
			_route_dbg.push(bidx);
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		} else if (prop.check(A1B3Pattern)) {
			if (append_idx_to_route_if_cached(bidx))
				return routing_result{true, bidx};
			if (append_idx_to_route_if_cached(bidx + 1))
				return routing_result{true, bidx + 1};
			// Append the frequent block index to the route (A1B3 => Offset of B: 1)
			_route.push(1);
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
			_route_dbg.push(bidx + 1);
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		} else if (prop.check(A3B1Pattern)) {
			if (append_idx_to_route_if_cached(bidx + 1))
				return routing_result{true, bidx + 1};
			if (append_idx_to_route_if_cached(bidx))
				return routing_result{true, bidx};
			// Append the frequent block index to the route (A3B1 => Offset of A: 0)
			_route.push(0);
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
			_route_dbg.push(bidx);
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
		}
	} while (bidx > 0);

	if (append_idx_to_route_if_cached(0))
		return routing_result{true, 0};

	return routing_result{false, bidx};
}

portable_buddy_system::portable_buddy_system(memrgn_t const & rgn, unsigned align, unsigned min_cof)
{
	init(rgn, align, min_cof);
}

void portable_buddy_system::init(memrgn_t const & rgn, unsigned align, unsigned min_cof)
{
	buddy.init(rgn, align, min_cof);
}

void * portable_buddy_system::allocate(uint64_t size)
{
	buddy_block * block = buddy.allocate_block(size);
	if (block == nullptr)
		return nullptr;
	void * p = block->rgn.ptr;
	assert(hashmap.find(p) == hashmap.end());
	hashmap.emplace(p, block);
	return p;
}

void portable_buddy_system::deallocate(void * p)
{
	assert(hashmap.find(p) != hashmap.end());
	buddy_block * block = hashmap.at(p);
	buddy.deallocate_block(block);
	hashmap.erase(p);
}

} // namespace _buddy_impl