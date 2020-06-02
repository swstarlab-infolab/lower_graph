/*
 * Portable version of GStream buddy system based on 5/6/2020 version.
 * A suballocator for the any memory region including preallocated pinned memory or GPU device memory.
 * 5/6/2020
 * Seyeon Oh
 */

#ifndef C126BD0E_DC93_472C_912E_D1755628B23E
#define C126BD0E_DC93_472C_912E_D1755628B23E
#include <assert.h>
#include <memory>
#include <stdint.h>
#include <string.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

typedef struct memory_region {
    void* ptr;
    uint64_t size;

    bool is_null() const
    {
        return ptr == nullptr;
    }
} memrgn_t;

namespace _buddy_impl {

constexpr unsigned char UniqueBuddyBlock = 0x1u;
constexpr unsigned char FrequentBuddyBlock = 0x1u << 1;
constexpr unsigned char RareBuddyBlock = 0x1u << 2;
constexpr unsigned char A1B3Pattern = 0x1u << 3;
constexpr unsigned char A3B1Pattern = 0x1u << 4;
using cof_type = int64_t;
using blkidx_t = unsigned;
using level_t = unsigned;
typedef struct buddy_block_propoerty {
    unsigned char flags : 5;
    unsigned char dist : 2;
    unsigned char offset : 1;

    bool check(unsigned char bits) const
    {
        return (flags & bits) == bits;
    }
} blk_prop_t;
static_assert(sizeof(buddy_block_propoerty) == 1, "a size of buddy_block_property is not 1");

namespace _noncopyable { // protection from unintended ADL

    class noncopyable {
    public:
        noncopyable(const noncopyable&) = delete;
        noncopyable& operator=(const noncopyable&) = delete;

    protected:
        /*constexpr*/ noncopyable() = default;
        ~noncopyable() = default;
    };

} // !namespace _noncopyable

class buddy_table : _noncopyable::noncopyable {
public:
    buddy_table();
    ~buddy_table() noexcept;
    buddy_table(cof_type root_cof, unsigned align, cof_type min_cof);
    void init(cof_type root_cof, unsigned align, cof_type min_cof);
    void clear() noexcept;
    void printout() const;

    blkidx_t best_fit(uint64_t block_size) const;

    unsigned align() const
    {
        return _align;
    }

    unsigned size() const
    {
        return _tbl_size;
    }

    level_t max_level() const
    {
        return _attr.level_v[_tbl_size - 1];
    }

    level_t level(blkidx_t const bidx) const
    {
        return _attr.level_v[bidx];
    }

    cof_type cof(blkidx_t const bidx) const
    {
        return _attr.cof_v[bidx];
    }

    blk_prop_t const& property(blkidx_t const bidx) const
    {
        return _attr.prof_v[bidx];
    }

protected:
    struct tbl_attr {
        level_t* level_v;
        cof_type* cof_v;
        blk_prop_t* prof_v;
    };
    static void _free_attr(tbl_attr* attr);
    static tbl_attr _init_attr(unsigned tbl_size, cof_type root, unsigned align, cof_type min_cof);

    unsigned _align;
    cof_type _min_cof;
    level_t _tbl_size;
    level_t _buddy_lv;
    tbl_attr _attr;
};

template <typename T, typename... Args>
inline void placement_new(T* p, Args&&... args)
{
    new (p) T(std::forward<Args>(args)...);
}

template <typename T>
inline void placement_delete(T* p)
{
    p->~T();
}

class list_impl {
public:
    template <typename T>
    struct list_node {
        using value_type = T;
        value_type value;
        list_node* prev;
        list_node* next;
    };

    template <typename T>
    class list_iterator {
    public:
        using node_type = list_node<T>;
        using value_type = T;
        explicit list_iterator(void* p = nullptr)
        {
            _p = static_cast<node_type*>(p);
        }

        inline list_iterator& operator++()
        {
            _p = _p->next;
            return *this;
        }

        inline list_iterator& operator++(int)
        {
            list_iterator tmp(*this);
            _p = _p->next;
            return tmp;
        }

        inline value_type& operator*()
        {
            return _p->value;
        }

        inline bool operator==(list_iterator const& rhs)
        {
            return _p == rhs._p;
        }

        inline bool operator!=(list_iterator const& rhs)
        {
            return _p != rhs._p;
        }

        inline node_type* node() const
        {
            return _p;
        }

        inline void invalidate()
        {
            _p = nullptr;
        }

    private:
        node_type* _p;
    };

    template <typename List, typename... Args>
    static typename List::iterator emplace_back_to(List& list, typename List::iterator const& target, Args&&... args)
    {
        auto dest = target.node();
        auto next = dest->next;
        auto node = list._allocate_node();
        placement_new<typename List::value_type>(&node->value, std::forward<Args>(args)...);
        node->prev = dest;
        node->next = next;
        dest->next = node;
        if (next != nullptr)
            next->prev = node;
        else
            list._tail = node;
        list._size += 1;
        return typename List::iterator { node };
    }

    template <typename List, typename... Args>
    static typename List::iterator emplace_front_of(List& list, typename List::iterator const& target, Args&&... args)
    {
        auto dest = target.node();
        auto prev = dest->prev;
        auto node = list._allocate_node();
        placement_new<typename List::value_type>(&node->value, std::forward<Args>(args)...);
        node->prev = prev;
        node->next = dest;
        dest->prev = node;
        prev->next = node;
        list._size == 1;
        return typename List::iterator { node };
    }

    template <typename List>
    static typename List::iterator remove_node(List& list, typename List::iterator const& target)
    {
        auto dest = target.node();
        auto prev = dest->prev;
        auto next = dest->next;
        placement_delete<typename List::value_type>(&dest->value);
        prev->next = next;
        if (next != nullptr)
            next->prev = prev;
        else
            list._tail = prev;
        list._deallocate_node(dest);
        list._size -= 1;
        return typename List::iterator { next };
    }
};

namespace detail {

    namespace _seq_cont_manipulation {

        template <typename T>
        using pod_type_tracing = std::integral_constant<bool, std::is_pod<T>::value>;

        template <typename T>
        using pod_type = std::integral_constant<bool, true>;

        template <typename T>
        using non_pod_t = std::integral_constant<bool, false>;

        template <typename T>
        void copy_array(T* __restrict dst, T* __restrict src, size_t size, pod_type<T>)
        {
            memcpy(dst, src, sizeof(T) * size);
        }

        template <typename T>
        void copy_array(T* __restrict dst, T* __restrict src, size_t size, non_pod_t<T>)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = src[i];
        }

        template <typename T>
        void move_array(T* __restrict dst, T* __restrict src, size_t size, pod_type<T>)
        {
            memcpy(dst, src, sizeof(T) * size);
        }

        template <typename T>
        void move_array(T* __restrict dst, T* __restrict src, size_t size, non_pod_t<T>)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = std::move(src[i]);
        }

        template <typename T>
        void clear_array(T* dst, size_t size, pod_type<T>)
        {
            memset(dst, 0, sizeof(T) * size);
        }

        template <typename T>
        void clear_array(T* dst, size_t size, non_pod_t<T>)
        {
            for (size_t i = 0; i < size; ++i)
                dst.~T();
        }

    } // !namespace _seq_cont_manipulation

} // !namespace detail

template <typename T>
void copy_array(T* __restrict dst, T* __restrict src, size_t size)
{
    using namespace detail;
    using namespace _seq_cont_manipulation;
    copy_array(dst, src, size, pod_type_tracing<T>());
}

template <typename T>
void move_array(T* __restrict dst, T* __restrict src, size_t size)
{
    using namespace detail;
    using namespace _seq_cont_manipulation;
    move_array(dst, src, size, pod_type_tracing<T>());
}

template <typename T>
void clear_array(T* dst, size_t size)
{
    using namespace detail;
    using namespace _seq_cont_manipulation;
    clear_array(dst, size, pod_type_tracing<T>());
}

template <typename ValueType, typename Allocator = std::allocator<ValueType>>
class stack {
public:
    using value_type = ValueType;
    using alloc_type = Allocator;
    explicit stack(size_t capacity_ = 256, alloc_type const& alloc = alloc_type());
    stack(stack const& other);
    stack(stack&& other) noexcept;
    ~stack() noexcept;
    stack& operator=(stack const& rhs);
    stack& operator=(stack&& rhs) noexcept;
    size_t push(value_type const& lvalue);
    size_t push(value_type&& rvalue);
    template <typename... Args>
    size_t emplace(Args&&...);
    bool pop();
    bool pop(value_type*);
    value_type& peek();
    bool empty() const;
    size_t capacity() const;
    size_t size() const;
    void clear();
    void reserve(size_t new_capacity);
    void iterate(void (*f)(value_type&));

    void _config(size_t top);
    value_type* _get_buffer();

protected:
    void _release_container();
    value_type* _acquire_buffer();

    alloc_type _alloc;
    size_t _cap;
    size_t _top;
    value_type* _cont;
};

template <typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type roundup(const T n, const T m)
{
    return ((n + ((n >= 0) ? 1 : 0) * (m - 1)) / m) * m;
}

template <typename T>
inline typename std::enable_if<std::is_unsigned<T>::value, T>::type roundup(const T n, const T m)
{
    return ((n + m - 1) / m) * m;
}

inline uint32_t roundup2_nonzero(uint32_t n)
{
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    ++n;
    return n;
}

inline uint32_t roundup2(uint32_t n)
{
    n += (n == 0);
    return roundup2_nonzero(n);
}

inline uint64_t roundup2_nonzero(uint64_t n)
{
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    ++n;
    return n;
}

inline uint64_t roundup2(uint64_t n)
{
    n += (n == 0);
    return roundup2_nonzero(n);
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>::stack(size_t const capacity_, alloc_type const& alloc)
    : _alloc(alloc)
{
    _cap = roundup2(capacity_);
    _top = 0;
    _cont = std::allocator_traits<alloc_type>::allocate(_alloc, _cap);
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>::stack(stack const& other)
    : _alloc(other._alloc)
{
    _cap = other._cap;
    _top = other._top;
    _cont = std::allocator_traits<alloc_type>::allocate(_alloc, _cap);
    copy_array(_cont, other._cont, _top);
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>::stack(stack&& other) noexcept
    : _alloc(std::move(other._alloc))
{
    _cap = other._cap;
    _top = other._top;
    _cont = other._cont;
    other._cont = nullptr;
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>::~stack() noexcept
{
    _release_container();
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>& stack<ValueType, Allocator>::operator=(stack const& rhs)
{
    _release_container();
    _alloc = rhs._alloc;
    _cap = rhs._cap;
    _top = rhs._top;
    _cont = std::allocator_traits<alloc_type>::allocate(_alloc, _cap);
    copy_array(_cont, rhs._cont, _top);
    return *this;
}

template <typename ValueType, typename Allocator>
stack<ValueType, Allocator>& stack<ValueType, Allocator>::operator=(stack&& rhs) noexcept
{
    _release_container();
    _alloc = std::move(rhs._alloc);
    _cap = rhs._cap;
    _top = rhs._top;
    _cont = rhs._cont;
    rhs._cont = nullptr;
    return *this;
}

template <typename ValueType, typename Allocator>
size_t stack<ValueType, Allocator>::push(value_type const& lvalue)
{
    value_type* buf = _acquire_buffer();
    new (buf) value_type(lvalue);
    return _top - 1;
}

template <typename ValueType, typename Allocator>
size_t stack<ValueType, Allocator>::push(value_type&& rvalue)
{
    value_type* buf = _acquire_buffer();
    new (buf) value_type(std::move(rvalue));
    return _top - 1;
}

template <typename ValueType, typename Allocator>
template <typename... Args>
size_t stack<ValueType, Allocator>::emplace(Args&&... args)
{
    value_type* buf = _acquire_buffer();
    new (buf) value_type(std::forward<Args>(args)...);
    return _top - 1;
}

template <typename ValueType, typename Allocator>
bool stack<ValueType, Allocator>::pop()
{
    if (empty())
        return false;
    value_type& val = peek();
    val.~value_type();
    _top -= 1;
    return true;
}

template <typename ValueType, typename Allocator>
bool stack<ValueType, Allocator>::pop(value_type* out)
{
    if (empty())
        return false;
    value_type& val = peek();
    *out = std::move(val);
    val.~value_type();
    _top -= 1;
    return true;
}

template <typename ValueType, typename Allocator>
typename stack<ValueType, Allocator>::value_type& stack<ValueType, Allocator>::peek()
{
    assert(_top > 0);
    return _cont[_top - 1];
}

template <typename ValueType, typename Allocator>
bool stack<ValueType, Allocator>::empty() const
{
    return _top == 0;
}

template <typename ValueType, typename Allocator>
size_t stack<ValueType, Allocator>::capacity() const
{
    return _cap;
}

template <typename ValueType, typename Allocator>
size_t stack<ValueType, Allocator>::size() const
{
    return _top;
}

template <typename ValueType, typename Allocator>
void stack<ValueType, Allocator>::clear()
{
    clear_array(_cont, size());
}

template <typename ValueType, typename Allocator>
void stack<ValueType, Allocator>::reserve(size_t new_capacity)
{
    if (_cap >= new_capacity)
        return;
    new_capacity = roundup2(new_capacity);
    assert(_top < new_capacity);
    value_type* old = _cont;
    _cont = std::allocator_traits<alloc_type>::allocate(_alloc, new_capacity);
    copy_array(_cont, old, size());
    std::allocator_traits<alloc_type>::deallocate(_alloc, old, _cap);
    _cap = new_capacity;
}

template <typename ValueType, typename Allocator>
void stack<ValueType, Allocator>::iterate(void (*f)(value_type&))
{
    if (size() == 0)
        return;
    for (size_t i = _top; i != 0; --i) {
        f(_cont[i - 1]);
    }
}

template <typename ValueType, typename Allocator>
void stack<ValueType, Allocator>::_config(size_t const top)
{
    _top = top;
}

template <typename ValueType, typename Allocator>
typename stack<ValueType, Allocator>::value_type* stack<ValueType, Allocator>::_get_buffer()
{
    return _cont;
}

template <typename ValueType, typename Allocator>
void stack<ValueType, Allocator>::_release_container()
{
    if (_cont != nullptr)
        std::allocator_traits<alloc_type>::deallocate(_alloc, _cont, _cap);
}

template <typename ValueType, typename Allocator>
typename stack<ValueType, Allocator>::value_type* stack<ValueType, Allocator>::_acquire_buffer()
{
    if (_top == _cap)
        reserve(_cap * 2);
    return &_cont[_top++];
}

class segregated_storage final : _noncopyable::noncopyable {
public:
    segregated_storage(void* preallocated, size_t bufsize, size_t block_size);
    void* allocate();
    void deallocate(void* p);
    void reset();
    double fill_rate() const;

    void* const buffer;
    size_t const bufsize;
    size_t const block_size;
    size_t const capacity;

private:
    stack<void*> _free_list;
};

template <typename ValueType, size_t ClusterSize = 1024, typename Allocator = std::allocator<ValueType>>
class object_pool : _noncopyable::noncopyable {
public:
    using value_type = ValueType;
    using alloc_type = Allocator;
    constexpr static size_t cluster_size = ClusterSize;
    constexpr static double recycle_factor = 0.5F;

    explicit object_pool(size_t reserved = cluster_size, const alloc_type& alloc = alloc_type {});
    ~object_pool() noexcept;
    template <typename... Args>
    value_type* construct(Args&&... args);
    value_type* allocate();
    value_type* allocate_zero_initialized();
    void destroy(value_type* p);
    void deallocate(value_type* p);
    void reserve(size_t required);

    size_t num_clusters() const
    {
        return _num_nodes;
    }

    size_t capacity() const
    {
        return _capacity;
    }

protected:
    struct block_type {
        void* key;
        value_type value;
    };
    constexpr static size_t cluster_region_size = sizeof(block_type) * cluster_size;
    using cluster_t = segregated_storage;

    struct cluster_node {
        cluster_node() = delete;
        char buffer[cluster_region_size];
        cluster_t cluster;
        cluster_node* next;
        cluster_node* prev;
        bool stacked;
    };

    using node_alloc_t = typename std::allocator_traits<alloc_type>::
        template rebind_alloc<cluster_node>;
    using node_stack_t = stack<cluster_node*,
        typename std::allocator_traits<alloc_type>::
            template rebind_alloc<cluster_node*>>;

    cluster_node* _allocate_node()
    {
        cluster_node* node = std::allocator_traits<node_alloc_t>::allocate(_node_alloc, 1);
        placement_new(&node->cluster, node->buffer, cluster_region_size, cluster_size);
        node->next = nullptr;
        node->prev = nullptr;
        node->stacked = false;
        _num_nodes += 1;
        _capacity += cluster_size;
        return node;
    }

    void _deallocate_node(cluster_node* node)
    {
        placement_delete(&node->cluster);
        std::allocator_traits<node_alloc_t>::deallocate(_node_alloc, node, 1);
        _num_nodes -= 1;
        _capacity -= cluster_size;
    }

    void _insert_back_to(cluster_node* target, cluster_node* node)
    {
        cluster_node* next = target->next;
        node->next = target->next;
        node->prev = target;
        target->next = node;
        if (next != nullptr)
            next->prev = node;
    }

    cluster_node* _detach_node(cluster_node* node) noexcept
    {
        cluster_node* prev = node->prev;
        cluster_node* next = node->next;
        if (prev != nullptr)
            prev->next = next;
        if (next != nullptr)
            next->prev = prev;
        return next;
    }

    alloc_type _alloc;
    size_t _capacity;
    size_t _num_nodes;
    node_alloc_t _node_alloc;
    node_stack_t _node_stack;
    cluster_node* _curr;
};

template <typename ValueType, size_t ClusterSize, typename Allocator>
object_pool<ValueType, ClusterSize, Allocator>::object_pool(size_t reserved, const alloc_type& alloc)
    : _alloc(alloc)
    , _capacity(0)
    , _num_nodes(0)
    , _node_alloc(alloc)
    , _node_stack(256)
{
    assert(_capacity % cluster_size == 0);
    _curr = _allocate_node();
    reserve(reserved);
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
object_pool<ValueType, ClusterSize, Allocator>::~object_pool() noexcept
{
#ifdef MIXX_DEBUG_ENABLE_OBJECT_LEAK_DETECTION
    assert(_curr->prev == nullptr);
    assert(_curr->next == nullptr);
    assert(_node_stack.size() == _num_nodes - 1);
#endif // !MIXX_DEBUG_ENABLE_OBJECT_LEAK_DETECTION
    _deallocate_node(_curr);
    while (!_node_stack.empty()) {
        cluster_node* p;
        _node_stack.pop(&p);
        _deallocate_node(p);
    }
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
template <typename... Args>
typename object_pool<ValueType, ClusterSize, Allocator>::value_type* object_pool<ValueType, ClusterSize, Allocator>::construct(Args&&... args)
{
    value_type* p = this->allocate();
    new (p) value_type(std::forward<Args>(args)...);
    return p;
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
typename object_pool<ValueType, ClusterSize, Allocator>::value_type* object_pool<ValueType, ClusterSize, Allocator>::allocate()
{
    auto deploy_block = [](block_type* blk, cluster_node* key) -> value_type* {
        blk->key = key;
        return &blk->value;
    };

    // Try to allocate a block from the current cluster
    {
        block_type* blk = static_cast<block_type*>(_curr->cluster.allocate());
        if (blk != nullptr)
            return deploy_block(blk, _curr);
    }

    // Try to pop a free cluster from the stack
    {
        cluster_node* node;
        if (_node_stack.pop(&node)) {
            assert(node->stacked == true);
            node->stacked = false;
            _insert_back_to(_curr, node);
            _curr = node;
            block_type* blk = static_cast<block_type*>(_curr->cluster.allocate());
            assert(blk != nullptr);
            return deploy_block(blk, _curr);
        }
    }

    // Allocate a new cluster
    assert(_curr->next == nullptr);
    cluster_node* node = _allocate_node();
    _insert_back_to(_curr, node);
    _curr = node;
    block_type* blk = static_cast<block_type*>(_curr->cluster.allocate());
    assert(blk != nullptr);
    return deploy_block(blk, _curr);
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
typename object_pool<ValueType, ClusterSize, Allocator>::value_type* object_pool<ValueType, ClusterSize, Allocator>::allocate_zero_initialized()
{
    value_type* p = this->allocate();
    memset(p, 0, sizeof(value_type));
    return p;
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
void object_pool<ValueType, ClusterSize, Allocator>::destroy(value_type* p)
{
    p->~value_type();
    this->deallocate(p);
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
void object_pool<ValueType, ClusterSize, Allocator>::deallocate(value_type* p)
{
    block_type* blk = reinterpret_cast<block_type*>(reinterpret_cast<char*>(p) - sizeof(void*));
    cluster_node* node = static_cast<cluster_node*>(blk->key);
    node->cluster.deallocate(blk);
    if (node == _curr)
        return;
    if (node->stacked)
        return;
    if (node->cluster.fill_rate() >= recycle_factor) {
        _detach_node(node);
        node->stacked = true;
        _node_stack.push(node);
    }
}

template <typename ValueType, size_t ClusterSize, typename Allocator>
void object_pool<ValueType, ClusterSize, Allocator>::reserve(size_t required)
{
    if (required <= _capacity)
        return;
    size_t remained = roundup(required, cluster_size) - _capacity;
    assert(remained > 0);
    assert(remained % cluster_size == 0);
    remained /= cluster_size;
    for (size_t i = 0; i < remained; ++i) {
        cluster_node* node = _allocate_node();
        node->stacked = true;
        _node_stack.push(node);
    }
    _capacity += remained * cluster_size;
}

template <typename T, size_t ClusterSize = 256>
using list_node_pool_trait = object_pool<list_impl::list_node<T>, ClusterSize>;

/* class pooling_list */

template <typename T, typename Pool = list_node_pool_trait<T>>
class pooling_list : _noncopyable::noncopyable {
    friend class list_impl;

public:
    using node_type = list_impl::list_node<T>;
    using node_pointer = node_type*;
    using value_type = T;
    using pool_type = Pool;
    using iterator = list_impl::list_iterator<T>;

    explicit pooling_list(pool_type& pool);
    ~pooling_list() noexcept;
    template <typename... Args>
    iterator emplace_front_of(iterator const& target, Args&&... args);
    template <typename... Args>
    iterator emplace_back_to(iterator const& target, Args&&... args);
    template <typename... Args>
    iterator emplace_front(Args&&... args);
    template <typename... Args>
    iterator emplace_back(Args&&... args);
    iterator remove_node(iterator const& target);
    iterator remove_node(node_pointer const& target);
    void clear();

    inline iterator begin() const
    {
        return iterator { _head->next };
    }

    inline iterator tail() const
    {
        return iterator { _tail };
    }

    inline iterator end() const
    {
        return iterator { nullptr };
    }

    inline bool empty() const
    {
        return _head->next == nullptr;
    }

    inline size_t size() const
    {
        return _size;
    }

private:
    template <typename... Args>
    static inline void _init_value(node_type* node, Args... args)
    {
        new (&node->value) value_type(std::forward<Args>(args)...);
    }

    static inline void _free_value(node_type* node)
    {
        node->value.~value_type();
    }

    node_type* _allocate_node();
    void _deallocate_node(node_type* node);

    pool_type& _pool;
    node_pointer _head, _tail;
    size_t _size;
};

template <typename T, typename PoolTy>
pooling_list<T, PoolTy>::pooling_list(pool_type& pool)
    : _pool(pool)
{
    _tail = _head = _allocate_node();
    _head->prev = nullptr;
    _head->next = nullptr;
    _size = 0;
}

template <typename T, typename PoolTy>
pooling_list<T, PoolTy>::~pooling_list() noexcept
{
    try {
        clear();
        _deallocate_node(_head);
    } catch (...) {
        fprintf(stderr, "Failed to cleanup pooling list!");
        std::abort();
    }
}

template <typename T, typename PoolTy>
template <typename... Args>
typename pooling_list<T, PoolTy>::iterator pooling_list<T, PoolTy>::
    emplace_front_of(iterator const& target, Args&&... args)
{
    return list_impl::emplace_front_of(*this, target, std::forward<Args>(args)...);
}

template <typename T, typename PoolTy>
template <typename... Args>
typename pooling_list<T, PoolTy>::iterator pooling_list<T, PoolTy>::
    emplace_back_to(iterator const& target, Args&&... args)
{
    return list_impl::emplace_back_to(*this, target, std::forward<Args>(args)...);
}

template <typename T, typename PoolTy>
template <typename... Args>
typename pooling_list<T, PoolTy>::iterator pooling_list<T, PoolTy>::emplace_front(Args&&... args)
{
    return emplace_back_to(iterator(_head), std::forward<Args>(args)...);
}

template <typename T, typename PoolTy>
template <typename... Args>
typename pooling_list<T, PoolTy>::iterator pooling_list<T, PoolTy>::emplace_back(Args&&... args)
{
    return emplace_back_to(iterator(_tail), std::forward<Args>(args)...);
}

template <typename T, typename PoolTy>
typename pooling_list<T, PoolTy>::iterator pooling_list<T, PoolTy>::remove_node(iterator const& target)
{
    return list_impl::remove_node(*this, target);
}

template <typename T, typename Pool>
typename pooling_list<T, Pool>::iterator pooling_list<T, Pool>::remove_node(node_pointer const& target)
{
    return list_impl::remove_node(*this, iterator { target });
}

template <typename T, typename PoolTy>
void pooling_list<T, PoolTy>::clear()
{
    while (_head->next != nullptr)
        remove_node(iterator(_head->next));
}

template <typename T, typename PoolTy>
typename pooling_list<T, PoolTy>::node_type* pooling_list<T, PoolTy>::_allocate_node()
{
    return _pool.allocate();
}

template <typename T, typename PoolTy>
void pooling_list<T, PoolTy>::_deallocate_node(node_type* node)
{
    _pool.deallocate(node);
}

/* class private_pooling_list */

template <typename ValueType, unsigned ClusterSize = 256, typename Allocator = std::allocator<ValueType>>
class private_pooling_list : _noncopyable::noncopyable {
    using node_type = list_impl::list_node<ValueType>;

public:
    using value_type = ValueType;
    using alloc_type = Allocator;
    using iterator = list_impl::list_iterator<value_type>;
    constexpr static unsigned cluster_size = ClusterSize;
    explicit private_pooling_list(size_t pool_reserved = cluster_size, const alloc_type& alloc = alloc_type());
    ~private_pooling_list() noexcept;
    template <typename... Args>
    iterator emplace_front_of(iterator target, Args&&... args);
    template <typename... Args>
    iterator emplace_back_to(iterator target, Args&&... args);
    template <typename... Args>
    iterator emplace_front(Args&&... args);
    template <typename... Args>
    iterator emplace_back(Args&&... args);
    iterator remove_node(iterator target) noexcept;
    void reserve_pool(size_t size);
    void clear();

    inline iterator begin() const
    {
        return iterator(_head->next);
    }

    inline iterator tail() const
    {
        return iterator(_tail);
    }

    inline iterator end() const
    {
        return iterator(nullptr);
    }

    inline bool empty() const
    {
        return _head->next == nullptr;
    }

private:
    using alproxy_t = typename std::allocator_traits<alloc_type>::template rebind_alloc<node_type>;
    node_type* _head;
    node_type* _tail;
    size_t _size;
    object_pool<node_type, cluster_size, alproxy_t> _pool;
    node_type* _allocate_node();
    void _deallocate_node(node_type* node) noexcept;

    template <typename... Args>
    static inline void _construct_data(node_type* node, Args... args)
    {
        new (&node->value) value_type(std::forward<Args>(args)...);
    }

    static inline void _destruct_data(node_type* node)
    {
        node->value.~value_type();
    }
};

template <typename ValueType, unsigned ClusterSize, typename Allocator>
private_pooling_list<ValueType, ClusterSize, Allocator>::private_pooling_list(size_t pool_reserved, const alloc_type& alloc)
    : _pool(pool_reserved, alloc)
{
    node_type* head = _allocate_node();
    head->next = nullptr;
    head->prev = nullptr;
    _tail = _head = head;
    _size = 0;
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
private_pooling_list<ValueType, ClusterSize, Allocator>::~private_pooling_list() noexcept
{
    clear();
    _deallocate_node(_head);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
template <typename... Args>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::iterator
private_pooling_list<ValueType, ClusterSize, Allocator>::emplace_front_of(iterator target, Args&&... args)
{
    return list_impl::emplace_front_of(*this, target, std::forward<Args>(args)...);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
template <typename... Args>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::iterator
private_pooling_list<ValueType, ClusterSize, Allocator>::emplace_back_to(iterator target, Args&&... args)
{
    return list_impl::emplace_back_to(*this, target, std::forward<Args>(args)...);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
template <typename... Args>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::iterator
private_pooling_list<ValueType, ClusterSize, Allocator>::emplace_front(Args&&... args)
{
    return emplace_back_to(iterator(_head), std::forward<Args>(args)...);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
template <typename... Args>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::iterator
private_pooling_list<ValueType, ClusterSize, Allocator>::emplace_back(Args&&... args)
{
    return emplace_back_to(iterator(_tail), std::forward<Args>(args)...);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::iterator
private_pooling_list<ValueType, ClusterSize, Allocator>::remove_node(iterator target) noexcept
{
    return list_impl::remove_node(*this, target);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
void private_pooling_list<ValueType, ClusterSize, Allocator>::reserve_pool(size_t size)
{
    _pool.reserve(size);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
void private_pooling_list<ValueType, ClusterSize, Allocator>::clear()
{
    while (_head->next != nullptr)
        remove_node(iterator(_head->next));
    assert(_size == 0);
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
typename private_pooling_list<ValueType, ClusterSize, Allocator>::node_type* private_pooling_list<ValueType, ClusterSize, Allocator>::_allocate_node()
{
    return _pool.allocate();
}

template <typename ValueType, unsigned ClusterSize, typename Allocator>
void private_pooling_list<ValueType, ClusterSize, Allocator>::_deallocate_node(node_type* node) noexcept
{
    _pool.deallocate(node);
}

class bit_stack {
public:
    explicit bit_stack(size_t capacity_ = 256);
    bit_stack(bit_stack const& other);
    bit_stack(bit_stack&& other) noexcept;
    ~bit_stack() noexcept;
    bit_stack& operator=(bit_stack const& rhs);
    bit_stack& operator=(bit_stack&& rhs) noexcept;
    bool push(bool value);
    bool pop();
    bool peek() const;
    void clear();
    bool reserve(size_t new_capacity);

    bool push(unsigned const value)
    {
        return push(value != 0);
    }

    bool push(int const value)
    {
        return push(value != 0);
    }

    size_t size() const
    {
        return _top;
    }

    bool empty() const
    {
        return _top == 0;
    }

protected:
    size_t _cap;
    size_t _top;
    int* _container;
};

struct buddy_block;

using free_list_t = pooling_list<buddy_block*>;

struct buddy_block {
    cof_type cof;
    buddy_block* pair;
    buddy_block* parent;
    bool in_use;
    free_list_t::node_pointer inv;
    blkidx_t blkidx;
    memrgn_t rgn;
};

struct buddy_system_status {
    uint64_t total_allocated;
    uint64_t total_deallocated;
};

class buddy_system {
    //using buddy_block = buddy_block;

public:
    buddy_system();
    ~buddy_system();
    buddy_system(memrgn_t const& rgn, unsigned align, unsigned min_cof);
    void init(memrgn_t const& rgn, unsigned align, unsigned min_cof);
    void* allocate(uint64_t size);
    void deallocate(void* p);
    buddy_block* allocate_block(uint64_t size);
    void deallocate_block(buddy_block* blk);

    memrgn_t const& rgn() const
    {
        return _rgn;
    }

    inline uint64_t max_alloc() const
    {
        return _max_blk_size;
    }

protected:
    struct routing_result {
        bool success;
        blkidx_t blkidx;
    };

    static free_list_t* _init_free_list_vec(unsigned size, free_list_t::pool_type& pool);
    static void _cleanup_free_list_vec(unsigned size, free_list_t* v);
    static void _split_block(buddy_block* parent, buddy_block* left, buddy_block* right, buddy_table const& tbl);

    void _deallocate(buddy_block* block);
    void _cleanup();
    buddy_block* _acquire_block(blkidx_t bidx) const;
    routing_result _create_route(blkidx_t bidx);

    memrgn_t _rgn;
    unsigned _align;
    uint64_t _max_blk_size;
    free_list_t::pool_type _node_pool;
    object_pool<buddy_block, 256> _block_pool;
    free_list_t* _flist_v;
    bit_stack _route;
    buddy_system_status _status;
    buddy_table _tbl;
#ifdef MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
    stack<unsigned> _route_dbg;
#endif // !MIXX_DEBUG_ENABLE_BUDDY_ROUTE_CORRECTNESS_CHECKING
    int64_t _total_allocated_size;
};

class portable_buddy_system {
public:
    portable_buddy_system() = default;
    portable_buddy_system(memrgn_t const& rgn, unsigned align, unsigned min_cof);
    void init(memrgn_t const& rgn, unsigned align, unsigned min_cof);
    void* allocate(uint64_t size);
    void deallocate(void* p);

protected:
    std::unordered_map<void*, buddy_block*> hashmap;
    buddy_system buddy;
};

} // namespace _buddy_impl

using _buddy_impl::portable_buddy_system;
#endif /* C126BD0E_DC93_472C_912E_D1755628B23E */
