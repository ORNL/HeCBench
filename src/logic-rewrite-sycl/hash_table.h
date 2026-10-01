#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <climits>

#include "common.h"

// *************** declaration ***************


template <typename KeyT, typename ValueT>
class HashTable {
    sycl::queue &q_ct1 = nsycl::queue();
    static_assert(sizeof(uint32) == 4, "Size of uint32 must be 4 bytes.");
    static_assert(sizeof(uint64) == 8, "Size of uint64 must be 8 bytes.");
    static_assert(
        std::is_same<KeyT, uint32>::value || std::is_same<KeyT, uint64>::value, 
        "Not supported key data type."
    );
    static_assert(
        std::is_same<ValueT, uint32>::value || std::is_same<ValueT, uint64>::value, 
        "Not supported value data type."
    );

public:
    HashTable() = delete;
    HashTable(int capacity) {
        this->capacity = capacity;

        keys = (KeyT *)sycl::malloc_device(capacity * sizeof(KeyT), q_ct1);
        values =
            (ValueT *)sycl::malloc_device(capacity * sizeof(ValueT), q_ct1);
        q_ct1.memset(keys, 0xff, capacity * sizeof(KeyT)).wait();
        q_ct1.memset(values, 0xff, capacity * sizeof(ValueT)).wait();
        allocated = true;
    }

    ~HashTable() {
        freeMem();
    }

    void freeMem() {
        if (allocated) {
            sycl::free(keys, q_ct1);
            sycl::free(values, q_ct1);
            allocated = false;
        }
    }

    void insert_batch_no_update(const KeyT * keys, const ValueT * values, const int len);
    void insert_batch_no_update_masked(const KeyT * keys, const ValueT * values, const int * mask, const int len);
    void retrieve_batch(const KeyT * keys, ValueT * values, const int len);
    void retrieve_batch_masked(const KeyT * keys, ValueT * values, const int * mask, const int len);
    int retrieve_all(KeyT * keys, ValueT * values, const int buffer_len, const int sorted = 0);
    int get_capacity() {return capacity;}
    KeyT * get_keys_storage() {return keys;}
    ValueT * get_values_storage() {return values;}

    inline static constexpr KeyT empty_key() {
        if (std::is_same<KeyT, uint32>::value)
            return UINT_MAX;
        else
            return ULLONG_MAX;
    }
    inline static constexpr ValueT empty_value() {
        if (std::is_same<ValueT, uint32>::value)
            return UINT_MAX;
        else
            return ULLONG_MAX;
    }

private:
    KeyT * keys;
    ValueT * values;
    int capacity;
    bool allocated;
};



// *************** definition ***************

template <typename KeyT, typename ValueT>
constexpr KeyT get_hashtable_empty_key() {
    if constexpr (std::is_same<KeyT, uint32>::value || std::is_same<KeyT, uint64>::value || 
                  std::is_same<ValueT, uint32>::value || std::is_same<ValueT, uint64>::value)
        return HashTable<KeyT, ValueT>::empty_key();
    else {
        static_assert(
            std::is_same<KeyT, int>::value, "Not supported key data type."
        );
        return -1;
    }
}

template <typename KeyT, typename ValueT>
constexpr ValueT get_hashtable_empty_value() {
    if constexpr (std::is_same<KeyT, uint32>::value || std::is_same<KeyT, uint64>::value || 
                  std::is_same<ValueT, uint32>::value || std::is_same<ValueT, uint64>::value)
        return HashTable<KeyT, ValueT>::empty_value();
    else {
        static_assert(
            std::is_same<ValueT, int>::value, "Not supported value data type."
        );
        return -1;
    }
}



template <typename KeyT, typename ValueT>
const KeyT HASHTABLE_EMPTY_KEY = get_hashtable_empty_key<KeyT, ValueT>();

template <typename KeyT, typename ValueT>
const ValueT HASHTABLE_EMPTY_VALUE = get_hashtable_empty_value<KeyT, ValueT>();

template <typename KeyT>
const KeyT SET_EMPTY_KEY = get_hashtable_empty_key<KeyT, KeyT>();

template <int capacity>
constexpr int maybeGetPrime() {
    // get the largest prime number that is smaller than capacity
    // valid for capacity <= 1024
    if constexpr (capacity == 32)
        return 31;
    if constexpr (capacity == 64)
        return 61;
    if constexpr (capacity == 128)
        return 127;
    if constexpr (capacity == 256)
        return 251;
    if constexpr (capacity == 512)
        return 509;
    if constexpr (capacity == 1024)
        return 1021;
    return capacity;
}

inline uint64 hash(uint64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53;
    x ^= x >> 33;
    return x;
}

inline uint32 hash(uint32 x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

inline uint32 hash(int x) {
    return hash((uint32)x);
}

inline uint32 rotl32(uint32 x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32 hashArray(const void *pArray, int nBytes) {
    // murmur3 hashing
    // make sure that nBytes is a multiple of 4. i.e., word size is 32-bit.
    
    int nBlocks = nBytes / 4;
    unsigned h1 = 2023;

    const unsigned c1 = 0xcc9e2d51;
    const unsigned c2 = 0x1b873593;

    const unsigned * puArray = (unsigned *) pArray;

    for (int i = 0; i < nBlocks; i++) {
        unsigned k1 = puArray[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    h1 ^= nBytes;
    h1 = hash(h1);
    return h1;
}

template <typename KeyT, typename ValueT>
void insert_batch_no_update_kernel(KeyT * ht_keys, ValueT * ht_values, 
                                              const KeyT * keys, const ValueT * values, 
                                              const int len, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < len) {
        KeyT key = keys[idx];
        ValueT value = values[idx];
        KeyT loc = hash(key);
        loc = loc % capacity;

        while (true) {
            KeyT prev = nsycl::atomic_compare_exchange(
                &ht_keys[loc], HASHTABLE_EMPTY_KEY<KeyT, ValueT>, key);
            if (prev == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
                // found a empty entry
                ht_values[loc] = value;
                return;
            } else if (prev == key) {
                // already have key inserted, no update
                return;
            }

            loc = (loc + 1) % capacity;
        }
    }
}

template <typename KeyT, typename ValueT>
void insert_batch_no_update_masked_kernel(KeyT * ht_keys, ValueT * ht_values, 
                                                     const KeyT * keys, const ValueT * values, const int * mask,
                                                     const int len, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < len && mask[idx] == 1) {
        KeyT key = keys[idx];
        ValueT value = values[idx];
        KeyT loc = hash(key);
        loc = loc % capacity;

        while (true) {
            KeyT prev = nsycl::atomic_compare_exchange(
                &ht_keys[loc], HASHTABLE_EMPTY_KEY<KeyT, ValueT>, key);
            if (prev == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
                // found a empty entry
                ht_values[loc] = value;
                return;
            } else if (prev == key) {
                // already have key inserted, no update
                return;
            }

            loc = (loc + 1) % capacity;
        }
    }
}

template <typename KeyT, typename ValueT>
void retrieve_batch_kernel(KeyT * ht_keys, ValueT * ht_values, 
                                      const KeyT * keys, ValueT * values,
                                      const int len, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < len) {
        KeyT key = keys[idx];
        KeyT loc = hash(key);
        loc = loc % capacity;

        while (true) {
            if (ht_keys[loc] == key) {
                values[idx] = ht_values[loc];
                return;
            } else if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
                values[idx] = HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
                return;
            }

            loc = (loc + 1) % capacity;
        }
    }
}

template <typename KeyT, typename ValueT>
void retrieve_batch_masked_kernel(KeyT * ht_keys, ValueT * ht_values, 
                                             const KeyT * keys, ValueT * values, const int * mask,
                                             const int len, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < len  && mask[idx] == 1) {
        KeyT key = keys[idx];
        KeyT loc = hash(key);
        loc = loc % capacity;

        while (true) {
            if (ht_keys[loc] == key) {
                values[idx] = ht_values[loc];
                return;
            } else if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
                values[idx] = HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
                return;
            }

            loc = (loc + 1) % capacity;
        }
    }
}


template <typename KeyT, typename ValueT>
void getEntryIndicator(int * indicator, KeyT * ht_keys, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < capacity) {
        indicator[idx] = (ht_keys[idx] != HASHTABLE_EMPTY_KEY<KeyT, ValueT>);
    }
}

template <typename KeyT, typename ValueT>
void gatherKeyValues(const int * scanned_indicator, const KeyT * ht_keys, const ValueT * ht_values, 
                                KeyT * keys, ValueT * values, const int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx == 0) {
        if (scanned_indicator[0] == 1) {
            keys[0] = ht_keys[0];
            values[0] = ht_values[0];
        }
    } else if (idx < capacity) {
        if (scanned_indicator[idx] > scanned_indicator[idx - 1]) {
            int dst_idx = scanned_indicator[idx] - 1;
            keys[dst_idx] = ht_keys[idx];
            values[dst_idx] = ht_values[idx];
        }
    }
}

template <typename KeyT, typename ValueT>
void HashTable<KeyT, ValueT>::insert_batch_no_update(const KeyT * keys, const ValueT * values, const int len) {
    nsycl::queue().submit([&](sycl::handler &cgh) {
        auto this_keys_ct0 = this->keys;
        auto this_values_ct1 = this->values;
        auto capacity_ct5 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(len, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                insert_batch_no_update_kernel<KeyT, ValueT>(
                    this_keys_ct0, this_values_ct1, keys, values, len,
                    capacity_ct5);
            });
    });
    // cudaDeviceSynchronize();
}

template <typename KeyT, typename ValueT>
void HashTable<KeyT, ValueT>::insert_batch_no_update_masked(const KeyT * keys, const ValueT * values, 
                                                                     const int * mask, const int len) {
    nsycl::queue().submit([&](sycl::handler &cgh) {
        auto this_keys_ct0 = this->keys;
        auto this_values_ct1 = this->values;
        auto capacity_ct6 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(len, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                insert_batch_no_update_masked_kernel<KeyT, ValueT>(
                    this_keys_ct0, this_values_ct1, keys, values, mask, len,
                    capacity_ct6);
            });
    });
    // cudaDeviceSynchronize();
}

template <typename KeyT, typename ValueT>
void HashTable<KeyT, ValueT>::retrieve_batch(const KeyT * keys, ValueT * values, const int len) {
    nsycl::queue().submit([&](sycl::handler &cgh) {
        auto this_keys_ct0 = this->keys;
        auto this_values_ct1 = this->values;
        auto capacity_ct5 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(len, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                retrieve_batch_kernel<KeyT, ValueT>(this_keys_ct0,
                                                    this_values_ct1, keys,
                                                    values, len, capacity_ct5);
            });
    });
    // cudaDeviceSynchronize();
}

template <typename KeyT, typename ValueT>
void HashTable<KeyT, ValueT>::retrieve_batch_masked(const KeyT * keys, ValueT * values, const int * mask, const int len) {
    nsycl::queue().submit([&](sycl::handler &cgh) {
        auto this_keys_ct0 = this->keys;
        auto this_values_ct1 = this->values;
        auto capacity_ct6 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(len, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                retrieve_batch_masked_kernel<KeyT, ValueT>(
                    this_keys_ct0, this_values_ct1, keys, values, mask, len,
                    capacity_ct6);
            });
    });
    // cudaDeviceSynchronize();
}

template <typename KeyT, typename ValueT>
int HashTable<KeyT, ValueT>::retrieve_all(KeyT *keys, ValueT *values,
                                          const int buffer_len,
                                          const int sorted) {
    sycl::queue &q_ct1 = nsycl::queue();
    int num_entries;
    int * indicator;
    indicator = sycl::malloc_device<int>(capacity, q_ct1);

    q_ct1.submit([&](sycl::handler &cgh) {
        auto this_keys_ct1 = this->keys;
        auto capacity_ct2 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(capacity, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                getEntryIndicator<KeyT, ValueT>(indicator, this_keys_ct1,
                                                capacity_ct2);
            });
    });
    q_ct1.wait();
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(q_ct1), indicator,
        indicator + capacity, indicator);
    q_ct1.wait();
    q_ct1.memcpy(&num_entries, &indicator[capacity - 1], sizeof(int)).wait();

    if (num_entries > buffer_len) {
        printf("[HashTable::retrieve_all] Error: provided buffer len is smaller than number of entries!\n");
        sycl::free(indicator, q_ct1);
        return 0;
    }

    q_ct1.submit([&](sycl::handler &cgh) {
        auto this_keys_ct1 = this->keys;
        auto this_values_ct2 = this->values;
        auto capacity_ct5 = capacity;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(capacity, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                gatherKeyValues<KeyT, ValueT>(indicator, this_keys_ct1,
                                              this_values_ct2, keys, values,
                                              capacity_ct5);
            });
    });
    q_ct1.wait();

    if (sorted) {
        // sort according to values
        // note that "key" passed to thrust::sort_by_key is actually our values
        oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q_ct1), values,
                   values + num_entries, keys);
        q_ct1.wait();
    }

    sycl::free(indicator, q_ct1);
    return num_entries;
}

// *************** device functions ***************

inline void unbindAndNodeKeys(const uint64 key, uint32 *lit1,
                                       uint32 *lit2) {
    *lit2 = (uint32)(key & 0xffffffffUL);
    *lit1 = (uint32)(key >> 32);
}

inline uint64 formAndNodeKey(const int lit1, const int lit2) {
    // make sure lit1 is smaller than lit2
    uint32 uLit1 = (uint32)lit1;
    uint32 uLit2 = (uint32)lit2;
    uint64 key = ((uint64)uLit1) << 32 | uLit2;
    return key;
}

inline uint32 checkTrivialAndCases(int lit1, int lit2) {
    if (lit1 == lit2)
        return (uint32)lit1;
    if (lit1 == dUtils::AigNodeNot(lit2))
        return (uint32)dUtils::AigConst0;
    if (lit1 == dUtils::AigConst1)
        return (uint32)lit2;
    if (lit2 == dUtils::AigConst1)
        return (uint32)lit1;
    if (lit1 == dUtils::AigConst0 || lit2 == dUtils::AigConst0)
        return (uint32)dUtils::AigConst0;
    
    // non-trivial
    return HASHTABLE_EMPTY_VALUE<uint64, uint32>;
}

// to use these functions, one must call Hashtable.get_xx to obtain raw device pointers

// IMPORTANT if using retrieve_single and insert_single_no_update in a single kernel, 
//           remember to call __threadfence between the two

template <typename KeyT, typename ValueT>
inline ValueT retrieve_single(const KeyT *ht_keys,
                                       const ValueT *ht_values, const KeyT key,
                                       const int capacity) {
    KeyT loc = hash(key);
    loc = loc % capacity;

    while (true) {
        if (ht_keys[loc] == key) {
            return ht_values[loc];
        } else if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            return HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
        }
        loc = (loc + 1) % capacity;
    }
}

template <typename KeyT, typename ValueT>
inline uint32 insert_single_no_update(KeyT *ht_keys, ValueT *ht_values,
                                               const KeyT key,
                                               const ValueT value,
                                               const int capacity) {
    KeyT loc = hash(key);
    loc = loc % capacity;

    while (true) {
        KeyT prev = nsycl::atomic_compare_exchange(
            &ht_keys[loc], HASHTABLE_EMPTY_KEY<KeyT, ValueT>, key);
        if (prev == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            // found a empty entry
            ht_values[loc] = value;
            return 2;
        } else if (prev == key) {
            // already have key inserted, no update
            return ht_values[loc];
        }

        loc = (loc + 1) % capacity;
    }
}

template <typename KeyT, typename ValueT>
inline ValueT retrieve_single_volatile(
    volatile const KeyT *ht_keys, volatile const ValueT *ht_values,
    const KeyT key, const int capacity) {
    KeyT loc = hash(key);
    loc = loc % capacity;

    while (true) {
        if (ht_keys[loc] == key) {
            return ht_values[loc];
        } else if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            return HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
        }
        loc = (loc + 1) % capacity;
    }
}

template <typename KeyT, typename ValueT>
inline uint32 insert_single_no_update_volatile(
    volatile KeyT *ht_keys, volatile ValueT *ht_values, const KeyT key,
    const ValueT value, const int capacity) {
    KeyT loc = hash(key);
    loc = loc % capacity;

    while (true) {
        KeyT prev = nsycl::atomic_compare_exchange(
            (KeyT *)&ht_keys[loc], HASHTABLE_EMPTY_KEY<KeyT, ValueT>, key);
        if (prev == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            // found a empty entry
            ht_values[loc] = value;
            return 2;
        } else if (prev == key) {
            // already have key inserted, no update
            return ht_values[loc];
        }

        loc = (loc + 1) % capacity;
    }
}



// single-threaded device memory hash table functions

template <typename KeyT, typename ValueT>
inline void st_map_clear(KeyT *ht_keys, ValueT *ht_values,
                                  const int capacity) {
    for (int i = 0; i < capacity; i++) {
        ht_keys[i] = HASHTABLE_EMPTY_KEY<KeyT, ValueT>;
        ht_values[i] = HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
    }
}

template <typename KeyT, typename ValueT>
inline ValueT st_map_insert_or_query(KeyT *ht_keys, ValueT *ht_values,
                                              const KeyT key,
                                              const ValueT value,
                                              const int capacity) {
    // if the returned value does not equal the inserted one, 
    // then there already exists a key in the hashtable with value equals the returned one
    int loc = (int)hash(key);
    loc = loc % capacity;
    while (true) {
        if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            ht_keys[loc] = key;
            ht_values[loc] = value;
            return value;
        } else if (ht_keys[loc] == key) {
            return ht_values[loc];
        }
        loc = (loc + 1) % capacity;
    }
}

template <typename KeyT, typename ValueT>
inline ValueT st_map_query(const KeyT *ht_keys,
                                    const ValueT *ht_values, const KeyT key,
                                    const int capacity) {
    int loc = (int)hash(key);
    loc = loc % capacity;
    while (true) {
        if (ht_keys[loc] == key) {
            return ht_values[loc];
        } else if (ht_keys[loc] == HASHTABLE_EMPTY_KEY<KeyT, ValueT>) {
            return HASHTABLE_EMPTY_VALUE<KeyT, ValueT>;
        }
        loc = (loc + 1) % capacity;
    }
}

template <typename KeyT>
inline void st_set_clear(KeyT *ht_keys, const int capacity) {
    for (int i = 0; i < capacity; i++) {
        ht_keys[i] = SET_EMPTY_KEY<KeyT>;
    }
}

template <typename KeyT>
inline bool st_set_insert(KeyT *ht_keys, const KeyT key,
                                   const int capacity) {
    int loc = (int)hash(key);
    loc = loc % capacity;
    while (true) {
        if (ht_keys[loc] == SET_EMPTY_KEY<KeyT>) {
            ht_keys[loc] = key;
            return true;
        } else if (ht_keys[loc] == key) {
            return false;
        }
        loc = (loc + 1) % capacity;
    }
    assert(0);
}

template <typename KeyT>
inline int st_set_locate(const KeyT *ht_keys, const KeyT key,
                                  const int capacity) {
    int loc = (int)hash(key);
    loc = loc % capacity;
    while (true) {
        if (ht_keys[loc] == key) {
            return loc;
        } else if (ht_keys[loc] == SET_EMPTY_KEY<KeyT>) {
            return -1;
        }
        loc = (loc + 1) % capacity;
    }
    assert(0);
}
