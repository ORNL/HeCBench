/*********************************************************************************
Copyright (c) 2016 Marius Erbert, Steffen Rechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*********************************************************************************/

#ifndef _COUNTINGHASHTABLE_H_
#define _COUNTINGHASHTABLE_H_

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "_KeyValuePair.h"

namespace cuda_ds {
	namespace internal {

		template<class KeyType>
		constexpr uint32_t intsPerKey() {
			return sizeof(KeyType) / sizeof(uint32_t);
		}

		/*****************************************************************
		 * External Declaration of CUDA Procedures.
		 * Needed to keep cuda code separated from cpp code.
		 ******************************************************************/

		/**
		 * Add key bundle to hash table.
		 */
		template<uint32_t intsPerKey>
		extern void addToTable(KeyValuePair<intsPerKey> *table, // pointer to the hash table
		                       const uint64_t numEntries,    // maximum number of entries in table
		                       const Key<intsPerKey> *keyBundle,// pointer to the keys that shall be insert
		                       const uint64_t numKeys,                // number of keys to be inserted
		                       cudaStream_t &stream,                // cuda stream to use
		                       Key<intsPerKey> *noSuccessArea,    // pointer to no success area (aka area of shame)
		                       uint64_t *numNoSuccessPtr,// pointer to number of unsuccessful inserted keys
		                       const uint64_t maxNumNoSuccess// maximum number of keys that can be inserted unsuccessfully
		);

		template<uint32_t intsPerKey>
		extern uint64_t _compress(KeyValuePair<intsPerKey> *table,
		                      const uint64_t numEntries, uint64_t *result);

		/**
		 * Sort Keys in gpu memory.
		 */
		template<uint32_t intsPerKey>
		extern void sortKeys(Key<intsPerKey> *keys, const uint64_t numKeys);

		/**
		 * A Data Structure for Counting keys. It is based on the idea
		 * of a counting hash table.
		 * To add method inserts a bundle of keys into the hash table.
		 * The key counts can be determined by calling the count method.
		 *
		 * @param KeyType The type of the keys. The only condition on the
		 * KeyType is that its size must be smaller then 32*4 byte.
		 * @param keyBufferSize The size of a bundle of keys in bytes.
		 * @param countBufferSize The size of the count Buffer in bytes.
		 */
		template<uint32_t intsPerKey, uint64_t keyBufferSize>
		class CountingHashTable {

		protected:

			// constants and tuning parameters
			const static uint32_t numStreams = 4;
			const static uint64_t minNoSuccessAreaSize = 32 * 1024 * 1024;  // 32 MB

			/**************************************************************************
			 * Data in global memory and pointers to it
			 *************************************************************************/
			char *data;                            // hash table data in raw form
			Key<intsPerKey> *keyBuffer;    // pointer to segment with keys to be inserted
			KeyValuePair<intsPerKey> *table;    // pointer to hash table segment
			Key<intsPerKey> *noSuccessArea;        // pointer to area of shame
			uint64_t *numNoSuccessPtr;// number of unsuccessful inserted kmers in current run
			uint64_t *tmp;                        // temporary memory

			/**************************************************************************
			 * Sizes of Segments, etc.
			 *************************************************************************/
			uint64_t size;                // number of bytes allocated to the hash table
			uint64_t maxNumNoSuccess;    // maximal capacity of the no-success-area
			uint64_t numEntries;        // maximal number of entries in hash table
			uint64_t numNoSuccess;        // number of free slots in noSuccessArea
			uint64_t keysProcessed;        // number of keys processed since last check

			// auxilliary data
			cudaStream_t stream[numStreams];    // cuda streams
			uint32_t currentStream;                // stream id
			uint32_t devID;                        // devide ID of cuda device

			/**
			 * Create a new hash table which can use up to size bytes of device memory.
			 *
			 * @param devID The id of the device that is to be used.
			 */
			CountingHashTable(const uint32_t devID = 0) :
					data(nullptr), keyBuffer(nullptr), table(nullptr), noSuccessArea(
					nullptr), numNoSuccessPtr(nullptr), size(0), maxNumNoSuccess(
					0), numEntries(0), numNoSuccess(0), keysProcessed(0), currentStream(
					0), devID(devID) {

				// set device id
				cudaSetDevice(devID);

				// determine amount of memory
				uint64_t freeMem, totalMem;
				cudaMemGetInfo(&freeMem, &totalMem);

				// allocate gpu memory
				size = (uint64_t) (0.8 * freeMem);

				// hash table
				cudaError_t err;
				err = cudaMalloc(&data, size);
				if (err != cudaSuccess) {
					throw std::runtime_error(
							std::string(
									std::string("cuda_ds::CountingHashTable::CountingHashTable::exception1: ")
									+ cudaGetErrorName(err)) + std::string(": ")
							+ std::string(cudaGetErrorString(err)));
				}

				// create cuda streams
				for (uint32_t i = 0; i < numStreams; i++) {
					err = cudaStreamCreate(&stream[i]);
					if (err != cudaSuccess) {
						throw std::runtime_error(
								std::string(
										std::string("cuda_ds::CountingHashTable::CountingHashTable::exception2: ")
										+ cudaGetErrorName(err))
								+ std::string(": ")
								+ std::string(cudaGetErrorString(err)));
					}
				}

				// do dummy-init
				init(0);
			}

			virtual ~CountingHashTable() {

				// set device id
				cudaSetDevice(devID);

				cudaError_t err;

				// wait for synchronize
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess) {
					fprintf(stderr,
							"cuda_ds::~CountingHashTable::exception1: %s: %s\n",
							cudaGetErrorName(err), cudaGetErrorString(err));
				}

				// free data
				err = cudaFree(data);
				if (err != cudaSuccess) {
					fprintf(stderr,
							"cuda_ds::~CountingHashTable::exception2: %s: %s\n",
							cudaGetErrorName(err), cudaGetErrorString(err));
				}

				// destroy streams
				for (int i = 0; i < numStreams; ++i) {
					err = cudaStreamDestroy(stream[i]);
					if (err != cudaSuccess) {
						fprintf(stderr,
								"cuda_ds::CountingHashTable::~CountingHashTable::exception3: %s: %s\n",
								cudaGetErrorName(err), cudaGetErrorString(err));
					}
				}
			}

			/**
			 * Add a bundle of keys to the hash table.
			 * @param keyBundle A bundle of keys of size keyBufferSize.
			 * @return True, if keys could be inserted without problems.
			 * False, if there is not enough free space to ensure the
			 * insertion of all keys. In this case, no key of this bundle
			 * will be inserted.
			 */
			bool add(const Key<intsPerKey> *keyBundle, const uint64_t numKeys) {

				/*printf("add: numkeys         = %lu\n", numKeys);
				printf("add: numNoSuccess    = %lu\n", numNoSuccess);
				printf("add: keysProcessed   = %lu\n", keysProcessed);
				printf("add: maxNumNoSuccess = %lu\n", maxNumNoSuccess);*/

				// set device id
				cudaSetDevice(devID);

				// if there is not enough room to insert all keys safely
				if (numNoSuccess + keysProcessed + numKeys > maxNumNoSuccess) {

					// refresh the number of kmers that could not be inserted in the hash table
					cudaMemcpy(&numNoSuccess, numNoSuccessPtr, sizeof(uint64_t),
					           cudaMemcpyDeviceToHost);

					// reset keysProcessed
					keysProcessed = 0;

					// if there is still not enough room to insert all keys safely
					if (numNoSuccess + keysProcessed + numKeys > maxNumNoSuccess) {
						// return false and do not insert anything. extract required!
						return false;
					}
				}

				// buffered copy the keys to gpu and launch add kernel
				uint64_t itemsPerBuffer = keyBufferSize / sizeof(Key<intsPerKey>);

				// for all keys in chunks of size itemsPerBuffer
				for (uint64_t i = 0; i < numKeys; i += itemsPerBuffer) {

					// number of keys in this chunk
					const uint64_t numKeysInChunk =
							(i + 1) * itemsPerBuffer <= numKeys ?
							itemsPerBuffer : (numKeys % itemsPerBuffer);

					// pointer to key buffer (gpu side)
					Key<intsPerKey> *keyBufferPtr =
							(Key<intsPerKey> *) ((char *) keyBuffer
							                     + currentStream * keyBufferSize);

					// copy to gpu
					cudaError_t err = cudaMemcpyAsync(keyBufferPtr, &keyBundle[i],
					                numKeysInChunk * sizeof(Key<intsPerKey>),
					                cudaMemcpyHostToDevice, stream[currentStream]);
					if (err != cudaSuccess) {
						throw std::runtime_error(
								std::string("cuda_ds::CountingHashTable::add::exception1: ")
								+ std::string(cudaGetErrorName(err)) + " "
								+ std::string(cudaGetErrorString(err)));
					}

					// launch Kernel to do the cuda stuff
					addToTable<intsPerKey>(table, numEntries, keyBufferPtr,
					                       numKeysInChunk, stream[currentStream], noSuccessArea,
					                       numNoSuccessPtr, maxNumNoSuccess);

					// switch to next stream
					currentStream++;
					if (currentStream == numStreams)
						currentStream = 0;
				}

				// update number of processed keys
				keysProcessed += numKeys;

				return true;
			}

		public:

			/**
			 * Initialize the hash table for a certain number of
			 * expected distinct keys.
			 *
			 * @param expectedNumberOfDistinctKeys The number of entries is set to be
			 * this number. Note that this number should neither be set to be too large
			 * nor to small.
			 */
			void init(const uint64_t expectedNumberOfDistinctKeys) {

				const size_t maxcap = getMaxCapacity();
				numEntries = std::min(maxcap, expectedNumberOfDistinctKeys);

				// set device id
				cudaSetDevice(devID);

				// wait for synchronize
				cudaError_t err = cudaDeviceSynchronize();
				if (err != cudaSuccess) {
					throw std::runtime_error(
							std::string("cuda_ds::CountingHashTable::init::exception1: ")
							+ std::string(cudaGetErrorName(err)) + " "
							+ std::string(cudaGetErrorString(err)));
				}

				// distribute pre-allocated memory:
				uint64_t offset = 0;

				// hash table
				table = (KeyValuePair<intsPerKey> *) &data[offset];
				offset += numEntries * sizeof(KeyValuePair<intsPerKey>);

				// kmer bundles to be inserted
				offset = 256 * ((offset + 255) / 256);    // round to next multiple of 256
				keyBuffer = (Key<intsPerKey> *) &data[offset];
				offset += numStreams * keyBufferSize;

				// numNoSuccess
				offset = 256 * ((offset + 255) / 256);    // round to next multiple of 256
				numNoSuccessPtr = (uint64_t *) &data[offset];
				offset += sizeof(uint64_t);

				// tmp
				offset = 256 * ((offset + 255) / 256);    // round to next multiple of 256
				tmp = (uint64_t *) &data[offset];
				offset += sizeof(uint64_t);

				// noSuccess Area
				offset = 256 * ((offset + 255) / 256);    // round to next multiple of 256
				noSuccessArea = (Key<intsPerKey> *) &data[offset];
				maxNumNoSuccess = (size - offset) / sizeof(Key<intsPerKey>);

				if (offset + minNoSuccessAreaSize > size)
					throw std::runtime_error(
							std::string(
									"cuda_ds::CountingHashTable::init::exception2: Allocated Device Memory not sufficient for hash table with ")
							+ std::to_string(numEntries)
							+ std::string(" entries (max. capacity is ")
							+ std::to_string(getMaxCapacity())
							+ std::string(")!"));

				if (maxNumNoSuccess < keyBufferSize / sizeof(Key<intsPerKey>))
					throw std::runtime_error(
							std::string(
									"cuda_ds::CountingHashTable::init::exception3: Not enough free memory available for key bundles of size  ")
							+ std::to_string(keyBufferSize) + "!");

				// do the initialization of the memory
				clear();
			}

			/**
			 * Clears the Hash Table. Do not change the size of the table.
			 */
			void clear() {

				// set device id
				cudaSetDevice(devID);

				cudaError_t err;

				// init hash table with zero
				err = cudaMemset(table, 0,
				                 numEntries * sizeof(KeyValuePair<intsPerKey>));
				if (err != cudaSuccess) {
					throw std::runtime_error(
							std::string(
									"cuda_ds::CountingHashTable::clear::exception1: ")
							+ std::string(cudaGetErrorName(err)) + " "
							+ std::string(cudaGetErrorString(err)));
				}

				// init numNoSuccess
				err = cudaMemset(numNoSuccessPtr, 0, sizeof(uint64_t));
				if (err != cudaSuccess) {
					throw std::runtime_error(
							std::string(
									"cuda_ds::CountingHashTable::clear::exception2: ")
							+ std::string(cudaGetErrorName(err)) + " "
							+ std::string(cudaGetErrorString(err)));
				}

				keysProcessed = 0;
				numNoSuccess = 0;
			}

			/**
			 * Wait for current threads to success.
			 */
			void join() const {

				// wait for threads to success
				cudaError_t err = cudaDeviceSynchronize();
				if (err != cudaSuccess) {
					throw std::runtime_error(
							std::string("gerbil::exception: ")
							+ std::string(cudaGetErrorName(err)) + " "
							+ std::string(cudaGetErrorString(err)));
				}

			}

			/**
			 * Returns the number of table entries of the current configuration.
			 * This number can be changed by calling init().
			 */
			uint64_t getNumEntries() const {
				return numEntries;
			}

			/**
			 * Returns the maximum number of distinct keys that can be inserted
			 * into the hash table (including no success area).
			 */
			uint64_t getMaxNumKeys() const {
				return numEntries + maxNumNoSuccess;
			}

			/**
			 * Returns the maximal number of entries that could be inserted.
			 */
			uint64_t getMaxCapacity() const {

				uint64_t usableSize = size;
				usableSize -= numStreams * keyBufferSize;
				usableSize -= 2 * sizeof(uint64_t);
				usableSize -= minNoSuccessAreaSize;
				usableSize -= 256 * 10;                      // padding

				// maximal number of key-value-pairs that fits in remaining memory
				uint64_t maxNumEntries = usableSize / sizeof(KeyValuePair<intsPerKey>);

				return maxNumEntries;
			}

			/**
			 * Compresses the entries of the hash table such that non-zero entries
			 * are stored subsequently after each other.
			 * @return The number of non-zero entries in the hash table.
			 */
			uint64_t compressEntries() const {
				const uint64_t numEntries = this->getNumEntries();
				if(numEntries>0) {
					return _compress<intsPerKey>(table, numEntries, tmp);
				} else {
					return 0;
				}
			}

		};

	}
}

#endif /* COUNTINGHASHTABLE_HPP_ */
