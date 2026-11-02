/*
 * FastReader.cpp
 *
 *  Created on: 20.05.2015
 *      Author: marius
 */

#define OLD_ZIP_DECOMPR false

#include "../../include/gerbil/FastReader.h"
#include <errno.h>

gerbil::FastReader::FastReader(const uint32_t &frBlocksNumber,
		std::string pPath, uint8 &_readerParserThreadsNumber) :
		_path(pPath), _processThreads(NULL), _fileType(ft_unknown), _totalBlocksRead(
				0), _totalReadBytes(0), _fastFilesNumber(0), _threadsNumber(
				_readerParserThreadsNumber), _fastFileNr(0) {

	// check if path exists
	if (!bfs::exists(_path)) {
		std::cerr << "ERROR: path [" << _path.string() << "] does not exists\n";
		exit(1);
	}

	std::queue<bfs::path> fastPathQueue;

	// determine file type
	if (bfs::is_directory(_path)) {
		bfs::directory_iterator dirIterEnd;
		for (bfs::directory_iterator dirIter(_path); dirIter != dirIterEnd;
				++dirIter)
			if (!bfs::is_directory(dirIter->status()))
				fastPathQueue.push(dirIter->path());
	} else {
		std::string s = _path.extension().string();
		if (_path.extension().string() == ".txt") {
			std::string line;
			std::ifstream file(_path.c_str());
			while (getline(file, line))
				fastPathQueue.push(line);
		} else
			fastPathQueue.push(_path);
	}

	_fastFilesNumber = fastPathQueue.size();

	// at least one file
	if (!_fastFilesNumber) {
		std::cerr << "ERROR: missing input files" << std::endl;
		exit(1);
	}

	bool someComprFastFiles = false;
	_fastFiles = new FastFile*[_fastFilesNumber];
	FastFile* ff[_fastFilesNumber];
	for (uint_fast32_t i(0); i < _fastFilesNumber; ++i) {
		ff[i] = new FastFile(fastPathQueue.front());
		fastPathQueue.pop();
		if (ff[i]->getCompr() != fc_none)
			someComprFastFiles = true;
	}

	// check that all files have the same (valid) type
	_fileType = ff[0]->getType();
	for (uint_fast32_t i(0); i < _fastFilesNumber; ++i) {
		if (ff[i]->getType() == ft_unknown) {
			std::cerr << "input file type with extension '"
					<< ff[i]->getPath().extension().string()
					<< "' is unsupported\n" << std::endl;
			exit(1);
		} else if (ff[i]->getType() != _fileType) {
			std::cerr << "files are of different types" << std::endl;
			exit(1);
		}
	}




	if(someComprFastFiles) {
		std::vector<uint64_t> fileSizes;

		for (uint64_t i(0); i < _fastFilesNumber; ++i) {
			//printf("%12lu\n", ff[i]->getSize());
			fileSizes.push_back(ff[i]->getSize());
		}

		sort(fileSizes.begin(), fileSizes.end(), std::greater<uint64_t>());

		const uint minNum = 2;
		const uint maxNum = 4;

		uint64 totalSize = 0;
		for (uint64_t i(0); i < _fastFilesNumber; ++i)
			totalSize += ff[i]->getSize();

		uint64 bestScore = UINT64_MAX;
		uint64 bestNum = minNum;

		for(uint num = minNum; num <= maxNum; num++) {
			uint64 optSize = totalSize / num;

			uint64 buckets[num];
			for(uint i = 0; i < num; ++i)
				buckets[i] = 0;

			for (auto it = fileSizes.begin(); it != fileSizes.end(); ++it) {
				uint64 minBucketI = 0;
				for(uint i = 1; i < num; ++i)
					if(buckets[i] < buckets[minBucketI])
						minBucketI = i;
				buckets[minBucketI] += *it;
			}

			uint64 score = 0;

			for(uint i = 0; i < num; ++i)
				score += (buckets[i] - optSize) * (buckets[i] - optSize);

			if(score < bestScore) {
				bestNum = num;
				bestScore = score;
			}
		}

		_threadsNumber = bestNum;
		//std::cout << "bestNum = " << (int) _threadsNumber << std::endl;
	}
	else
		_threadsNumber = 1;
	_readerParserThreadsNumber = _threadsNumber;

	_processThreads = new std::thread*[_threadsNumber];

	_syncSwapQueues = new SyncSwapQueueSPSC<FastBundle>*[_threadsNumber];
	for (uint i(0); i < _threadsNumber; ++i)
		_syncSwapQueues[i] = new SyncSwapQueueSPSC<FastBundle>(
				frBlocksNumber / _threadsNumber);

	std::vector<std::pair<uint64_t, uint64_t>> fastFileOrder;

	for (uint64_t i(0); i < _fastFilesNumber; ++i)
		fastFileOrder.push_back(std::make_pair(ff[i]->getSize(), i));

	sort(fastFileOrder.begin(), fastFileOrder.end(),
			std::greater<std::pair<uint32, uint32>>());

	uint64 nr(0);
	for (std::vector<std::pair<uint64, uint64>>::iterator it = fastFileOrder.begin();
			it != fastFileOrder.end(); ++it)
		_fastFiles[nr++] = ff[it->second];
}
;

// reads a single file
void gerbil::FastReader::readFile(
		const uint tId,
		const FastFile &fastFile,
		SyncSwapQueueSPSC<FastBundle> &syncSwapQueue
#ifdef DEB_MESS_FASTREADER
		, StopWatch* sw
#endif
		) {
	FastBundle* curFastBundle = new FastBundle;
	bool inGB = fastFile.getSize() > ((uint64) 1 << 30);

	if(verbose) {
		printf("Thread[%i]: read file '%s' (%4lu %sB)...\n",
		       tId,
		       fastFile.getPath().filename().c_str(),
		       inGB ? B_TO_GB(fastFile.getSize()) : B_TO_MB(fastFile.getSize()),
		       inGB ? "G" : "M");
	}
	//if (fastFile.getSize() > ((uint64) 1 << 30))
	//	printf("read file '%s' (%4lu GB)...\n",
	//			fastFile.getPath().filename().c_str(), B_TO_GB(fastFile.getSize()));
	//else
	//	printf("read file '%s' (%4lu MB)...\n",
	//			fastFile.getPath().filename().c_str(), B_TO_MB(fastFile.getSize()));

	//open file
	FILE* file;
	gzFile_s* gzipFile;
	BZFILE* bzip2File;
	switch (fastFile.getCompr()) {
#if not OLD_ZIP_DECOMPR
	case fc_gzip:
#endif
	case fc_none:
		file = fopen(fastFile.getPath().c_str(), "rb");
		if (!file) {
			std::cerr << "ERROR: unable to open File ["
					<< fastFile.getPath().string() << "] for read\n";
			exit(1);
		}
		setbuf(file, NULL);
		break;
#if OLD_ZIP_DECOMPR
		case fc_gzip:
		gzipFile = gzopen(fastFile.getPath().c_str(), "rb");
		if (!gzipFile) {
			std::cerr << "ERROR: unable to open File ["
					<< fastFile.getPath().string() << "] for read\n";
			exit(1);
		}
		gzbuffer(gzipFile, 1024 * 1024);
		break;
#endif
	case fc_bz2:
		file = fopen(fastFile.getPath().c_str(), "rb");
		if (!file) {
			std::cerr << "ERROR: unable to open File ["
					<< fastFile.getPath().string() << "] for read\n";
			exit(1);
		}
		setvbuf(file, NULL, _IOFBF, 1024 * 1024);
		int bzerror;
		bzip2File = BZ2_bzReadOpen(&bzerror, file, 0, 0, NULL, 0);
		if (!bzip2File) {
			fclose(file);
			std::cerr << "ERROR: unable to open File(bz2) ["
					<< fastFile.getPath().string() << "] for read\n";
			std::cerr << "error code: " << bzerror << std::endl;
			exit(1);
		}
		break;
	}

	uint32 readSize;
	char* curData = curFastBundle->data;

	if (fastFile.getCompr() == fc_none
#if not OLD_ZIP_DECOMPR
	    || fastFile.getCompr() == fc_gzip
#endif
			) {

		// determine file size
		//fseek(file, 0, SEEK_END);
		//uint64 fileSize(ftell(file));
		//rewind(file);
		uint64 fileSize(fastFile.getSize());
		_totalReadBytes += fileSize;
		// block wise reading
		uint64 filePos(0);
		uint fastBundleCounter = 0;
		while (filePos < fileSize) {
			if (curFastBundle->isFull()) {
				curFastBundle->finalize(fastFile.getCompr());
				IF_MESS_FASTREADER(sw->hold());
				++fastBundleCounter;
				syncSwapQueue.swapPush(curFastBundle);
				IF_MESS_FASTREADER(sw->proceed());
				curData = curFastBundle->data;
			}
			readSize =
					filePos + FAST_BLOCK_SIZE_B < fileSize ?
							FAST_BLOCK_SIZE_B : fileSize - filePos;
			if (fread(curData, 1, readSize, file) != readSize) {
				std::cerr << "ERROR: read block from file\n";
				exit(5);
			}
			curFastBundle->size += readSize;

			curData += readSize;
			filePos += readSize;
			++_totalBlocksRead;
			//if(_totalBlocksRead % (1024 * 1024 * 256/ FAST_BLOCK_SIZE_B) == 0)
			//	printf("\r%4.3f GB", (double)_totalBlocksRead / 1024 * FAST_BLOCK_SIZE_B / 1024 / 1024);
		}
		bool lastIsEmpty = curFastBundle->size == 0;
		curFastBundle->finalize(fastFile.getCompr());
		IF_MESS_FASTREADER(sw->hold());
		syncSwapQueue.swapPush(curFastBundle);
		IF_MESS_FASTREADER(sw->proceed());
		if(!lastIsEmpty && fastFile.getCompr() != fc_none) {
			curFastBundle->size = 0;
			curFastBundle->finalize(fastFile.getCompr());
			syncSwapQueue.swapPush(curFastBundle);
			++fastBundleCounter;
		}
		//std::cout << "fastBundleCounter = " << fastBundleCounter << std::endl;

		//printf("\r%4.3f GB\n", (double)fileSize / 1024 / 1024 / 1024);

		// close file
		fclose(file);
	} else if (fastFile.getCompr() == fc_bz2) {
		uint64 fileSize(0);
		int bzError = BZ_OK;
		while (bzError != BZ_STREAM_END) {
			curFastBundle->size = BZ2_bzRead(&bzError, bzip2File,
					curFastBundle->data, FAST_BUNDLE_DATA_SIZE_B);
			curFastBundle->finalize(fc_none);
			fileSize += curFastBundle->size;
			IF_MESS_FASTREADER(sw->hold());
			syncSwapQueue.swapPush(curFastBundle);
			IF_MESS_FASTREADER(sw->proceed());
			++_totalBlocksRead;
			//if(_totalBlocksRead % (1024 * 1024 * 256/ FAST_BUNDLE_DATA_SIZE_B) == 0)
			//	printf("\r%4.3f GB", (double)_totalBlocksRead / 1024 * FAST_BUNDLE_DATA_SIZE_B / 1024 / 1024);
		}
		//printf("\r%4.3f GB\n", (double)fileSize / 1024 / 1024 / 1024);
		BZ2_bzReadClose(&bzError, bzip2File);
		fclose(file);
	}
#if OLD_ZIP_DECOMPR
	else if (fastFile.getCompr() == fc_gzip) {
		uint64 fileSize(0);
		//FILE* file = fopen("/home/rechner/Desktop/out.fa", "wb");
		while ((curFastBundle->size = gzread(gzipFile, curFastBundle->data,
		FAST_BUNDLE_DATA_SIZE_B))) {
			//fwrite(curFastBundle->data, 1, curFastBundle->size, file);
			curFastBundle->finalize(fc_none);
			fileSize += curFastBundle->size;
			IF_MESS_FASTREADER(sw->hold());
			syncSwapQueue.swapPush(curFastBundle);
			IF_MESS_FASTREADER(sw->proceed());
			++_totalBlocksRead;
			//if(_totalBlocksRead % (1024 * 1024 * 256/ FAST_BUNDLE_DATA_SIZE_B) == 0)
			//	printf("\r%4.3f GB", (double)_totalBlocksRead / 1024 * FAST_BUNDLE_DATA_SIZE_B / 1024 / 1024);
		}
		//printf("\r%4.3f GB\n", (double)fileSize / 1024 / 1024 / 1024);
		gzclose(gzipFile);
		//fclose(file);
	}
#endif

	delete curFastBundle;
}

void gerbil::FastReader::processThread(const size_t tId,
		SyncSwapQueueSPSC<FastBundle> &syncSwapQueue) {
	IF_MESS_FASTREADER(
			StopWatch sw;
			sw.start();
	)
	//read file(s)
	uint64 curFastFileNr;
	while ((curFastFileNr = _fastFileNr++) < _fastFilesNumber) {
		readFile(tId, *(_fastFiles[curFastFileNr]), syncSwapQueue
#ifdef DEB_MESS_FASTREADER
				, &sw
#endif
				);
		delete _fastFiles[curFastFileNr];
	}

	// no further blocks
	syncSwapQueue.finalize();

IF_MESS_FASTREADER(
		sw.stop();
		printf("time reader[%2lu]: %.3f s\n", tId, sw.get_s());
)
}

void gerbil::FastReader::process() {
for (uint i = 0; i < _threadsNumber; ++i)
	_processThreads[i] = new std::thread([this](uint8 tId) {
		processThread(tId, *(_syncSwapQueues[tId]));
	}, i);
}

void gerbil::FastReader::join() {
for (uint i = 0; i < _threadsNumber; ++i) {
	_processThreads[i]->join();
	delete _processThreads[i];
}
IF_DEB(printf("fastReaders are rdy...\n"));
}

void gerbil::FastReader::print() {
printf("total bytes read       : %12lu B\n", _totalReadBytes);
printf("total blocks read      : %12lu\n",
		_totalBlocksRead.load(std::memory_order_relaxed));
}

gerbil::SyncSwapQueueSPSC<gerbil::FastBundle>** gerbil::FastReader::getSyncSwapQueues() {
return _syncSwapQueues;
}

gerbil::TFileType gerbil::FastReader::getFileType() const {
return _fileType;
}
gerbil::FastReader::~FastReader() {
delete[] _processThreads;
delete[] _fastFiles;
for (uint i(0); i < _threadsNumber; ++i)
	delete _syncSwapQueues[i];
delete[] _syncSwapQueues;
}
