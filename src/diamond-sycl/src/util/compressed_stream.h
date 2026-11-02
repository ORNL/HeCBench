/****
DIAMOND protein aligner
Copyright (C) 2013-2017 Benjamin Buchfink <buchfink@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
****/

#ifndef COMPRESSED_STREAM_H_
#define COMPRESSED_STREAM_H_

#include <string>
#include <memory>
#include <zlib.h>
#include "binary_file.h"
#include "util.h"

using std::string;
using std::unique_ptr;

struct Stream_read_exception : public std::runtime_error
{
	Stream_read_exception(size_t line_count, const char *msg) :
		runtime_error(string("Error reading input stream at line ") + to_string(line_count) + ": " + msg)
	{}
};

struct Compressed_istream : public Input_stream
{
	Compressed_istream(const string &file_name);
	virtual ~Compressed_istream();
	virtual size_t read_bytes(char *ptr, size_t count);
	virtual void close();
	static Input_stream* auto_detect(const string &file_name);
private:
	z_stream strm;
	static const size_t chunk_size = 1llu << 20;
	unique_ptr<char[]> in, out;
	size_t read_, total_;
	bool eos_;
	bool z_open_;
};

struct Compressed_ostream : public Output_stream
{
	Compressed_ostream(const string &file_name);
	virtual ~Compressed_ostream();
	virtual void write_raw(const char *ptr, size_t count);
	virtual void close();
private:
	void deflate_loop(const char *ptr, size_t count, int code);
	static const size_t chunk_size = 1llu << 20;
	z_stream strm;
	unique_ptr<char[]> out;
	bool z_open_;
};

#endif