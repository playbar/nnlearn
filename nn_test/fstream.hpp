#ifndef FBC_NN_FSTREAM_HPP_
#define FBC_NN_FSTREAM_HPP_

#include <fstream>

template<typename _Tp>
static int write_file(const _Tp* data, size_t length, const char* file_name)
{
	std::ofstream outfile;
	outfile.open(file_name, std::ios::out | std::ios::binary);
	if (!outfile.is_open()) {
		fprintf(stderr, "failed to open file: %s\n", file_name);
		return -1;
	}

	outfile.write((char*)&length, sizeof(size_t));
	for (int i = 0; i < length; i++)
		outfile.write((char*)&data[i], sizeof(_Tp));

	outfile.close();
	return 0;
}

template<typename _Tp>
static int read_file(_Tp* data, size_t length, const char* file_name)
{
	std::ifstream infile;
	infile.open(file_name, std::ios::in | std::ios::binary);
	if (!infile.is_open()) {
		fprintf(stderr, "failed to open file: %s\n", file_name);
		return -1;
	}

	size_t length_ = 0;
	infile.read((char*)&length_, sizeof(size_t));
	if (length != length_) {
		fprintf(stderr, "their length is mismatch: required length: %d, actual length: %d\n", length, length_);
		return -1;
	}

	for (int i = 0; i < length; i++)
		infile.read((char*)&data[i], sizeof(_Tp));

	infile.close();
	return 0;
}

#endif //FBC_NN_FSTREAM_HPP_
