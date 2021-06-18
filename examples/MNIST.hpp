#ifndef __MNIST_HPP__
#define __MNIST_HPP__

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
#include <random>
#include <cassert>
#include <DataHolder.hpp>

template <class T>
void endswap(T *objp)
{
	unsigned char *memp = reinterpret_cast<unsigned char *>(objp);
	std::reverse(memp, memp + sizeof(T));
}

template <typename T>
class MNIST
{
	std::random_device seed_;
	std::mt19937 rdev_;
public:
	struct Image {
	    T data[28 * 28];
	    int label;
	    using sPtr=std::shared_ptr<Image>;
	};
    using sPtr = std::shared_ptr<MNIST>;
private:
	class MNIST_set
	{
		MNIST *owner_;
	public:

		std::vector<std::shared_ptr<Image>> images_;

		MNIST_set(MNIST *owner, std::string folder = "../", std::string prefix = "train"): owner_(owner)
		{
			std::ifstream file(folder + prefix + "-images-idx3-ubyte");
			if (!file.is_open())
				std::cout << "file not found!" << std::endl;
			int32_t magic, numItems, picWidth, picHeight;
			file.read((char *)&magic, sizeof(int32_t));
			endswap(&magic);
			file.read((char *)&numItems, sizeof(int32_t));
			endswap(&numItems);
			file.read((char *)&picWidth, sizeof(int32_t));
			endswap(&picWidth);
			file.read((char *)&picHeight, sizeof(int32_t));
			endswap(&picHeight);
			assert(magic==2051);
			assert(numItems==60000 || numItems==10000);
			assert(picWidth==28);
			assert(picHeight==28);
			images_.resize(numItems);
			uint8_t img[28 * 28];
			for (size_t i = 0; i < numItems; i++){
				file.read((char *)&img, 28 * 28);
				////	    endswap(img);
				images_[i] = std::make_shared<Image>();
				for (size_t k = 0; k < 28 * 28; k++){
					images_[i]->data[k] = static_cast<T>(static_cast<float>(img[k]) / 255.0f);
				}
			};
			file.close();
			file = std::ifstream(folder + prefix + "-labels-idx1-ubyte");
			if (!file.is_open())
				std::cout << "file not found!" << std::endl;
			file.read((char *)&magic, sizeof(int32_t));
			endswap(&magic);
			assert(magic==2049);
			file.read((char *)&numItems, sizeof(int32_t));
			endswap(&numItems);
			assert(numItems==60000 || numItems==10000);
			for (size_t i = 0; i < numItems; i++) {
				char label;
				file.read((char *)&label, 1);
				images_[i]->label = static_cast<int>(label);
			};
		};
		typename Image::sPtr	getRandomSample() {
			std::uniform_int_distribution<int> dist(0, images_.size()-1);
			return images_[dist(owner_->rdev_)];
		};
		size_t numSamples() {
			return images_.size();
		}
		auto begin() { return images_.begin(); };
		auto end() { return images_.end(); };
	}; // MNIST_set
	std::shared_ptr<MNIST_set> train;
	std::shared_ptr<MNIST_set> test;
public:

	MNIST(std::string path) : seed_{}, rdev_{seed_()} {
	    train = std::make_shared<MNIST_set>(this, path, "train");
	    test  = std::make_shared<MNIST_set>(this, path, "t10k");
	};

	std::shared_ptr<MNIST_set> getTrainSet() { return train; };
	std::shared_ptr<MNIST_set> getTestSet() { return test; };

};

#endif
