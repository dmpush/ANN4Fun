#ifndef __TENSOR_TO_JPEG__
#define __TENSOR_TO_JPEG__

#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>
#include <algorithm>
#include <DataHolder.hpp>
#include <iostream>

namespace tensormath {
template<typename T>
void toJPEG(Tensor<T> tensor, std::string fname) {
    int width{0};
    int height{0};
    int bytes_per_pixel{3};   // 1
    auto color_space{JCS_RGB}; // JCS_GRAYSCALE

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];
    FILE *outfile = fopen( fname.c_str(), "wb" );
    if(!outfile)
	throw std::runtime_error("Не могу создать файл JPG");
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);


    if(tensor->dim()==2) {
	auto shape=tensor->dims();
	width = shape[0];
	height = shape[1];
	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = bytes_per_pixel;
	cinfo.in_color_space = color_space;
	jpeg_set_defaults( &cinfo );
	jpeg_start_compress( &cinfo, TRUE );
	uint8_t *buff=new uint8_t[width*height*3];
	
	auto m1=tensor->raw(0), m2=tensor->raw(0);
	for (size_t ind=0; ind<tensor->size(); ind++) {
	    m1=std::min(m1, tensor->raw(ind));
	    m2=std::max(m2, tensor->raw(ind));
	};
//	std::cout<<m1<<", "<<m2<<std::endl;
        size_t ind=0;
        for(size_t j=0; j<height; j++) 
	    for(size_t i=0; i<width; i++) {
		auto v=(tensor->val({i,j}) - m1)/(m2-m1);
		uint8_t c=static_cast<uint8_t>(v*T(255.0));
                buff[ind++]=c;
                buff[ind++]=c;
                buff[ind++]=c;
	    };
	while( cinfo.next_scanline < cinfo.image_height )
	{
		row_pointer[0] = &buff[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines( &cinfo, row_pointer, 1 );
	}
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	delete []buff;
    }
    else {
	throw std::runtime_error("Такой тип тензора не поддерживатеся");
    };
    fclose( outfile );
};

}; //namespace;
#endif