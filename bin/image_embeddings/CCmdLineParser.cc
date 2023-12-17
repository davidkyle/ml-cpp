/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <torch/csrc/api/include/torch/version.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace torch {


bool CCmdLineParser::parse(int argc,
                      const char* const* argv,
                      std::string& imageFile,
                      std::string& imagePreprocessor,
                      std::string& imageEmbeddingModel) {  
    try {
        boost::program_options::options_description desc("Image Embeddings prototype");
        // clang-format off
        desc.add_options()      
            ("imageFile", boost::program_options::value<std::string>(),
                        "Image to read (PNG/JPEG)")        
            ("preprocessor", boost::program_options::value<std::string>(),
                        "Image preprocessor")      
            ("embeddingModel", boost::program_options::value<std::string>(),
                        "Image embedding mode;")                             
        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv)
                .options(desc)
                .run();
        boost::program_options::store(parsed, vm);


        if (vm.count("preprocessor") > 0) {
            imagePreprocessor = vm["preprocessor"].as<std::string>();
        }
        if (vm.count("imageFile") > 0) {
            imageFile = vm["imageFile"].as<std::string>();
        }
        if (vm.count("embeddingModel") > 0) {
            imageEmbeddingModel = vm["embeddingModel"].as<std::string>();
        }        
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
