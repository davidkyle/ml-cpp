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

#include <core/CLogger.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include "CCmdLineParser.h"

#include <torch/script.h>
#include <vision.h>
#include <io/image/image.h>

#include <iostream>
#include <string>

namespace {
using TRapidJsonLineWriter = ml::core::CRapidJsonLineWriter<rapidjson::StringBuffer>;
}


template<typename T>
void writeTensor(const ::torch::TensorAccessor<T, 1UL>& accessor,
                    TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        jsonWriter.Double(static_cast<double>(accessor[i]));
    }
    jsonWriter.EndArray();
}

//! Write an N dimensional tensor for N > 1.
template<typename T, std::size_t N_DIMS>
void writeTensor(const ::torch::TensorAccessor<T, N_DIMS>& accessor,
                    TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        writeTensor(accessor[i], jsonWriter);
    }
    jsonWriter.EndArray();
}

template<std::size_t N>
void writePrediction(const torch::Tensor& prediction,
                     TRapidJsonLineWriter& jsonWriter) {

    if (prediction.dtype() == torch::kFloat32) {
        auto accessor = prediction.accessor<float, N>();
        writeTensor(accessor, jsonWriter);        
    } else if (prediction.dtype() == torch::kFloat64) {
        auto accessor = prediction.accessor<double, N>();
        writeTensor(accessor, jsonWriter);     
    } else if (prediction.dtype() == torch::kUInt8) {
        auto accessor = prediction.accessor<unsigned char, N>();
        writeTensor(accessor, jsonWriter);          
    } else {
        LOG_ERROR( << "Cannot process result tensor of type [" << prediction.dtype() << "]");
    }
}

void writeImageTensor(const torch::Tensor& imageTensor,
                     TRapidJsonLineWriter& jsonWriter) {
        auto sizes = imageTensor.sizes();

        switch (sizes.size()) {
        case 4:
            writePrediction<4>(imageTensor, jsonWriter);
            break;            
        case 3:
            writePrediction<3>(imageTensor, jsonWriter);
            break;
        case 2:
            writePrediction<2>(imageTensor, jsonWriter);
            break;
        default: {
            LOG_ERROR( << "Cannot convert results tensor of size [" << sizes << "]");            
            break;
        }
        }
}

int main(int argc, char** argv) {
    // command line options
    std::string imageFile;
    std::string imagePreprocessor;
    std::string imageEmbeddingModel;

    if (ml::torch::CCmdLineParser::parse(
            argc, argv, imageFile, imagePreprocessor, imageEmbeddingModel) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{std::cout};
    ml::core::CRapidJsonConcurrentLineWriter jsonWriter{wrappedOutputStream};

    LOG_INFO(<< "Reading image " << imageFile);

    torch::Tensor image = vision::image::read_file(imageFile);
    torch::Tensor decoded = vision::image::decode_image(image);
       

    torch::jit::script::Module preprocessor;
    torch::jit::script::Module embeddingModel;
    try {
        preprocessor = torch::jit::load(imagePreprocessor);
        preprocessor.eval();
        LOG_DEBUG(<< "Image preprocessor loaded " << imagePreprocessor);

        embeddingModel = torch::jit::load(imageEmbeddingModel);
        embeddingModel.eval();
        LOG_DEBUG(<< "Image embedding model loaded" << imageEmbeddingModel);
    } catch (const c10::Error& e) {
        LOG_FATAL(<< "Error loading the model: " << imagePreprocessor << " " << e.msg());
        return EXIT_FAILURE;
    } catch (std::runtime_error& e) {
        LOG_FATAL(<< "Error loading the model: " << imagePreprocessor << " " << e.what());
        return EXIT_FAILURE;
    }

    {
        torch::InferenceMode inferenceModeGuard;
        std::vector<torch::jit::IValue> imageValues;
        imageValues.emplace_back(decoded);
        // inputs
        auto prepared = preprocessor.forward(imageValues).toTensor();
        auto corrected = torch::unsqueeze(prepared, 0);

        std::vector<torch::jit::IValue> preProcessedValues;
        preProcessedValues.emplace_back(corrected);
        auto embedding = embeddingModel.forward(preProcessedValues);

        
        if (embedding.isTuple()) {
            LOG_DEBUG( << "got tuple");
            writeImageTensor(embedding.toTuple()->elements()[0].toTensor(), jsonWriter);
        } else if (embedding.isTensor()) {
            LOG_DEBUG( << "tensor");
            writeImageTensor(embedding.toTensor(), jsonWriter);
        } else {
            LOG_WARN( << "unknown result type" << embedding.toString());
        }        

    }

    return EXIT_SUCCESS;
}
