#include "detector.h"

YOLODetector::YOLODetector(const std::string &modelPath,
                           const bool &isGPU = true,
                           const cv::Size &inputSize = cv::Size(640, 640))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    // for (auto shape : inputTensorShape)
    std::cout << "Input shape0: " << inputTensorShape[0] << std::endl;
    std::cout << "Input shape1: " << inputTensorShape[1] << std::endl;
    std::cout << "Input width: " << inputTensorShape[2] << std::endl;
    std::cout << "Input height: " << inputTensorShape[3] << std::endl;
    std::cout << "Input shape4: " << inputTensorShape[4] << std::endl;

    // Получение информации о выходных тензорах модели
    size_t numOutputTensors = session.GetOutputCount();
    // inputNames.push_back(session.GetInputName(0, allocator));
    // outputNames.push_back(session.GetOutputName(0, allocator));
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    std::cout << "Input name: " << inputNames[0] << std::endl;

    for (size_t i = 0; i < numOutputTensors; i++)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        // outputNames.push_back(session.GetOutputName(i, allocator));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    std::double_t confidence_threshold = 0.5;
    std::double_t iou_threshold = 0.45;

    double anchorsArray[] = {1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875};
    const double *anchors = &anchorsArray[0];
    std::uint32_t anchlen = sizeof(anchorsArray) / sizeof(double);

    // !Необходимо самостоятельно подготовить анкеры!
    // !Первые шесть значений делить на 8, вторые на 16, третьи на 32...
    parameters.anchors.assign(anchors, anchors + anchlen);

    parameters.input_width = inputTensorShape[2];
    parameters.input_height = inputTensorShape[3];
    parameters.output_tensors.resize(numOutputTensors);
    parameters.model_ver = Parameters::MODEL_VER::YOLO5;

    parameters.confidence_threshold = confidence_threshold;
    parameters.iou_threshold = iou_threshold;

    const std::vector<std::string> classes = utils::getClasses("/home/ap/work/satel/mpp/onnxruntime/yolov5-onnxruntime/models/coco.names");
    parameters.classes = classes.size();

    // Получение параметров для каждого выходного тензора
    for (size_t i = 0; i < numOutputTensors; i++)
    {

        Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> outputDims = tensorInfo.GetShape();

        std::cout << "\nTotal output tensors: " << numOutputTensors << std::endl;
        std::cout << "Output tensor " << i << std::endl;
        std::cout << "Output name: " << outputNames[i] << std::endl;
        std::cout << "\twidth: " << outputDims[2] << std::endl;
        std::cout << "\theight: " << outputDims[3] << std::endl;
        std::cout << "\tdepth: " << outputDims[1] << std::endl;
        parameters.output_tensors[i].width = outputDims[2];
        parameters.output_tensors[i].height = outputDims[3];
        parameters.output_tensors[i].depth = outputDims[1];
    }
    std::cout << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
                                    float &bestConf, int &bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}

void YOLODetector::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLODetector::postprocessing(const cv::Size &resizedImageShape,
                                                    const cv::Size &originalImageShape,
                                                    std::vector<Ort::Value> &outputTensors,
                                                    const float &confThreshold, const float &iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    // std::cout << "Output Shape: " << shape << std::endl;
    // Yolo v5 simplified
    // Output Shape: 1
    // Output Shape: 255
    // Output Shape: 80
    // Output Shape: 80
    // parameters.output_tensors[0].width:80
    // parameters.output_tensors[0].height:80
    // parameters.output_tensors[0].depth:255

    // parameters.output_tensors[1].width:40
    // parameters.output_tensors[1].height:40
    // parameters.output_tensors[1].depth:255

    // parameters.output_tensors[2].width:20
    // parameters.output_tensors[2].height:20
    // parameters.output_tensors[2].depth:255

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat &image, const float &confThreshold = 0.4,
                                            const float &iouThreshold = 0.45)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    std::list<std::map<std::string, double>> box_container;

    for (const auto &tensor : outputTensors)
    {
        std::cout << "Output Tensor:" << std::endl;
        const float *data = tensor.GetTensorData<float>();
        std::cout << "First 5 values: ";
        for (size_t i = 0; i < 5; ++i)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    // auto *rawOutput = outputTensors[0].GetTensorData<float>();
    // size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    // std::vector<float> scores(rawOutput, rawOutput + count);

    std::vector<std::vector<float>> scores;
    for (const auto &tensor : outputTensors)
    {
        auto *rawOutput = tensor.GetTensorData<float>();
        size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> tensorValues(rawOutput, rawOutput + count);
        scores.emplace_back(std::move(tensorValues));
    }

    // ! Эталонные
    // 0.206396
    // 0.204041
    // 0.00235387
    // -0.298017

    std::cout << "Первые 5 значений из scores[0]:" << std::endl;
    for (std::size_t i = 0; i < 5; ++i)
    {
        std::cout << scores[0][i] << std::endl;
    }

    std::vector<YOLODetector::Box> boxes = GetBoxes(scores, parameters);

    // #ifdef DEBUG
    //! Возвращает 0. Нужно разобраться!
    std::cout << "Всего boxes:" << boxes.size() << std::endl;
    // #endif

    for (std::vector<Box>::size_type i = 0; i < boxes.size(); i++)
    {
        auto &boxe = boxes[i];

        std::map<std::string, double> box;
        box["x"] = boxe.x;
        box["y"] = boxe.y;
        box["width"] = boxe.width;
        box["height"] = boxe.height;
        box["confidence"] = boxe.confidence;
        box["class_index"] = boxe.class_index;

        box_container.push_back(box);
        // #ifdef DEBUG
        std::cout << "Box " << (i + 1) << ":" << std::endl;
        std::cout << "\tx = " << box["x"] << std::endl;
        std::cout << "\ty = " << box["y"] << std::endl;
        std::cout << "\twidth = " << box["width"] << std::endl;
        std::cout << "\theight = " << box["height"] << std::endl;
        std::cout << "\tconfidence = " << box["confidence"] << std::endl;
        std::cout << "\tclass_index = " << box["class_index"] << std::endl;
        // #endif
    }

    delete[] blob;

    return result;
}
