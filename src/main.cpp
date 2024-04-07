#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ostream>

#include "cmdline.h"
#include "detector.h"
#include "utils.h"

#define DEBUG

/* void Delay(int time) // time*1000为秒数
{
    clock_t now = clock();

    while (clock() - now < time)
        ;
} */

int main(int argc, char *argv[]) {
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", false, "models/yolov5m.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", false, "images/bus.jpg");
    cmd.add<std::string>("v4l2", 'v', "video dev node to be detected.", false);
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "models/coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");
    cmd.add("show", '\0', "Visualize single image detection");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    bool visualize = cmd.exist("show");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    std::string imagePath = cmd.get<std::string>("image");
    const std::string videoPath = cmd.get<std::string>("v4l2");
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty()) {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    if (imagePath.empty() && videoPath.empty()) {
        std::cerr << "At least give one source! jpg or /dev/videox" << std::endl;
        return -1;
    }
    else if (!videoPath.empty())
    {
        imagePath = "";
    }
    

    YOLODetector detector{nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // Single image
    if (!imagePath.empty()) {
        image = cv::imread(imagePath);
        // detector.detect(image, confThreshold, iouThreshold);
        result = detector.detect(image, confThreshold, iouThreshold);
        // Print boxes
        std::vector<utils::Box> boxes = utils::getBoxes(result, classNames);

        for (std::vector<utils::Box>::size_type i = 0; i < boxes.size(); i++) {
            auto &boxe = boxes[i];

            std::map<std::string, double> box;
            box["x"] = boxe.x;
            box["y"] = boxe.y;
            box["width"] = boxe.width;
            box["height"] = boxe.height;
            box["confidence"] = boxe.confidence;
            box["class_index"] = boxe.class_index;

            #ifdef DEBUG
            std::cout << "Box " << (i + 1) << ":" << std::endl;
            std::cout << "\tx = " << box["x"] << std::endl;
            std::cout << "\ty = " << box["y"] << std::endl;
            std::cout << "\twidth = " << box["width"] << std::endl;
            std::cout << "\theight = " << box["height"] << std::endl;
            std::cout << "\tconfidence = " << box["confidence"] << std::endl;
            std::cout << "\tclass_index = " << box["class_index"] << std::endl;
            std::cout << "\tobject = " << box["object"] << std::endl;
            #endif
        }

        // Show image with boxes
        if (visualize) {
            utils::visualizeDetection(image, result, classNames);
            cv::imshow("result", image);
            // cv::imwrite("result.jpg", image);
            cv::waitKey(0);
        }
        // Video
    } else if (!videoPath.empty()) {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Could not read frame." << std::endl;
                continue;
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            result = detector.detect(frame, confThreshold, iouThreshold);
            utils::visualizeDetection(frame, result, classNames);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Execution time: " << duration.count() << " ms." << std::endl;

            cv::imshow("result", frame);
            // Exit on ESC
            char k = cv::waitKey(10);
            if( k == 27 ) break;
        }
    }
    return 0;
}

