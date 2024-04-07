#pragma once
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

namespace utils
{
    struct Box
    {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::int32_t width = 0;
        std::int32_t height = 0;
        double confidence = 0.0;
        std::int32_t class_index = 0;
        std::string object = "";
    };

    size_t vectorProduct(const std::vector<int64_t> &vector);
    std::wstring charToWstring(const char *str);
    std::vector<std::string> loadNames(const std::string &path);
    void visualizeDetection(cv::Mat &image, std::vector<Detection> &detections,
                            const std::vector<std::string> &classNames);
    std::vector<utils::Box> getBoxes(std::vector<Detection> &detections,
                                     const std::vector<std::string> &classNames);
    void letterbox(const cv::Mat &image, cv::Mat &outImage,
                   const cv::Size &newShape,
                   const cv::Scalar &color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(const cv::Size &imageShape, cv::Rect &box, const cv::Size &imageOriginalShape);

    template <typename T>
    T clip(const T &n, const T &lower, const T &upper);
    // Считает количество не пустых строк в файле с классами
    std::vector<std::string> getClasses(const std::string &filename);

}
