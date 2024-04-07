#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>

#include "utils.h"


#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>

class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t){};
    YOLODetector(const std::string &modelPath,
                 const bool &isGPU,
                 const cv::Size &inputSize);

    std::vector<Detection> detect(cv::Mat &image, const float &confThreshold, const float &iouThreshold);

    struct Parameters
    {
        struct OutputTensor
        {
            std::size_t width;
            std::size_t height;
            std::size_t depth;
        };

        enum class MODEL_VER
        {
            YOLO2,
            YOLO3,
            YOLO5,
            YOLO7,
            SQUEEZENET
        } model_ver;

        std::size_t classes;
        std::size_t input_width;
        std::size_t input_height;
        std::vector<OutputTensor> output_tensors;
        std::vector<double> anchors;
        double confidence_threshold;
        double iou_threshold;
    };

    struct Box
    {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::int32_t width = 0;
        std::int32_t height = 0;
        double confidence = 0.0;
        std::int32_t class_index = 0;

        static auto overlap(double x1, double w1, double x2, double w2)
        {
            auto l1 = x1 - w1 / 2;
            auto l2 = x2 - w2 / 2;
            auto left = l1 > l2 ? l1 : l2;
            auto r1 = x1 + w1 / 2;
            auto r2 = x2 + w2 / 2;
            auto right = r1 < r2 ? r1 : r2;
            return right - left;
        }

        static auto box_intersection(const Box &a, const Box &b)
        {
            auto w = overlap(a.x, a.width, b.x, b.width);
            auto h = overlap(a.y, a.height, b.y, b.height);
            return w <= 0 || h <= 0 ? 0.0 : w * h;
        }

        static auto box_union(const Box &a, const Box &b)
        {
            auto i = box_intersection(a, b);
            auto u = static_cast<double>(a.width) * a.height + static_cast<double>(b.width) * b.height - i;
            return u;
        }

        static auto box_iou(const Box &a, const Box &b)
        {
            return box_intersection(a, b) / box_union(a, b);
        }
    };

    static auto GetBoxes(const std::vector<std::vector<float>> &scores, const Parameters &parameters)
    {
        std::vector<Box> dst_tmp;
        Ort::AllocatorWithDefaultOptions allocator;

        for (std::size_t t = 0; t < scores.size(); ++t)
        {
            const std::size_t OUTPUT_HEIGHT = parameters.output_tensors[t].height;
            const std::size_t OUTPUT_WIDTH = parameters.output_tensors[t].width;
            const std::size_t OUTPUT_DEPTH = parameters.output_tensors[t].depth;
            const std::size_t BOUNDING_BOXES = OUTPUT_DEPTH / (5 + parameters.classes);
            const std::uint32_t STRIDE = 1 << (3 + t);
            auto cur_tensor = scores[t].data();
            auto anchors = parameters.model_ver == Parameters::MODEL_VER::YOLO2 ? &parameters.anchors[0] : parameters.model_ver == Parameters::MODEL_VER::YOLO3 ? &parameters.anchors[2 * BOUNDING_BOXES * (scores.size() - t - 1)]
                                                                                                                                                                : &parameters.anchors[2 * BOUNDING_BOXES * t];

            for (std::size_t y = 0; y < OUTPUT_HEIGHT; ++y)
            {
                for (std::size_t x = 0; x < OUTPUT_WIDTH; ++x)
                {
                    for (std::size_t b = 0; b < BOUNDING_BOXES; ++b)
                    {
                        auto cur_scores = cur_tensor + y * OUTPUT_WIDTH * OUTPUT_DEPTH + x * OUTPUT_DEPTH + b * (parameters.classes + 5);

                        // Coordinates
                        double tx = *cur_scores++;
                        double ty = *cur_scores++;
                        double tw = *cur_scores++;
                        double th = *cur_scores++;

                        // Verify
                        double objectness = Sigmoid(*cur_scores++);

                        // Classes
                        std::vector<double> classes(cur_scores, cur_scores + parameters.classes);

                        classes = parameters.model_ver == Parameters::MODEL_VER::YOLO2 ? SoftMax(classes) : Logistic(classes);
                        auto max_probability = std::max_element(classes.begin(), classes.end());
                        auto confidence = *max_probability * objectness;
                        if (confidence <= parameters.confidence_threshold)
                        {
                            continue;
                        }

                        std::int32_t left, top, right, bottom;
                        if (parameters.model_ver == Parameters::MODEL_VER::YOLO2)
                        {
                            double box_center_x = (x + Sigmoid(tx)) * 32;
                            double box_center_y = (y + Sigmoid(ty)) * 32;
                            double box_width = std::exp(tw) * anchors[static_cast<std::size_t>(b) * 2] * 32;
                            double box_height = std::exp(th) * anchors[static_cast<std::size_t>(b) * 2 + 1] * 32;
                            left = (std::max)(0, static_cast<int>(box_center_x - box_width / 2.0));
                            top = (std::max)(0, static_cast<int>(box_center_y - box_height / 2.0));
                            right = (std::min)(static_cast<std::int32_t>(parameters.input_width) - 1, static_cast<int>(box_center_x + box_width / 2.0));
                            bottom = (std::min)(static_cast<std::int32_t>(parameters.input_height) - 1, static_cast<int>(box_center_y + box_height / 2.0));
                        }
                        else if (parameters.model_ver == Parameters::MODEL_VER::YOLO3)
                        {
                            double box_center_x = (x + Sigmoid(tx)) / OUTPUT_WIDTH;
                            double box_center_y = (y + Sigmoid(ty)) / OUTPUT_HEIGHT;
                            double box_width = std::exp(tw) * anchors[static_cast<std::size_t>(b) * 2] / parameters.input_width;
                            double box_height = std::exp(th) * anchors[static_cast<std::size_t>(b) * 2 + 1] / parameters.input_height;
                            left = (std::max)(0, static_cast<int>((box_center_x - box_width / 2.0) * parameters.input_width));
                            top = (std::max)(0, static_cast<int>((box_center_y - box_height / 2.0) * parameters.input_height));
                            right = (std::min)(static_cast<std::int32_t>(parameters.input_width) - 1, static_cast<int>((box_center_x + box_width / 2.0) * parameters.input_width));
                            bottom = (std::min)(static_cast<std::int32_t>(parameters.input_height) - 1, static_cast<int>((box_center_y + box_height / 2.0) * parameters.input_height));
                        }
                        else
                        {
                            double box_center_x = (x + Sigmoid(tx) * 2.0 - 0.5f) * STRIDE;
                            double box_center_y = (y + Sigmoid(ty) * 2.0 - 0.5f) * STRIDE;
                            double box_width = std::pow((Sigmoid(tw) * 2.0), 2) * anchors[static_cast<std::size_t>(b) * 2] * STRIDE;
                            double box_height = std::pow((Sigmoid(th) * 2.0), 2) * anchors[static_cast<std::size_t>(b) * 2 + 1] * STRIDE;
                            left = (std::max)(0, static_cast<int>(box_center_x - box_width / 2.0));
                            top = (std::max)(0, static_cast<int>(box_center_y - box_height / 2.0));
                            right = (std::min)(static_cast<std::int32_t>(parameters.input_width) - 1, static_cast<int>(box_center_x + box_width / 2.0));
                            bottom = (std::min)(static_cast<std::int32_t>(parameters.input_height) - 1, static_cast<int>(box_center_y + box_height / 2.0));
                        }

                        Box yolo_box;
                        yolo_box.x = left;
                        yolo_box.y = top;
                        yolo_box.width = right - left;
                        yolo_box.height = bottom - top;
                        yolo_box.confidence = confidence;
                        yolo_box.class_index = std::distance(classes.begin(), max_probability);
                        dst_tmp.push_back(yolo_box);
                    }
                }
            }
        }

        std::stable_sort(dst_tmp.begin(), dst_tmp.end(), [](auto &lhs, auto &rhs)
                         { return lhs.confidence > rhs.confidence; });

        std::vector<Box> dst;
        for (std::size_t i = 0; i < dst_tmp.size(); ++i)
        {
            if (dst_tmp[i].class_index == -1)
            {
                continue;
            }
            dst.push_back(dst_tmp[i]);

            // Zeroize others
            for (auto j = i + 1; j < dst_tmp.size(); ++j)
            {
                if (Box::box_iou(dst_tmp[i], dst_tmp[j]) > parameters.iou_threshold)
                {
                    dst_tmp[j].class_index = -1; // for Wga
                }
            }
        }

        return dst;
    }

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    Parameters parameters;

    void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size &resizedImageShape,
                                          const cv::Size &originalImageShape,
                                          std::vector<Ort::Value> &outputTensors,
                                          const float &confThreshold, const float &iouThreshold);

    static void getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
                                 float &bestConf, int &bestClassId);

    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

    // Inputs
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    // Outputs
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    /**
     * @brief Calculates the sigmoid function value for a given input.
     *
     * @tparam T The data type of the input value.
     * @param val The input value.
     * @return The sigmoid function value.
     */
    template <typename T>
    static double Sigmoid(T val)
    {
        if (val > 0)
        {
            return 1.0 / (1.0 + std::exp(-val));
        }
        else
        {
            double exp_val = std::exp(val);
            return exp_val / (1.0 + exp_val);
        }
    }

    /**
     * @brief Computes the SoftMax function for a vector of classes.
     *
     * @tparam T The type of the classes.
     * @param classes Pointer to the array of classes.
     * @param number_of_classes The number of classes in the array.
     * @return A vector containing the SoftMax values for each class.
     */
    template <typename T>
    static std::vector<T> SoftMax(const T *classes, std::size_t number_of_classes)
    {
        auto max = *std::max_element(classes, classes + number_of_classes);
        double sum = std::accumulate(classes, classes + number_of_classes, 0.0, [max](auto current_sum, auto current_element)
                                     { return current_sum + std::exp(current_element - max); });
        double term = std::log(sum);

        std::vector<T> result;
        result.reserve(number_of_classes);
        std::transform(classes, classes + number_of_classes, std::back_inserter(result), [max, term](auto current_element)
                       { return static_cast<T>(std::exp(current_element - max - term)); });
        return result;
    }

    /**
     * @brief Applies the SoftMax function to a vector of classes.
     *
     * @tparam T The data type of the vector elements.
     * @param classes The vector of classes.
     * @return The vector after applying the SoftMax function.
     */
    template <typename T>
    static std::vector<T> SoftMax(const std::vector<T> &classes)
    {
        return SoftMax(classes.data(), classes.size());
    }

    /**
     * @brief Applies the Logistic (Sigmoid) function to a vector of classes.
     *
     * @tparam T The data type of the vector elements.
     * @param classes The vector of classes.
     * @return The vector after applying the Logistic function.
     */
    template <typename T>
    static std::vector<T> Logistic(const std::vector<T> &classes)
    {
        std::vector<T> result;
        std::transform(classes.begin(), classes.end(), std::back_inserter(result), [](auto current_element)
                       { return Sigmoid(current_element); });
        return result;
    }
};
