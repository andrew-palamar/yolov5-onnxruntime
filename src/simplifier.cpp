#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

using namespace std;
// using namespace Onnx;
// using namespace OrtApi;

Ort::AllocatorWithDefaultOptions allocator;

std::vector<std::string> getInputNames(const OrtApi &api, OrtSession *session)
{
    std::vector<std::string> input_names;
    size_t num_inputs;
    api.SessionGetInputCount(session, &num_inputs);
    for (size_t i = 0; i < num_inputs; i++)
    {
        char *name;
        // char* input_name = mSession->GetInputNameAllocated(i, allocator).get()
        api.SessionGetInputName(session, i, allocator, &name);
        input_names.push_back(std::string(name));
        api.AllocatorFree(allocator, name);
    }
    return input_names;
}

std::vector<std::string> getLastThreeConvOutputs(const OrtApi &api, OrtSession *session)
{
    std::vector<std::string> conv_outputs;
    size_t num_outputs;
    api.SessionGetOutputCount(session, &num_outputs);
    for (size_t i = 0; i < num_outputs; i++)
    {
        char *name;
        api.SessionGetOutputName(session, i, allocator, &name);
        if (std::string(name).find("Conv") != std::string::npos)
        {
            conv_outputs.push_back(std::string(name));
        }
        api.AllocatorFree(allocator, name);
    }
    // Берём только последние три значения
    if (conv_outputs.size() >= 3)
    {
        return {conv_outputs[conv_outputs.size() - 3], conv_outputs[conv_outputs.size() - 2], conv_outputs[conv_outputs.size() - 1]};
    }
    else
    {
        return conv_outputs;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " --input_path <input_path> --output_path <output_path> --batch_size <batch_size> --channels <channels> --height <height> --width <width>" << std::endl;
        return 1;
    }

    std::string input_path, output_path;
    int batch_size, channels, height, width;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--input_path")
        {
            input_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--output_path")
        {
            output_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--batch_size")
        {
            batch_size = std::stoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--channels")
        {
            channels = std::stoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--height")
        {
            height = std::stoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--width")
        {
            width = std::stoi(argv[++i]);
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "");
    Ort::Session session(env, input_path.c_str(), Ort::SessionOptions{nullptr});

    std::vector<std::string> input_names = getInputNames(env.GetApi(), session.GetSessionHandle());
    std::vector<std::string> output_names = getLastThreeConvOutputs(env.GetApi(), session.GetSessionHandle());

    std::vector<const char *> input_c_names(input_names.size());
    std::transform(input_names.begin(), input_names.end(), input_c_names.begin(), [](const std::string &s)
                   { return s.c_str(); });

    std::vector<const char *> output_c_names(output_names.size());
    std::transform(output_names.begin(), output_names.end(), output_c_names.begin(), [](const std::string &s)
                   { return s.c_str(); });

    std::vector<int64_t> input_shape = {batch_size, channels, height, width};
    std::vector<std::vector<int64_t>> output_shapes(output_names.size(), std::vector<int64_t>{batch_size, channels, height, width});

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, nullptr, 0, input_shape.data(), input_shape.size());

    std::vector<Ort::Value> output_tensors(output_names.size());
    for (size_t i = 0; i < output_names.size(); i++)
    {
        output_tensors[i] = Ort::Value::CreateTensor<float>(mem_info, nullptr, 0, output_shapes[i].data(), output_shapes[i].size());
    }

    session.Run(Ort::RunOptions{nullptr}, input_c_names.data(), &input_tensor, 1, output_c_names.data(), output_tensors.data(), output_tensors.size());

    onnx::ModelProto model;
    model.ParseFromString(session.GetModelProto());
    onnx::ModelProto optimized_model = optimizeModel(model);
    std::ofstream output_file(output_path, std::ios::binary);
    optimized_model.SerializeToOstream(&output_file);

    return 0;
}
