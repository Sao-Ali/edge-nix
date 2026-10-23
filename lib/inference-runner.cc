#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sys/utsname.h>
#include <gnu/libc-version.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

using json = nlohmann::json;

static json read_json(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  json value;
  in >> value;
  return value;
}

static void check_debian_13() {
  std::ifstream release("/etc/os-release");
  if (!release) throw std::runtime_error("cannot read /etc/os-release");
  bool debian = false, version = false;
  std::string line;
  while (std::getline(release, line)) {
    if (line == "ID=debian" || line == "ID=\"debian\"") debian = true;
    if (line == "VERSION_ID=13" || line == "VERSION_ID=\"13\"") version = true;
  }
  if (!debian || !version) throw std::runtime_error("requires Debian 13 userland");
}

static std::string sha256(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing bundle file: " + path);
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (!ctx) throw std::runtime_error("SHA-256 initialization failed");
  EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
  char buffer[65536];
  while (in) {
    in.read(buffer, sizeof(buffer));
    if (in.gcount()) EVP_DigestUpdate(ctx, buffer, static_cast<size_t>(in.gcount()));
  }
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  EVP_DigestFinal_ex(ctx, digest, &length);
  EVP_MD_CTX_free(ctx);
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned int i = 0; i < length; ++i) {
    result += hex[digest[i] >> 4];
    result += hex[digest[i] & 15];
  }
  return result;
}

static void flatten(const json &node, std::vector<int64_t> &dims,
                    std::vector<double> &values, size_t depth = 0) {
  if (node.is_array()) {
    if (dims.size() <= depth) dims.push_back(static_cast<int64_t>(node.size()));
    if (dims[depth] != static_cast<int64_t>(node.size()))
      throw std::runtime_error("ragged input array");
    for (const auto &part : node) flatten(part, dims, values, depth + 1);
  } else {
    if (depth != dims.size() || !node.is_number())
      throw std::runtime_error("input must be a rectangular numeric tensor");
    values.push_back(node.get<double>());
  }
}

static json tensor_json(const std::vector<json> &values, const std::vector<int64_t> &dims,
                        size_t &offset, size_t depth = 0) {
  if (depth == dims.size()) return values[offset++];
  json out = json::array();
  for (int64_t i = 0; i < dims[depth]; ++i)
    out.push_back(tensor_json(values, dims, offset, depth + 1));
  return out;
}

static json output_json(const Ort::Value &value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  auto dims = info.GetShape();
  size_t count = info.GetElementCount();
  std::vector<json> numbers(count);
  switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const auto *data = value.GetTensorData<float>();
      for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) throw std::runtime_error("nonfinite float32 output is not supported by JSON contract");
        numbers[i] = data[i];
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const auto *data = value.GetTensorData<int64_t>();
      for (size_t i = 0; i < count; ++i) {
        if (data[i] < -9007199254740991LL || data[i] > 9007199254740991LL)
          throw std::runtime_error("int64 JSON output exceeds exact 2^53-1 range");
        numbers[i] = data[i];
      }
      break;
    }
    default: throw std::runtime_error("unsupported output dtype; only float32 and int64 are supported");
  }
  size_t offset = 0;
  return tensor_json(numbers, dims, offset);
}

static json infer(Ort::Session &session, const json &input) {
  if (!input.is_object()) throw std::runtime_error("input must be an object keyed by model input name");
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> names;
  std::vector<const char *> name_ptrs;
  std::vector<Ort::Value> tensors;
  std::vector<std::vector<float>> floats;
  std::vector<std::vector<int64_t>> integers;
  size_t n = session.GetInputCount();
  floats.reserve(n); integers.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    names.emplace_back(name.get());
    if (!input.contains(names.back())) throw std::runtime_error("missing input: " + names.back());
    auto type_info = session.GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims;
    std::vector<double> values;
    flatten(input.at(names.back()), dims, values);
    auto expected = info.GetShape();
    if (dims.size() != expected.size())
      throw std::runtime_error("input rank mismatch: " + names.back() +
        " got " + std::to_string(dims.size()) + " expected " + std::to_string(expected.size()));
    for (size_t d = 0; d < dims.size(); ++d)
      if (expected[d] >= 0 && dims[d] != expected[d])
        throw std::runtime_error("input shape mismatch: " + names.back());
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      auto &data = floats.emplace_back();
      for (double v : values) {
        if (!std::isfinite(v) || std::abs(v) > 3.402823466e38)
          throw std::runtime_error("input value outside float32 range");
        data.push_back(static_cast<float>(v));
      }
      tensors.push_back(Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), dims.data(), dims.size()));
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto &data = integers.emplace_back();
      for (double v : values) {
        if (!std::isfinite(v) || std::trunc(v) != v || std::abs(v) > 9007199254740991.0)
          throw std::runtime_error("int64 JSON input must contain exact integers within 2^53-1");
        data.push_back(static_cast<int64_t>(v));
      }
      tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory, data.data(), data.size(), dims.data(), dims.size()));
    } else throw std::runtime_error("unsupported input dtype; only float32 and int64 are supported");
  }
  for (auto &name : names) name_ptrs.push_back(name.c_str());
  std::vector<std::string> out_names;
  std::vector<const char *> out_ptrs;
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    auto name = session.GetOutputNameAllocated(i, allocator);
    out_names.emplace_back(name.get());
  }
  for (auto &name : out_names) out_ptrs.push_back(name.c_str());
  auto results = session.Run(Ort::RunOptions{nullptr}, name_ptrs.data(), tensors.data(), tensors.size(),
                             out_ptrs.data(), out_ptrs.size());
  json out = json::object();
  for (size_t i = 0; i < results.size(); ++i) out[out_names[i]] = output_json(results[i]);
  return out;
}

static void compare(const json &actual, const json &expected, json &checks) {
  const auto &specifications = expected.at("outputs");
  if (!specifications.is_object() || specifications.empty())
    throw std::runtime_error("expectations require a nonempty outputs object");
  for (auto it = specifications.begin(); it != specifications.end(); ++it) {
    if (!actual.contains(it.key())) throw std::runtime_error("missing output: " + it.key());
    const auto &spec = it.value();
    if (!spec.is_object() ||
        !(spec.contains("shape") || spec.contains("argmax") || spec.contains("values")))
      throw std::runtime_error("output has no supported checks: " + it.key());
    const auto &value = actual.at(it.key());
    std::vector<int64_t> shape;
    std::vector<double> flat;
    flatten(value, shape, flat);
    if (spec.contains("shape") && shape != spec.at("shape").get<std::vector<int64_t>>())
      throw std::runtime_error("output shape differs: " + it.key());
    if (spec.contains("argmax")) {
      auto best = std::max_element(flat.begin(), flat.end());
      if (best == flat.end() || std::distance(flat.begin(), best) != spec.at("argmax").get<int64_t>())
        throw std::runtime_error("output argmax differs: " + it.key());
    }
    if (spec.contains("values")) {
      std::vector<int64_t> ref_shape;
      std::vector<double> ref;
      flatten(spec.at("values"), ref_shape, ref);
      if (shape != ref_shape) throw std::runtime_error("output value shape differs: " + it.key());
      double atol = spec.value("atol", 0.0), rtol = spec.value("rtol", 0.0);
      for (size_t i = 0; i < flat.size(); ++i)
        if (!std::isfinite(flat[i]) || std::abs(flat[i] - ref[i]) > atol + rtol * std::abs(ref[i]))
          throw std::runtime_error("output values differ: " + it.key());
    }
    checks.push_back({{"output", it.key()}, {"passed", true}});
  }
}

int main(int argc, char **argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: edge-run --check | --input FILE; edge-validate --input FILE --expected FILE");
    const std::string root = argv[1];
    const std::string command = argv[2];
    const json manifest = read_json(root + "/manifest.json");
    if (manifest.at("schemaVersion") != 3 || manifest.at("artifactType") != "inference-bundle")
      throw std::runtime_error("unsupported inference manifest");
    struct utsname system;
    if (uname(&system) || std::string(system.machine) != "aarch64")
      throw std::runtime_error("requires aarch64 Linux");
    int kernel_major = 0, kernel_minor = 0;
    if (std::sscanf(system.release, "%d.%d", &kernel_major, &kernel_minor) != 2 ||
        kernel_major < 6 || (kernel_major == 6 && kernel_minor < 1))
      throw std::runtime_error("requires Linux kernel 6.1 or newer");
    for (auto it = manifest.at("files").begin(); it != manifest.at("files").end(); ++it)
      if (sha256(root + "/" + it.key()) != it.value().get<std::string>())
        throw std::runtime_error("bundle hash mismatch: " + it.key());
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-nix");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(2);
    Ort::Session session(env, (root + "/model/model.onnx").c_str(), options);
    if (command == "--check") {
      check_debian_13();
      std::cout << json({{"passed", true}, {"architecture", system.machine},
                         {"kernel", system.release}, {"glibc", gnu_get_libc_version()},
                         {"onnxruntime", OrtGetApiBase()->GetVersionString()},
                         {"modelSha256", manifest.at("preparedModelSha256")}}).dump() << '\n';
      return 0;
    }
    std::string input_path, expected_path;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
      else if (arg == "--expected" && i + 1 < argc) expected_path = argv[++i];
      else throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (input_path.empty()) throw std::runtime_error("--input is required");
    json result = infer(session, read_json(input_path));
    if (expected_path.empty()) { std::cout << result.dump() << '\n'; return 0; }
    json checks = json::array();
    compare(result, read_json(expected_path), checks);
    std::cout << json({{"passed", true}, {"checks", checks}, {"outputs", result}}).dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << json({{"passed", false}, {"error", error.what()}}).dump() << '\n';
    return 1;
  }
}
