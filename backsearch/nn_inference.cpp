/*
 * nn_inference.cpp — libtorch TorchScript inference shim.
 *
 * Exposes a tiny extern "C" API (see nn_inference.h) so the C code
 * in backsearch.c can call into a libtorch-loaded value model.
 *
 * Builds against the pip-installed PyTorch's C++ headers and dylibs
 * under $TORCH_DIR (see README's Building section).
 */

#include "nn_inference.h"

#include <torch/script.h>
#include <ATen/Parallel.h>

#include <cstdio>
#include <memory>
#include <vector>

namespace {

// Value-head model slot.
std::unique_ptr<torch::jit::script::Module> g_module;
float g_target_scale = 1.0f;
int   g_rows     = 0;
int   g_cols     = 0;
int   g_channels = 9;

// Surrogate model slot (forward_solve predictor).
std::unique_ptr<torch::jit::script::Module> g_surrogate;
float g_surrogate_target_scale = 1.0f;
int   g_surrogate_rows = 0;
int   g_surrogate_cols = 0;
int   g_surrogate_channels = 9;

} // namespace

extern "C" int nn_load(const char *model_path, float target_scale,
                       int rows, int cols, int channels) {
    try {
        // Pin libtorch to a single thread for predictable latency in the
        // single-state hot path.  Batched calls amortise this anyway.
        at::set_num_threads(1);
        at::set_num_interop_threads(1);

        auto mod = std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_path));
        mod->eval();
        g_module       = std::move(mod);
        g_target_scale = target_scale;
        g_rows         = rows;
        g_cols         = cols;
        g_channels     = channels;
        return 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_load failed: %s\n", e.what());
        return 0;
    }
}

extern "C" float nn_score(const float *features) {
    if (!g_module) return 0.0f;
    try {
        torch::NoGradGuard no_grad;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        // from_blob is a non-owning view; clone() so libtorch owns the
        // memory it operates on (allows backsearch to reuse the buffer).
        auto t = torch::from_blob(
            const_cast<float*>(features),
            {1, g_channels, g_rows, g_cols}, opts).clone();
        std::vector<torch::jit::IValue> inputs{t};
        auto out = g_module->forward(inputs).toTensor();
        return out.item<float>() * g_target_scale;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_score failed: %s\n", e.what());
        return 0.0f;
    }
}

extern "C" void nn_score_batch(const float *features, int batch_size,
                               float *out_scores) {
    if (!g_module || batch_size <= 0) return;
    try {
        torch::NoGradGuard no_grad;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto t = torch::from_blob(
            const_cast<float*>(features),
            {batch_size, g_channels, g_rows, g_cols}, opts).clone();
        std::vector<torch::jit::IValue> inputs{t};
        auto out = g_module->forward(inputs).toTensor().contiguous().cpu();
        const float *p = out.data_ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            out_scores[i] = p[i] * g_target_scale;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_score_batch failed: %s\n", e.what());
        for (int i = 0; i < batch_size; ++i) out_scores[i] = 0.0f;
    }
}

extern "C" void nn_close(void) {
    g_module.reset();
}


// --- surrogate model: duplicate of the above, separate state. -------------

extern "C" int nn_surrogate_load(const char *model_path, float target_scale,
                                 int rows, int cols, int channels) {
    try {
        auto mod = std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_path));
        mod->eval();
        g_surrogate              = std::move(mod);
        g_surrogate_target_scale = target_scale;
        g_surrogate_rows         = rows;
        g_surrogate_cols         = cols;
        g_surrogate_channels     = channels;
        return 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_surrogate_load failed: %s\n", e.what());
        return 0;
    }
}

extern "C" float nn_surrogate_score(const float *features) {
    if (!g_surrogate) return 0.0f;
    try {
        torch::NoGradGuard no_grad;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto t = torch::from_blob(
            const_cast<float*>(features),
            {1, g_surrogate_channels, g_surrogate_rows, g_surrogate_cols}, opts).clone();
        std::vector<torch::jit::IValue> inputs{t};
        auto out = g_surrogate->forward(inputs).toTensor();
        return out.item<float>() * g_surrogate_target_scale;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_surrogate_score failed: %s\n", e.what());
        return 0.0f;
    }
}

extern "C" void nn_surrogate_score_batch(const float *features, int batch_size,
                                         float *out_scores) {
    if (!g_surrogate || batch_size <= 0) return;
    try {
        torch::NoGradGuard no_grad;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto t = torch::from_blob(
            const_cast<float*>(features),
            {batch_size, g_surrogate_channels, g_surrogate_rows, g_surrogate_cols}, opts).clone();
        std::vector<torch::jit::IValue> inputs{t};
        auto out = g_surrogate->forward(inputs).toTensor().contiguous().cpu();
        const float *p = out.data_ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            out_scores[i] = p[i] * g_surrogate_target_scale;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nn_surrogate_score_batch failed: %s\n", e.what());
        for (int i = 0; i < batch_size; ++i) out_scores[i] = 0.0f;
    }
}

extern "C" void nn_surrogate_close(void) {
    g_surrogate.reset();
}
