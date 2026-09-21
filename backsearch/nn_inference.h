/*
 * nn_inference.h — C-ABI hook into a libtorch TorchScript value model.
 *
 * The actual implementation lives in nn_inference.cpp and links against
 * libtorch / libtorch_cpu / libc10 (provided by pip-installed PyTorch
 * at $TORCH_DIR/lib).
 *
 * Usage from C:
 *   if (nn_load("checkpoints/value_v2.ts", target_scale, rows, cols, 9)) {
 *       float score = nn_score(features_buffer);
 *   }
 *
 * The features buffer must be laid out as channels × rows × cols
 * float32 in row-major order — see corpus_features.state_to_tensor
 * for the canonical channel definitions.
 */

#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load a TorchScript model.  Returns 1 on success, 0 on failure.
 * target_scale is multiplied into the model's output so the caller
 * sees real depth units (the training pipeline normalised by this). */
int nn_load(const char *model_path, float target_scale,
            int rows, int cols, int channels);

/* Score a single state.  features must point to channels*rows*cols
 * float32 values. */
float nn_score(const float *features);

/* Score a batch of states.  features points to batch_size*channels*rows*cols
 * float32 values; out_scores receives batch_size values. */
void nn_score_batch(const float *features, int batch_size, float *out_scores);

/* Release the model. */
void nn_close(void);


/* Second model slot for the solver surrogate.  Same API surface; separate
 * weights / metadata.  Use this when the value-head and the surrogate are
 * loaded simultaneously. */
int  nn_surrogate_load(const char *model_path, float target_scale,
                       int rows, int cols, int channels);
float nn_surrogate_score(const float *features);
void  nn_surrogate_score_batch(const float *features, int batch_size, float *out_scores);
void  nn_surrogate_close(void);

#ifdef __cplusplus
}
#endif

#endif
