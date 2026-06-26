// Voice Router Lite — native ASR one-shot worker
//
// Reads int16 LE mono PCM from stdin, writes recognized text to stdout.
// Designed to be spawned by the Python router daemon; process exit frees all
// sherpa-onnx memory (no Python runtime overhead).
//
// Build: see native/README.md

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace {

struct Options {
  std::string models_dir = "models";
  int32_t sample_rate = 16000;
  int32_t num_threads = 1;
};

void PrintUsage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [--models-dir DIR] [--sample-rate HZ] [--threads N]\n"
          "  Reads int16 mono PCM from stdin, prints transcript to stdout.\n",
          prog);
}

bool ParseArgs(int argc, char *argv[], Options *opts) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--models-dir") == 0 && i + 1 < argc) {
      opts->models_dir = argv[++i];
    } else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
      opts->sample_rate = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opts->num_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return false;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

std::string JoinPath(const std::string &dir, const char *name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

std::vector<uint8_t> ReadAllStdin() {
  std::vector<uint8_t> buf;
  uint8_t chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof(chunk), stdin)) > 0) {
    buf.insert(buf.end(), chunk, chunk + n);
  }
  return buf;
}

}  // namespace

int main(int argc, char *argv[]) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    return 2;
  }

  const std::string encoder = JoinPath(opts.models_dir, "asr/encoder.onnx");
  const std::string decoder = JoinPath(opts.models_dir, "asr/decoder.onnx");
  const std::string joiner = JoinPath(opts.models_dir, "asr/joiner.onnx");
  const std::string tokens = JoinPath(opts.models_dir, "asr/tokens.txt");

  std::vector<uint8_t> raw = ReadAllStdin();
  if (raw.size() < 2) {
    return 0;
  }

  const size_t num_samples = raw.size() / sizeof(int16_t);
  std::vector<float> samples(num_samples);
  const auto *pcm = reinterpret_cast<const int16_t *>(raw.data());
  for (size_t i = 0; i < num_samples; ++i) {
    samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
  }

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));
  config.feat_config.sample_rate = opts.sample_rate;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.encoder = encoder.c_str();
  config.model_config.transducer.decoder = decoder.c_str();
  config.model_config.transducer.joiner = joiner.c_str();
  config.model_config.tokens = tokens.c_str();
  config.model_config.num_threads = opts.num_threads;
  config.model_config.provider = "cpu";
  config.decoding_method = "greedy_search";
  config.enable_endpoint = 0;

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&config);
  if (!recognizer) {
    fprintf(stderr, "Failed to create recognizer. Check models under %s/asr/\n",
            opts.models_dir.c_str());
    return 1;
  }

  const SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(recognizer);
  SherpaOnnxOnlineStreamAcceptWaveform(stream, opts.sample_rate, samples.data(),
                                       static_cast<int32_t>(num_samples));
  SherpaOnnxOnlineStreamInputFinished(stream);

  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *result =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);
  if (result && result->text && result->text[0] != '\0') {
    fputs(result->text, stdout);
  }
  fflush(stdout);

  SherpaOnnxDestroyOnlineRecognizerResult(result);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  return 0;
}
