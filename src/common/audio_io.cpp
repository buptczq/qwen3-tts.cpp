#include "qwen3_tts.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace qwen3_tts {

// WAV file loading (16-bit PCM or 32-bit float)
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read the whole file into memory, then parse with load_audio_from_memory.
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) {
        fprintf(stderr, "ERROR: Empty WAV file: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)sz);
    if (fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        return false;
    }
    fclose(f);
    return load_audio_from_memory(buf.data(), buf.size(), samples, sample_rate);
}

// Parse a WAV byte buffer (16/32-bit PCM or 32-bit float) into mono float
// samples. Multi-channel input is downmixed to mono by averaging channels.
bool load_audio_from_memory(const void * data, size_t data_size,
                            std::vector<float> & samples, int & sample_rate) {
    samples.clear();
    sample_rate = 0;
    if (!data || data_size < 44) return false;
    const uint8_t * p = (const uint8_t *)data;
    const uint8_t * end = p + data_size;

    auto read_u16 = [&](const uint8_t * q) -> uint16_t {
        return (uint16_t)(q[0] | (q[1] << 8));
    };
    auto read_u32 = [&](const uint8_t * q) -> uint32_t {
        return (uint32_t)(q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24));
    };

    if (memcmp(p, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        return false;
    }
    p += 8;  // skip RIFF + file_size
    if (memcmp(p, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        return false;
    }
    p += 4;

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (p + 8 <= end) {
        uint32_t chunk_size = read_u32(p + 4);
        const uint8_t * chunk_data = p + 8;
        if (chunk_data + chunk_size > end) chunk_size = (uint32_t)(end - chunk_data);

        if (memcmp(p, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;
            audio_format     = read_u16(chunk_data + 0);
            num_channels     = read_u16(chunk_data + 2);
            sr               = read_u32(chunk_data + 4);
            bits_per_sample  = read_u16(chunk_data + 14);
        } else if (memcmp(p, "data", 4) == 0) {
            sample_rate = (int)sr;

            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n = (int)(chunk_size / (2 * num_channels));
                    samples.resize(n);
                    const int16_t * raw = (const int16_t *)chunk_data;
                    for (int i = 0; i < n; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c)
                            sum += raw[i * num_channels + c] / 32768.0f;
                        samples[i] = sum / num_channels;
                    }
                } else if (bits_per_sample == 32) {
                    int n = (int)(chunk_size / (4 * num_channels));
                    samples.resize(n);
                    const int32_t * raw = (const int32_t *)chunk_data;
                    for (int i = 0; i < n; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c)
                            sum += raw[i * num_channels + c] / 2147483648.0f;
                        samples[i] = sum / num_channels;
                    }
                } else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    return false;
                }
            } else if (audio_format == 3) {  // IEEE float
                int n = (int)(chunk_size / (4 * num_channels));
                samples.resize(n);
                const float * raw = (const float *)chunk_data;
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c)
                        sum += raw[i * num_channels + c];
                    samples[i] = sum / num_channels;
                }
            } else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                return false;
            }
            return true;
        }
        // advance to next chunk (chunk_size + padding to even)
        size_t adv = 8 + chunk_size + (chunk_size & 1);
        if (p + adv > end) break;
        p += adv;
    }

    fprintf(stderr, "ERROR: No data chunk found\n");
    return false;
}

// WAV file saving (16-bit PCM at specified sample rate)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }

    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = (uint32_t) (samples.size() * block_align);
    uint32_t file_size = 36 + data_size;

    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = (uint32_t) sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t) (sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }

    fclose(f);
    return true;
}

} // namespace qwen3_tts
