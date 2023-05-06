#include <stdio.h>
#include <vector>
#include <string>
#include <string_view>
#include <stdarg.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define LODEPNG_COMPILE_ENCODER
#include "lodepng.h"

#include <queue>

using u8 = uint8_t;
using s32 = int32_t;
using u32 = uint32_t;
using f32 = float;
#define H_SQRT_2 (0.5f * M_SQRT2f32)

struct svec2 {
    s32 x, y;
    
    inline s32 area() const {
        return x * y;
    }

    friend bool operator<(const svec2& a, const svec2& b) {
        return false;
    }

    friend bool operator>(const svec2& a, const svec2& b) {
        return true;
    }
};

constexpr u32 magic_number = 0xffee;
constexpr u32 MAX_PACKAGE_SIZE = 1 * 1024 * 1024;

bool ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

void print_error(std::string str, ...) {
    va_list l;
    va_start(l, str);

    printf("An error occured. ");
    vprintf(str.c_str(), l);
    printf("\n");

    va_end(l);

    exit(-1);
}

void read_entire_file(const char* path, u8*& data, u32& size) {
    FILE* file = fopen(path, "rb");
    
    if(!file) {
        print_error("Tried to open file at '%s' for reading but the path does not exist.", path);
        return;
    }

    fseek(file, 0, SEEK_END);
    size = (u32)ftell(file);
    rewind(file);

    if(!size) {
        print_error("The file at '%s' is empty.", path);
        return;
    }

    data = new u8[size];
    size_t read = fread(data, 1, size, file);
    
    if(read != size) {
        print_error("The file at '%s' could not be read. Read %u, but expected %u.", path, (u32)read, size);
        return;
    }
    
    fclose(file);
}

void write_entire_file(const char* path, u8* data, u32 size) {
    FILE* file = fopen(path, "w");

    if(!file) {
        print_error("Tried to open file at '%s' for writing but it failed. Maybe the directory does not exist?", path);
        return;
    }

    size_t wrote = fwrite(data, 1, size, file);

    if(wrote != size) {
        print_error("Tried to write %u bytes to file but only wrote %u.", size, (u32)wrote);
        return;
    }

    fclose(file);
}

#if true

u8* encode_png(u8* data, const svec2& size, u32& out_size) {
    u8* out;
    size_t out_size_t;

    u32 error = lodepng_encode_memory(&out, &out_size_t, data, size.x, size.y, LCT_GREY, 8);
    out_size = (u32)out_size_t;

    if(error) return {};
    return out;
}

#else
std::vector<u8> encode_png(u8* data, const svec2& size) {
    std::vector<u8> buffer;
    u32 error;

    lodepng::State state;
    state.encoder.filter_palette_zero = 0; //We try several filter types, including zero, allow trying them all on palette images too.
    state.encoder.add_id = false; //Don't add LodePNG version chunk to save more bytes
    state.encoder.text_compression = 1; //Not needed because we don't add text chunks, but this demonstrates another optimization setting
    state.encoder.zlibsettings.nicematch = 258; //Set this to the max possible, otherwise it can hurt compression
    state.encoder.zlibsettings.lazymatching = 1; //Definitely use lazy matching for better compression
    state.encoder.zlibsettings.windowsize = 32768; //Use maximum possible window size for best compression

    size_t bestsize = 0;
    bool inited = false;

    LodePNGFilterStrategy strategies[4] = { LFS_ZERO, LFS_MINSUM, LFS_ENTROPY, LFS_BRUTE_FORCE };
    std::string strategynames[4] = { "LFS_ZERO", "LFS_MINSUM", "LFS_ENTROPY", "LFS_BRUTE_FORCE" };

    // min match 3 allows all deflate lengths. min match 6 is similar to "Z_FILTERED" of zlib.
    int minmatches[2] = { 3, 6 };

    int autoconverts[2] = { 0, 1 };
    std::string autoconvertnames[2] = { "0", "1" };
    
    // Try out all combinations of everything
    for(int i = 0; i < 4; i++)   //filter strategy
    for(int j = 0; j < 2; j++)   //min match
    for(int k = 0; k < 2; k++)   //block type (for small images only)
    for(int l = 0; l < 2; l++) { //color convert strategy
        if(bestsize > 3000 && (k > 0 || l > 0)) continue; /* these only make sense on small images */
        std::vector<unsigned char> temp;
        state.encoder.filter_strategy = strategies[i];
        state.encoder.zlibsettings.minmatch = minmatches[j];
        state.encoder.zlibsettings.btype = k == 0 ? 2 : 1;
        state.encoder.auto_convert = autoconverts[l];
        error = lodepng::encode(temp, data, size.x, size.y, state);

        if(error) {
            return {};
        }

        if(!inited || temp.size() < bestsize)
        {
            bestsize = temp.size();
            temp.swap(buffer);
            inited = true;
        }
    }

    return buffer;
}
#endif

struct buffer {
    u8* data;
    u32 used, size;

    buffer() : used(0), size(MAX_PACKAGE_SIZE) {
        data = new u8[size];
    }

    template<typename T>
    u32 write(T t) {
        u32 temp = used;
        if(used + sizeof(T) > size) {
            print_error("Max package size has been exceeded.");
            return 0;
        }

        *reinterpret_cast<T*>(data + used) = t;

        used += sizeof(T);
        return temp;
    }

    u32 write_buffer(u8* data, u32 data_size) {
        u32 temp = used;
        if(used + data_size > size) {
            print_error("Max package size has been exceeded.");
            return 0;
        }

        memcpy(this->data + used, data, data_size);

        used += data_size;
        return temp;
    }
};

enum class asset_type : u32 {
    image, sdf_animation, font, neural_network
};

struct asset_entry {
    asset_type type;
    std::string name;
    u32 name_offset;
    u32 entry_offset;

    u8* data;
    u32 data_size;
    u32 data_offset;

    union {
        struct {
            s32 width, height, channels;
        } image;
        struct {
            s32 width, height, depth;
            f32 zero_dist;
        } sdf_animation;
    };
};

#define PIXEL_AT(data, x, y) data[(y) * sx + (x)]
#define PIXEL_NOT_SET(x, y) (PIXEL_AT(data, x, y) == 0 && PIXEL_AT(sdf, x, y) == -1.0f)

inline void add_pixel_neighbors(s32 x, s32 y, std::priority_queue<std::pair<f32, svec2>, std::vector<std::pair<f32, svec2>>, std::greater_equal<std::pair<f32, svec2>>>& queue,
    u8* data, f32* sdf, const svec2& size) {
    
    s32 sx = size.x, sy = size.y;
    f32 dist = PIXEL_AT(sdf, x, y);
    
    if(x - 1 >= 0 && PIXEL_NOT_SET(x - 1, y)) {
        queue.push(std::make_pair<f32, svec2>(dist + 0.5f, { x - 1, y }));
    }

    if(x + 1 < sx && PIXEL_NOT_SET(x + 1, y)) {
        queue.push(std::make_pair<f32, svec2>(dist + 0.5f, { x + 1, y }));
    }
            
    if(y - 1 >= 0 && PIXEL_NOT_SET(x, y - 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + 0.5f, { x, y - 1 }));
    }

    if(y + 1 < sy && PIXEL_NOT_SET(x, y + 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + 0.5f, { x, y + 1 }));
    }

    // diagonals
    if(x - 1 >= 0 && y - 1 >= 0 && PIXEL_NOT_SET(x - 1, y - 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + H_SQRT_2, { x - 1, y - 1 }));
    }

    if(x + 1 < sx && y - 1 >= 0 && PIXEL_NOT_SET(x + 1, y - 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + H_SQRT_2, { x + 1, y - 1 }));
    }
            
    if(x - 1 >= 0 && y + 1 < sy && PIXEL_NOT_SET(x - 1, y + 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + H_SQRT_2, { x - 1, y + 1 }));
    }

    if(x + 1 < sx && y + 1 < sy && PIXEL_NOT_SET(x + 1, y + 1)) {
        queue.push(std::make_pair<f32, svec2>(dist + H_SQRT_2, { x + 1, y + 1 }));
    }
} 

f32* half_sdf_from_mask(u8* data, const svec2& size) {
    f32* sdf = new f32[size.area()];
    s32 sx = size.x, sy = size.y;

    std::priority_queue<std::pair<f32, svec2>, std::vector<std::pair<f32, svec2>>, std::greater_equal<std::pair<f32, svec2>>> queue;
    
    for(s32 i = 0; i < size.area(); i++) sdf[i] = -1.0f;

    // add the initial border pixels
    for(s32 y = 0; y < size.y; y++) {
        for(s32 x = 0; x < size.x; x++) {
            if(PIXEL_AT(data, x, y) == 0) continue;

            bool mark = false;

            if(x - 1 >= 0 && PIXEL_AT(data, x - 1, y) == 0) {
                queue.push(std::make_pair<f32, svec2>(0.5f, { x - 1, y }));
                mark = true;
            }

            if(x + 1 < sx && PIXEL_AT(data, x + 1, y) == 0) {
                queue.push(std::make_pair<f32, svec2>(0.5f, { x + 1, y }));
                mark = true;
            }

            if(y - 1 >= 0 && PIXEL_AT(data, x, y - 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(0.5f, { x, y - 1 }));
                mark = true;
            }

            if(y + 1 < sy && PIXEL_AT(data, x, y + 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(0.5f, { x, y + 1 }));
                mark = true;
            }

            // diagonals

            if(x - 1 >= 0 && y - 1 >= 0 && PIXEL_AT(data, x - 1, y - 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(H_SQRT_2, { x - 1, y - 1 }));
                mark = true;
            }

            if(x + 1 < sx && y - 1 >= 0 && PIXEL_AT(data, x + 1, y - 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(H_SQRT_2, { x + 1, y - 1 }));
                mark = true;
            }

            if(x - 1 >= 0 && y + 1 < sy && PIXEL_AT(data, x - 1, y + 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(H_SQRT_2, { x - 1, y + 1 }));
                mark = true;
            }

            if(x + 1 < sx && y + 1 < sy && PIXEL_AT(data, x + 1, y + 1) == 0) {
                queue.push(std::make_pair<f32, svec2>(H_SQRT_2, { x + 1, y + 1 }));
                mark = true;
            }

            if(mark) PIXEL_AT(sdf, x, y) = 0.0f;
        }
    }

    // work through the entire image
    while(!queue.empty()) {
        const std::pair<f32, svec2>& pixel = queue.top();
        queue.pop();

        f32 dist = pixel.first;
        const svec2& pos = pixel.second;

        if(PIXEL_NOT_SET(pos.x, pos.y)) {
            PIXEL_AT(sdf, pos.x, pos.y) = dist;
            add_pixel_neighbors(pos.x, pos.y, queue, data, sdf, size);
        }
    }

    // zero out inner regions
    for(s32 i = 0; i < size.area(); i++) {
        if(sdf[i] == -1.0f) sdf[i] = 0.0f;
    }

    return sdf;
}

f32* full_sdf_from_mask(u8* data, const svec2& size) {
    f32* outer_sdf = half_sdf_from_mask(data, size);

    for(s32 i = 0; i < size.area(); i++) data[i] = ~data[i];
    f32* inner_sdf = half_sdf_from_mask(data, size);

    f32* sdf = inner_sdf;
    for(s32 i = 0; i < size.area(); i++) {
        sdf[i] = outer_sdf[i] - inner_sdf[i];
    }

    delete[] outer_sdf;
    return sdf;
}

u8* normalize_sdf(f32* sdf, const svec2& size, f32 min_dist, f32 max_dist) {
    u8* norm_sdf = new u8[size.area()];

    for(s32 i = 0; i < size.area(); i++) {
        f32 norm = (sdf[i] - min_dist) / (max_dist - min_dist);
        norm = std::min(std::max(norm, 0.0f), 1.0f);
        norm_sdf[i] = (u8)(255.0f * norm);
    }

    return norm_sdf;
}

u8* normalize_sdf(f32* sdf, const svec2& size, f32& min_dist, f32& max_dist, f32& zero_dist) {
    min_dist = 9999999.0f;
    for(s32 i = 0; i < size.area(); i++) min_dist = std::min(min_dist, sdf[i]);

    max_dist = 0.0f;
    for(s32 i = 0; i < size.area(); i++) max_dist = std::max(max_dist, sdf[i]);

    zero_dist = -min_dist / (max_dist - min_dist);

    return normalize_sdf(sdf, size, min_dist, max_dist);
}

struct table_of_contents {
    std::vector<asset_entry> entries;
};

struct package {
    table_of_contents contents;
    
    void add_image(const std::string& name, const std::string& path) {
        asset_entry entry = {};
        entry.type = asset_type::image;
        entry.name = name;

        bool is_image = ends_with(path, ".jpg") || ends_with(path, ".png");

        if(!is_image) {
            print_error("Tried to add image at path '%s' which has an unsupported extension.", path.c_str());
            return;
        }

        read_entire_file(path.c_str(), entry.data, entry.data_size);

        u8* pixels = stbi_load_from_memory(entry.data, entry.data_size, &entry.image.width, &entry.image.height, &entry.image.channels, 0);
        
        if(!pixels) {
            print_error("Tried to add image at path '%s' which has an unsupported internal type.", path.c_str());
            return;
        }
        
        stbi_image_free(pixels);

        contents.entries.push_back(entry);

        printf("Added image entry '%s' (width=%d, height=%d, channels=%d).\n", 
            entry.name.c_str(), entry.image.width, entry.image.height, entry.image.channels);
    }
    
    void add_sdf_animation(const std::string& name, const std::vector<std::string>& paths) {
        asset_entry entry = {};
        entry.type = asset_type::sdf_animation;
        entry.name = name;

        svec2 sdf_size;
        u8* full_sdf;
        f32 min_dist, max_dist;

        for(s32 i = 0; i < paths.size(); i++) {
            const std::string& path = paths[i];
            bool is_sdf_anim = ends_with(path, ".png");

            if(!is_sdf_anim) {
                print_error("Tried to add sdf animation at path '%s' which has an unsupported extension.", path.c_str());
                return;
            }

            read_entire_file(path.c_str(), entry.data, entry.data_size);

            svec2 size; 
            s32 channels;
            u8* pixels = stbi_load_from_memory(entry.data, entry.data_size, &size.x, &size.y, &channels, 0);

            if(i != 0 && (size.x != sdf_size.x || size.y != sdf_size.y)) {
                print_error("Tried to add sdf animation with inconsistent sizes.");
                return;
            }

            sdf_size = size;
            if(i == 0) full_sdf = new u8[sdf_size.area() * paths.size()];
            
            if(!pixels) {
                print_error("Tried to load sdf animation image at path '%s' which has an unsupported internal type.", path.c_str());
                return;
            }
            
            f32* sdf = full_sdf_from_mask(pixels, size);
            u8* norm_sdf;
            
            if(i == 0) norm_sdf = normalize_sdf(sdf, size, min_dist, max_dist, entry.sdf_animation.zero_dist);
            else norm_sdf = normalize_sdf(sdf, size, min_dist, max_dist);

            memcpy(full_sdf + sdf_size.area() * i, norm_sdf, sdf_size.area());

            delete[] sdf;
            delete[] norm_sdf;
            stbi_image_free(pixels);
        }
        
        entry.data = encode_png(full_sdf, { sdf_size.x, sdf_size.y * (s32)paths.size() }, entry.data_size);
        entry.sdf_animation.width = sdf_size.x;
        entry.sdf_animation.height = sdf_size.y;
        entry.sdf_animation.depth = (u32)paths.size();

        contents.entries.push_back(entry);

        printf("Added sdf animation entry '%s'.\n", entry.name.c_str());
    }
    
    void add_font(const std::string& name, const std::string& path) {
        asset_entry entry = {};
        entry.type = asset_type::font;
        entry.name = name;

        bool is_font = ends_with(path, ".ttf");

        if(!is_font) {
            print_error("Tried to add font at path '%s' which has an unsupported extension.", path.c_str());
            return;
        }

        read_entire_file(path.c_str(), entry.data, entry.data_size);

        contents.entries.push_back(entry);

        printf("Added font entry '%s'.\n", entry.name.c_str());
    }
    
    void add_neural_network(const std::string& name, const std::string& path) {
        asset_entry entry = {};
        entry.type = asset_type::neural_network;
        entry.name = name;

        bool is_nn = ends_with(path, ".tflite");

        if(!is_nn) {
            print_error("Tried to add neural network at path '%s' which has an unsupported extension.", path.c_str());
            return;
        }

        read_entire_file(path.c_str(), entry.data, entry.data_size);

        contents.entries.push_back(entry);

        printf("Added neural network entry '%s'.\n", path.c_str());
    }

    void write(const char* path) {
        buffer output;
        output.write<u32>(magic_number);

        // write out all data first
        for(s32 i = 0; i < contents.entries.size(); i++) {
            asset_entry& entry = contents.entries[i];

            entry.data_offset = output.write_buffer(entry.data, entry.data_size);
        }

        // write out the names
        for(s32 i = 0; i < contents.entries.size(); i++) {
            asset_entry& entry = contents.entries[i];
        
            entry.name_offset = output.write_buffer((u8*)entry.name.c_str(), entry.name.size());
        }
        
        // write out the entries
        for(s32 i = 0; i < contents.entries.size(); i++) {
            asset_entry& entry = contents.entries[i];

            entry.entry_offset = output.write<u32>((u32)entry.type);
            output.write<u32>(entry.name_offset);
            output.write<u32>((u32)entry.name.size());
            output.write<u32>(entry.data_offset);
            output.write<u32>(entry.data_size);

            switch(entry.type) {
                case asset_type::image: {
                    output.write<u32>(entry.image.width);
                    output.write<u32>(entry.image.height);
                    output.write<u32>(entry.image.channels);
                } break;
                case asset_type::sdf_animation: {
                    output.write<u32>(entry.sdf_animation.width);
                    output.write<u32>(entry.sdf_animation.height);
                    output.write<u32>(entry.sdf_animation.depth);
                    output.write<f32>(entry.sdf_animation.zero_dist);
                } break;
                case asset_type::font: break;
                case asset_type::neural_network: break;
                default: {
                    print_error("Asset type %u has not been implemented for entry write.", entry.type);
                    return;
                }
            }
        }

        // write out the table of contents
        u32 table_of_contents_offset = output.used;

        for(s32 i = 0; i < contents.entries.size(); i++) {
            asset_entry& entry = contents.entries[i];

            output.write<u32>(entry.entry_offset);
        }

        // write out the offset of the table of contents
        output.write<u32>(table_of_contents_offset);
        output.write<u32>((u32)contents.entries.size());

        // write out the file
        write_entire_file(path, output.data, output.used);

        printf("The package was successfully written to '%s'.\n", path);
    }
};

int main() {
    package pack;

    pack.add_image("one_note_icon", "/media/shared/Projekte/DocumentScanner/assets/packs/test/images/one_note_icon.png");
    pack.add_image("word_icon", "/media/shared/Projekte/DocumentScanner/assets/packs/test/images/word_icon.png");
    pack.add_image("gallery_icon", "/media/shared/Projekte/DocumentScanner/assets/packs/test/images/gallery_icon.png");
    pack.add_image("pdf_icon", "/media/shared/Projekte/DocumentScanner/assets/packs/test/images/pdf_icon.png");

    pack.add_sdf_animation("checked", {
        "/media/shared/Projekte/DocumentScanner/assets/packs/test/sdf_animations/checked/checked.png"
    });

    pack.add_sdf_animation("flash", {
        "/media/shared/Projekte/DocumentScanner/assets/packs/test/sdf_animations/flash/off.png",
        "/media/shared/Projekte/DocumentScanner/assets/packs/test/sdf_animations/flash/on.png"
    });

    pack.add_sdf_animation("stripes", {
        "/media/shared/Projekte/DocumentScanner/assets/packs/test/sdf_animations/stripes/default.png",
        "/media/shared/Projekte/DocumentScanner/assets/packs/test/sdf_animations/stripes/thick.png"
    });

    pack.add_font("main_font", "/media/shared/Projekte/DocumentScanner/assets/packs/test/fonts/font.ttf");
    pack.add_neural_network("contour_network", "/media/shared/Projekte/DocumentScanner/assets/packs/test/neural_networks/contour_model.tflite");

    pack.write("/media/shared/Projekte/DocumentScanner/app/app/src/main/assets/test.assetpack");
    pack.write("/media/shared/Projekte/DocumentScanner/supportlib/bin/linux/x86_64/test.assetpack");
}