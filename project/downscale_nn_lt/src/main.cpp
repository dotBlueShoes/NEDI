
// Made by Matthew Strumillo 05.12.2025
//
#define BLUELIB_IMPLEMENTATION
#include "blue_impl.hpp"
#include "image.hpp"
//

s32 main (s32 argumentsCount, c8* arguments[]) {

    const c8 IN_FILEPATH_DEFAULT [] = "res\\textures\\hr\\0_HR.png";
    const c8 otFilepath [] = "res\\textures\\N_LR.png";
    c8* inFilepath;

    // --- Args interpreter --- Issue -> Hardcoded
        switch (argumentsCount) { 
    
            case 2: {
                const u8 argInFilepath = 1;
                LOGINFO ("[%d]-arg: %s\n", argInFilepath - 1, arguments[argInFilepath]);
                inFilepath = arguments[argInFilepath];
            } break;
    
            default:
                LOGWARN ("Invalid number of arguments passed!\n");
            case 1: { // Apply DEFAULT values for arguments.
                inFilepath = (c8*)(void*)IN_FILEPATH_DEFAULT;
            };
    
        }
    // ---

    BINIT ("Executing: Downscale\n");

    IMAGE::PACKED::Meta32 otImageMeta;
    u8* otImageData;
    IMAGE::PACKED::Meta32 inImageMeta;
    u8* inImageData;

    IMAGE::PACKED::Load (inImageMeta, inImageData, inFilepath);

    { // Create output image buffer.
        REG u8 channels; REG u16 height; REG u16 width; 
        IMAGE::PACKED::Get (width, height, channels, inImageMeta);

        LOGINFO ("image-in: w: %d, h: %d, c: %d\n", width, height, channels);

        u16 newHeight = height / 2;
        u16 newWidth = width / 2;

        u64 size = newWidth * newHeight * channels;

        IMAGE::PACKED::Set (otImageMeta, newWidth, newHeight, channels);
        ALLOCATE (u8, otImageData, size);

        { // downscale nn based on lt corner pixel (only 2x down)
            for (u16 iChannel = 0; iChannel < channels; ++iChannel) {
                for (u16 iiy = 0; iiy < newHeight; ++iiy) {
                    for (u16 iix = 0; iix < newWidth; ++iix) {
                        u64 ot = (iiy * newWidth * channels) + (iix * channels) + iChannel;
                        u64 in = (iiy * newWidth * 2 * channels * 2) + (iix * channels * 2) + iChannel;
                        otImageData[ot] = inImageData[in];
                    }
                }
            }
        }

    }

    { // Create an output image in the filesystem.
        IMAGE::PNG::Save (otImageMeta, otImageData, otFilepath);
    }

    { // Free output image data.
        REG u8 channels; REG u16 height; REG u16 width; 
        IMAGE::PACKED::Get (width, height, channels, otImageMeta);

        REG u64 size = width * height * channels;
        FREE (size, otImageData);
    }

    { // Free input image data.
        IMAGE::PACKED::Free (inImageData);
    }

    BSTOP ("Finalizing: Downscale\n");

	return 0;
}