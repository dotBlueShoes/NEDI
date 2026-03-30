// Made by Matthew Strumillo 2024.07.20
//
#pragma once
//
#include <assert.h>
//
#include <stb_image.h>
#include <stb_image_write.h>
//
#include "blue_impl.hpp"
#include <blue/struct.hpp>

//  ABOUT
// Packed impl. uses BMI2 cpu-extension. Might produce - illegal instruction (#UD).
//
//  IMPORTANT
// 'Meta32' type has more then one 'ZERO' definition. That's because it only 
//  makes sense of the image to have at least 1 pixel width/height and 1 channel.
//  Therefore if any/all of these values are 0 it is equal to 0.
// 
//  Instead of patching that vulnerability it is recommended to use a different system/type 
//  that ensures the size and channel count of an image. For example "flags" that would 
//  dictate if the image is "256x256x3" or "64x64x4" and such...
//
namespace IMAGE::PACKED {

    // -----------------------------------------
    // Meta32 - BIT STRUCTURE
    // -----------------------------------------
    // 0b0000'0000'0000'0000'0000'0000'0000'0000
    //   ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^
    //   3333 2222 2222 2222 2211 1111 1111 1111
    //  1 -> width (4 095) +1
    //  2 -> height (4 095) +1
    //  3 -> channels (15)
    // -----------------------------------------

    using Meta32 = u32;
    const u16 META32_MASK_WIDTH_OR_HEIGHT = 0b0011'1111'1111'1111;
    const u32 META32_MASK_CHANNELS = 0b1100'0000'0000'0000'1100'0000'0000'0000;
    const u32 META32_ZERO = 0b0000'0000'0000'0000'0000'0000'0000'0000;

    // --- Memory Cycles --- (~) (also reffert by me as MM-cycles)
    // RAM -> 100+ cycles
    // L3  -> 36   cycles	
    // L2  -> 12   cycles	
    // L1  -> 4    cycles	
    // REG -> 1    cycles
    // ---------------------

    inline void Get (
        OT u16    REF width,
        OT u16    REF height,
        OT u8     REF channels,
        IN Meta32 REF meta
    ) {
        width    = v32l (meta) & META32_MASK_WIDTH_OR_HEIGHT; // 2op (mov + and) : (MM-cycles + 1-cycles)
        height   = v32h (meta) & META32_MASK_WIDTH_OR_HEIGHT; // 2op (mov + and) : (MM-cycles + 1-cycles)
        channels = _pext_u32 (meta, META32_MASK_CHANNELS);    // 2op (mov + pext): (MM-cycles + 3-cycles)
    }

    inline void Set (
        OT Meta32 REF meta,
        IN u16    REF width,
        IN u16    REF height,
        IN u8     REF channels
    ) {

        { // debug-only parameters range check
            assert (height < 16384);
            assert (channels < 16);
            assert (width < 16384);
        }

        a32l (meta) = width;    // 1op (mov): (MM-cycles)
        a32h (meta) = height;   // 1op (mov): (MM-cycles)

        // 1op (pdep): (3-cycles)
        //
        REG u32 packedChannel = _pdep_u32 (channels, META32_MASK_CHANNELS);

        //  ABOUT
        // Preserve width/height bits.
        //
        meta |= packedChannel; // 3op (mov + or + mov): (MM-cycles + 1-cycles + MM-cycles)
    }

    void Load (
		OT Meta32 REF meta,
        OT u8*    REF data,
		IN c8*    CEF filepath
	) {
		s32 channels, width, height;

		data = stbi_load (filepath, &width, &height, &channels, 0);
		if (data == nullptr) ERROR ("IMAGE. Incorrect filepath provided! (%s)\n", filepath);

        if ((width > 16383) && (height > 16383) && (channels > 15))
            ERROR ("IMAGE. The following size/channel_count is not supported! (min: 16383, 16383, 15)\n");

        IMAGE::PACKED::Set (meta, width, height, channels);

		INCALLOCCO ();
	}

	void Free (
		IT void* CPY handleData
	) {
		stbi_image_free (handleData);
		DECALLOCCO ();
	}

}

namespace IMAGE {

    STRUCT (48, Meta48) {
        u16 width;
        u16 height;
        u8  channels;
        BB (8)
    };

	void Load (
		OT Meta48 REF meta,
        OT u8*    REF imageData,
		IN c8*    CEF filepath
	) {
		s32 channels, width, height;

		imageData = stbi_load (filepath, &width, &height, &channels, 0);
		if (imageData == nullptr) ERROR ("IMAGE. Incorrect filepath provided! (%s)\n", filepath);

		{
			meta.channels = channels;
			meta.height = height;
			meta.width = width;
		}

		INCALLOCCO ();
	}

	void Free (
		IT void* CPY handleData
	) {
		stbi_image_free (handleData);
		DECALLOCCO ();
	}

}

namespace IMAGE::PNG {

    void Save (
		IN Meta48 REF header,
        IN u8*    CEF imageData,
		IN c8*    CEF filepath
	) {
		const auto strideBytes = header.width * header.channels;
		stbi_write_png (filepath, header.width, header.height, header.channels, imageData, strideBytes);
	}

    void Save (
		IN PACKED::Meta32 REF header,
        IN u8*            CEF imageData,
		IN c8*            CEF filepath
	) {
        u16 width; u16 height; u8 channels; u32 strideBytes;

        PACKED::Get (width, height, channels, header);
		strideBytes = width * channels;

		stbi_write_png (filepath, width, height, channels, imageData, strideBytes);
	}

}
