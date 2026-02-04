// Made by Matthew Strumillo 2024.10.07
//
#pragma once
// ---
#define BLUELIB_IMPLEMENTATION
#include "blue_impl.hpp"
#include "cuda_impl.hpp"
#include "eigen_impl.hpp"
// ---

//using mat16x4u8 = Eigen::Matrix<u8, 16, 4>;
//using mat4u8    = Eigen::Matrix<u8, 4, 4>;
//using mat4x1u32 = Eigen::Matrix<r32, 4, 1>;
//using vec16u8   = Eigen::Vector<u8, 16>;
//using mat4x16r32 = Eigen::Matrix<r32, 4, 16>;
//using mat4r32    = Eigen::Matrix<r32, 4, 4>;

using mat4x1r32  = Eigen::Matrix<r32, 4, 1>;

using mat16x4r32 = Eigen::Matrix<r32, 16, 4>;
using vec16r32   = Eigen::Vector<r32, 16>;

using mat12x4r32 = Eigen::Matrix<r32, 12, 4>;
using vec12r32   = Eigen::Vector<r32, 12>;

using mat24x4r32 = Eigen::Matrix<r32, 24, 4>;
using vec24r32   = Eigen::Vector<r32, 24>;

using mat = Eigen::MatrixXf;
using vec = Eigen::VectorXf;

// ---

#define GRAYSCALE_A(r, g, b) \
    (0.299f * r) + (0.587f * g) + (0.114f * b);

#define GRAYSCALE_B(r, g, b) \
    (0.21f * r) + (0.72f * g) + (0.07f * b);

#define GRAYSCALE_C(r, g, b) \
    (r + g + b) / 3;

#define GRAYSCALE(r, g, b) \
    GRAYSCALE_A (r, g, b)

// ---

//#define IMAGE_O0_FILEPATH "res\\o_640x640.png" 
//#define IMAGE_O1_FILEPATH "res\\o_320x320.png" 

#define IMAGE_FILEPATH_O0 "res\\" "upscale\\" "NEDIN_320x320x3.png"
#define IMAGE_FILEPATH_O1 "res\\" "upscale\\" "NEDIN_640x640x3.png"


// diagonal window grid (Y-vector):
// o-o-o-o
// -------
// o-o-o-o
// ---x---
// o-o-o-o
// -------
// o-o-o-o
// 
// diagonal coeffs (C-matrix-row):
// -o---o-
// -------
// ---y---
// -------
// -o---o-

// IMPORTANT !
//  From my understanding this is the grid used for 'Y' matrix in second step
//  and 'C' matrix coeffs position.

// second step window grid (Y-vector):
// ---o---
// --o-o--
// -o-o-o-
// o-oxo-o
// -o-o-o-
// --o-o--
// ---o---
// 
// then the second step coeffs (C-matrix-row) (same as diagonal):
// -o---o-
// -------
// ---y---
// -------
// -o---o-
//
// but the values we're waging from are:
// --o--
// -oyo-
// --o--

//DEBUG { // DEBUG -> Print Eigen Matrix
//DEBUG     std::stringstream ss;
//DEBUG     ss << C;
//DEBUG     LOGINFO ("C: %s\n", ss.str().c_str());
//DEBUG 
//DEBUG     ss.str (std::string());  // clear the buffer
//DEBUG     ss.clear ();             // reset flags (VERY IMPORTANT)
//DEBUG 
//DEBUG     ss << Y;
//DEBUG     LOGINFO ("Y: %s\n", ss.str().c_str());
//DEBUG }
