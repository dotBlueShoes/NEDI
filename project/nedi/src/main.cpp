// Made by Matthew Strumillo 05.12.2025
//
#include "framework.hpp"
//
#include "image.hpp"

// -----------------------------------------------------------------------------------------------------
//  IMPORTANT -> Many ways of calculating Coeffs and why it gives different result.
// -----------------------------------------------------------------------------------------------------
// All 3 procedures if translated into mathematical equation: 
//  CalcCoeffsInverse, CalcCoeffsLDLT, CalcCoeffsQR
// Should give the same mathematical output. But it does not happen in the program, Why?
//  - Floating-point rounding	> Accumulates differently (and breaks most likely).
//  - Normal equations	        > Square conditioning.
//  - Pivoting (QR)	            > Adapts to column correlation.
//  - Order of ops	            > Different summation paths.
//
// CalcCoeffsInverse
//  - squares the condition number.
//  - explicit matrix inverse amplifies roundoff.
//  gives:
//   \> Large negative coefficient and large positive coefficient.
//
// CalcCoeffsLDLT
//  - condition number is squared.
//  - avoids explicit inverse and uses a symmetric factorization.
//  - sensitive when columns are correlated.
//  gives:
//   \> Still enlarges negative and positive coeffs.
//
// CalcCoeffsQR
//  - no condition-number squaring.
//  - pivoting handles near-dependencies.
//  gives:
//   \> Most accurate solution.
//
// but in realworld computer programming give 3 different results!
// -----------------------------------------------------------------------------------------------------

//__global__ void CudaDummy (
//    IN u16 CPY x,
//    IN u16 CPY y
//) {}

const u8 WINDOW_DIAGONAL_POSITIONS_CIRCULAR_4_SIZE = 12;
const u8 WINDOW_DIAGONAL_POSITIONS_CIRCULAR_4 [12][2] { 
            {1, 0}, {2, 0},
    {0, 1}, {1, 1}, {2, 1}, {3, 1},
    {0, 2}, {1, 2}, {2, 2}, {3, 2},
            {1, 3}, {2, 3},
};

const u8 WINDOW_DIAGONAL_POSITIONS_CIRCULAR_6_SIZE = 24;
const u8 WINDOW_DIAGONAL_POSITIONS_CIRCULAR_6 [24][2] { 
                    {2, 0}, {3, 0}, 
            {1, 1}, {2, 1}, {3, 1}, {4, 1}, 
    {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 2},
    {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3}, {5, 3},
            {1, 4}, {2, 4}, {3, 4}, {4, 4}, 
                    {2, 5}, {3, 5}, 
};


// Modified New Edge-Directed Interpolation Using Window Extension 2017
void ClampMNEDI (
    IT r32 REF out,
    IN u8 REF lu,
    IN u8 REF ur,
    IN u8 REF dl,
    IN u8 REF rd
) {
    u8 nmin = std::min (lu, std::min(ur, std::min(dl, rd)));
    u8 nmax = std::max (lu, std::max(ur, std::max(dl, rd)));
    out = std::min (std::max ((u8)out, nmin), nmax);
}


// simple clamp to u8
//#define ValueClamp(out, lu, ur, dl, rd) fminf (fmaxf (out, 0.0f), 255.0f); 
// value cannot be lower/higher then an original neighbour
#define ValueClamp(out, lu, ur, dl, rd) ClampMNEDI(out, lu, ur, dl, rd)


//  ABOUT
// Check for `bad rank(C)`, `alpha_min = 0`, `condition = inf`. Without it the algorithm might
// produce noise, ringing, jitter, halo and dark pixels.
// - Quite expensive!
//
bool ConditionNumber (
    IT mat16x4r32 REF C
) {
    Eigen::JacobiSVD<mat16x4r32> svd (C); // Singular Value Decomposition
    r32 condition = svd.singularValues ()(0) / svd.singularValues ().tail (1)(0);

    // ∞       > singular
    // 10⁶     > dangerous
    // 10³–10⁵ > marginal
    // 10–100  > good
    // 1       > perfect
    //
    const r32 CONDITION_NUM = 1e5f; 
            
    return (condition > CONDITION_NUM) || !std::isfinite (condition);
}


bool ConditionNumberN (
    IT mat REF C
) {
    Eigen::JacobiSVD<mat24x4r32> svd (C); // Singular Value Decomposition
    r32 condition = svd.singularValues ()(0) / svd.singularValues ().tail (1)(0);

    // ∞       > singular
    // 10⁶     > dangerous
    // 10³–10⁵ > marginal
    // 10–100  > good
    // 1       > perfect
    //
    const r32 CONDITION_NUM = 1e5f; 
            
    return (condition > CONDITION_NUM) || !std::isfinite (condition);
}


//  ABOUT
// NEDI assumes structure. With no structure regression is meaningless.
//  Quite cheap!
// 
bool VarianceHeuristic (
    IT mat16x4r32 REF C
) {
    // Flat regions -> variance ≈ 0
    // Edges        -> large variance
    // Noise        -> small variance
    //
    r32 variance = (C.array () - C.mean ()).square ().mean ();
    return variance < 1e-6f;
}


bool VarianceHeuristicN (
    IT mat REF C
) {
    // Flat regions -> variance ≈ 0
    // Edges        -> large variance
    // Noise        -> small variance
    //
    r32 variance = (C.array () - C.mean ()).square ().mean ();
    return variance < 1e-6f;
}


//  ABOUT
// Complexity -> O(n³), Numerical Stability -> worst.
// Explicit inverse can cause: ringing, diagonal artifacts, color overshoot.
//
void CalcCoeffsInverse (
    IN mat16x4r32 REF C, 
    IN vec16r32   REF Y,
    OT mat4x1r32  REF coeffs
) {
    coeffs = (C.transpose() * C).inverse() * (C.transpose() * Y);
}


//#define CoeffsCheck(C) ConditionNumber(C) || VarianceHeuristic(C)
#define CoeffsCheck(C) VarianceHeuristic(C)

//#define CoeffsCheckN(C) ConditionNumberN(C) || VarianceHeuristicN(C)
#define CoeffsCheckN(C) VarianceHeuristicN(C)


//  ABOUT
// Complexity -> O(n³/3), Numerical Stability -> ok.
//
void CalcCoeffsLDLT (
    IN mat16x4r32 REF C, 
    IN vec16r32   REF Y,
    OT mat4x1r32  REF coeffs
) {
    coeffs = (C.transpose() * C).ldlt().solve(C.transpose() * Y);
}


//  ABOUT
// Complexity -> O(2n³/3), Numerical Stability -> best.
//
void CalcCoeffsQR (
    IN mat16x4r32 REF C, 
    IN vec16r32   REF Y,
    OT mat4x1r32  REF coeffs
) {
    coeffs = C.colPivHouseholderQr().solve(Y);
}


void CalcCoeffsQRN (
    IN mat        REF C, 
    IN vec        REF Y,
    OT mat4x1r32  REF coeffs
) {
    coeffs = C.colPivHouseholderQr().solve(Y);
}


//  ABOUT
// It is necessary to clamp coeffs as without bad things happen.
//  Negative weights imply: ringing, overshoot, undershoot, diagonal halos, color clipping.
// 
void ClampCoeffs (
    IT mat4x1r32  REF coeffs
) {
    // In image interpolation pixels should add light, not subtract it.
    // That's mathematically legal, but physically wrong.
    //
    coeffs = coeffs.cwiseMax (0.0f);

    // Clamping breaks the scale. Brightness would be increased and energy 
    //  would not be conserved. Therefore normalize.
    //
    coeffs /= coeffs.sum ();
}


//  ABOUT
// todo
//
void SecondStepPointsAxis (
    IT u8*        CEF otImageData,
    //
    IN u16        REF width,
    IN u16        REF height,
    IN u8         REF channels,
    //
    IN u8         REF windowSizeX,
    IN u8         REF windowSizeY,
    //
    IN u16        REF iix,
    IN u16        REF iiy,
    IN u8         REF iChannel,
    //
    IN u8         REF axisX,
    IN u8         REF axisY,
    IN u8         REF wX,
    IN u8         REF wY,
    //
    IT vec16r32   REF Y,
    IT mat16x4r32 REF C
) {
    const u32 row = wY * windowSizeX + wX;

    const u64 scaledY = ((iiy - 2) * (width * 2) * channels * 2);
    const u64 scaledX = (iix * channels * 2);
    const u64 offsetY = (width * 2) * channels;
    const u64 offsetX = 1 * channels;

    const u64 initY = scaledY + (axisY * offsetY);
    const u64 initX = scaledX - offsetX + (axisX * offsetX);

    { // Y
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) + 
            iChannel;
        Y[row] = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [lu] (-1,-1)
        const u64 ii = 
            initY + (wX * offsetY) - offsetY + (wY * offsetY) + 
            initX + (wX * offsetX) - offsetX - (wY * offsetX) + 
            iChannel;
        C(row, 0) = ((r32)otImageData[ii]) / 255.0f;
    } 

    { // corner pixel [ru] (+1,-1)
        const u64 ii = 
            initY + (wX * offsetY) - offsetY + (wY * offsetY) + 
            initX + (wX * offsetX) + offsetX - (wY * offsetX) + 
            iChannel;
        C(row, 1) = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [ld] (-1,+1)
        const u64 ii = 
            initY + (wX * offsetY) + offsetY + (wY * offsetY) + 
            initX + (wX * offsetX) - offsetX - (wY * offsetX) + 
            iChannel;
        C(row, 2) = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [rd] (+1,+1)
        const u64 ii = 
            initY + (wX * offsetY) + offsetY + (wY * offsetY) + 
            initX + (wX * offsetX) + offsetX - (wY * offsetX) + 
            iChannel;
        C(row, 3) = ((r32)otImageData[ii]) / 255.0f;
    }
}


//  ABOUT
// todo
//
void SecondStepPointsDiagonal (
    IT u8*        CEF otImageData,
    //
    IN u16        REF width,
    IN u16        REF height,
    IN u8         REF channels,
    //
    IN u8         REF windowSizeX,
    IN u8         REF windowSizeY,
    //
    IN u16        REF iix,
    IN u16        REF iiy,
    IN u8         REF iChannel,
    //
    IN u8         REF axisX,
    IN u8         REF axisY,
    IN u8         REF wX,
    IN u8         REF wY,
    //
    IT vec16r32   REF Y,
    IT mat16x4r32 REF C
) {
    const u32 row = wY * windowSizeX + wX;

    const u64 scaledY = ((iiy - 2) * (width * 2) * channels * 2);
    const u64 scaledX = (iix * channels * 2);
    const u64 offsetY = (width * 2) * channels;
    const u64 offsetX = 1 * channels;

    const u64 initY = scaledY + (axisY * offsetY);
    const u64 initX = scaledX - offsetX + (axisX * offsetX);

    { // Y
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) + 
            iChannel;
        Y[row] = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [u] (+0,-2)
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) - (2 * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) + 
            iChannel;
        C(row, 0) = ((r32)otImageData[ii]) / 255.0f;
    } 

    { // corner pixel [r] (+2,+0)
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) + (2 * offsetX) + 
            iChannel;
        C(row, 1) = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [l] (-2,+0)
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) - (2 * offsetX) + 
            iChannel;
        C(row, 2) = ((r32)otImageData[ii]) / 255.0f;
    }

    { // corner pixel [d] (+0,+2)
        const u64 ii = 
            initY + (wX * offsetY) + (wY * offsetY) + (2 * offsetY) + 
            initX + (wX * offsetX) - (wY * offsetX) + 
            iChannel;
        C(row, 3) = ((r32)otImageData[ii]) / 255.0f;
    }
}

//old #define SecondStepPoints( \
//old     inImageData, width, height, channels, \
//old     windowSizeX, windowSizeY, iix, iiy, iChannel, \
//old     axisX, axisY, wX, wY, \
//old     Y, C \
//old ) SecondStepPointsAxis( \
//old     inImageData, width, height, channels, \
//old     windowSizeX, windowSizeY, iix, iiy, iChannel, \
//old     axisX, axisY, wX, wY, \
//old     Y, C \
//old )

#define SecondStepPoints( \
    inImageData, width, height, channels, \
    windowSizeX, windowSizeY, iix, iiy, iChannel, \
    axisX, axisY, wX, wY, \
    Y, C \
) SecondStepPointsDiagonal( \
    inImageData, width, height, channels, \
    windowSizeX, windowSizeY, iix, iiy, iChannel, \
    axisX, axisY, wX, wY, \
    Y, C \
)


namespace DIAGONAL_ONLY {

    //  ABOUT
    // It's the diagonal pixel so it is (A + B + C + D) / 4.
    //
    void Linear4 (
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        OT u8*        CEF out
    ) {
        const u64 scaledY = (iiy - 1) * width * channels;
        const u64 scaledX = (iix - 1) * channels;
        const u64 io = scaledY  + scaledX  + iChannel;
            
        const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel; // lu
        const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel; // ur
        const u64 ii2 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel; // dl
        const u64 ii3 = ((iiy - 0) * width * channels) + ((iix - 0) * channels) + iChannel; // rd
    
        r32 i0 = inImageData[ii0];
        r32 i1 = inImageData[ii1];
        r32 i2 = inImageData[ii2];
        r32 i3 = inImageData[ii3];
    
        r32 value = (i0 + i1 + i2 + i3) / 4.0f;
        out[io] = fminf (fmaxf (value, 0.0f), 255.0f); // clamp to u8
    }


    void WindowSingle (
        IT u8*        CEF otImageData,
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u8         REF windowSizeX,
        IN u8         REF windowSizeY,
        IN u8         REF windowOffsetX,
        IN u8         REF windowOffsetY,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        IT vec16r32   REF Y,
        IT mat16x4r32 REF C, 
        IT mat4x1r32  REF coeffs
    ) {

        for (u8 wy = 0; wy < windowSizeY; ++wy) {     // window y iterator
            for (u8 wx = 0; wx < windowSizeX; ++wx) { // window x iterator

                u32 row = wy * windowSizeX + wx;

                {   // Y-vector Assignment.
                    const u64 y = iiy - windowOffsetY + wy;
                    const u64 x = iix - windowOffsetX + wx;

                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    Y[row] = ((r32)inImageData[io]) / 255.0f;
                }

                    
                {   // C-matrix Assignment.

                    { // lu pixel
                        const u64 y = iiy - windowOffsetY + wy - 1;
                        const u64 x = iix - windowOffsetX + wx - 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 0) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // ur pixel
                        const u64 y = iiy - windowOffsetY + wy - 1;
                        const u64 x = iix - windowOffsetX + wx + 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 1) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // dl pixel
                        const u64 y = iiy - windowOffsetY + wy + 1;
                        const u64 x = iix - windowOffsetX + wx - 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 2) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // rd pixel
                        const u64 y = iiy - windowOffsetY + wy + 1;
                        const u64 x = iix - windowOffsetX + wx + 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 3) = ((r32)inImageData[io]) / 255.0f;
                    }

                }
            }
        }

        //{ // DEBUG -> Print Eigen Matrix
        //    std::stringstream ss;
        //    ss << C;
        //    LOGINFO ("C:\n %s\n", ss.str().c_str());
        //
        //    ss.str (std::string());  // clear the buffer
        //    ss.clear ();             // reset flags (VERY IMPORTANT)
        //
        //    ss << Y;
        //    LOGINFO ("Y:\n %s\n", ss.str().c_str());
        //}

        //{ // only for test
        //    const u64 scaledY = (iiy * (width * 2) * channels * 2);
        //    const u64 scaledX = (iix * channels * 2);
        //    const u64 offsetY = (width * 2) * channels;
        //    const u64 offsetX = 1 * channels;
        //
        //    const u64 io = scaledY - offsetY + scaledX - offsetX + iChannel;
        //    const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel; // dl
        //    const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel; // rd
        //    const u64 ii2 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel; // lu
        //    const u64 ii3 = ((iiy - 0) * width * channels) + ((iix - 0) * channels) + iChannel; // ur
        //    //LOGINFO ("0: %d, 1: %d, 2: %d, 3: %d\n", 
        //    //    inImageData[ii0], inImageData[ii1], inImageData[ii2], inImageData[ii3]
        //    //);
        //    otImageData[io] = 255;
        //}

        if (CoeffsCheck (C)) {
            Linear4 ( // fallback
                inImageData, 
                width, height, channels, 
                iix, iiy, iChannel, 
                otImageData
            );  
        } else {
            CalcCoeffsQR (C, Y, coeffs);
            ClampCoeffs (coeffs);
        
            const u64 scaledY = (iiy - 1) * width * channels;
            const u64 scaledX = (iix - 1) * channels;
            const u64 io = scaledY  + scaledX  + iChannel;
            
            const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel; // lu
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel; // ur
            const u64 ii2 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel; // dl
            const u64 ii3 = ((iiy - 0) * width * channels) + ((iix - 0) * channels) + iChannel; // rd

            auto&& i0 = inImageData[ii0];
            auto&& i1 = inImageData[ii1];
            auto&& i2 = inImageData[ii2];
            auto&& i3 = inImageData[ii3];

            r32 value = 
                coeffs[0] * i0 + 
                coeffs[1] * i1 + 
                coeffs[2] * i2 + 
                coeffs[3] * i3;

            ValueClamp (value, i0, i1, i2, i3);
            otImageData[io] = value; 
        }

    }


    void WindowSingleLUT (
        IT u8*        CEF otImageData,
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u8         REF windowOffsetX,
        IN u8         REF windowOffsetY,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        IT vec        REF Y,
        IT mat        REF C, 
        IT mat4x1r32  REF coeffs,
        //
        //IN u8* const * const& positions,
        IN u8         (*positions)[2],
        IN u8         REF positionsSize
    ) {
        for (u16 i = 0; i < positionsSize; ++i) {
            
            const u16 wx = positions[i][0];
            const u16 wy = positions[i][1];

            {   // Y-vector Assignment.
                const u64 y = iiy - windowOffsetY + wy;
                const u64 x = iix - windowOffsetX + wx;
                u64 io = (y * width * channels) + (x * channels) + iChannel;
                Y[i] = ((r32)inImageData[io]) / 255.0f;
            }

            {   // C-matrix Assignment.
        
                { // lu pixel
                    const u64 y = iiy - windowOffsetY + wy - 1;
                    const u64 x = iix - windowOffsetX + wx - 1;
                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    C(i, 0) = ((r32)inImageData[io]) / 255.0f;
                }
        
                { // ur pixel
                    const u64 y = iiy - windowOffsetY + wy - 1;
                    const u64 x = iix - windowOffsetX + wx + 1;
                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    C(i, 1) = ((r32)inImageData[io]) / 255.0f;
                }
        
                { // dl pixel
                    const u64 y = iiy - windowOffsetY + wy + 1;
                    const u64 x = iix - windowOffsetX + wx - 1;
                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    C(i, 2) = ((r32)inImageData[io]) / 255.0f;
                }
        
                { // rd pixel
                    const u64 y = iiy - windowOffsetY + wy + 1;
                    const u64 x = iix - windowOffsetX + wx + 1;
                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    C(i, 3) = ((r32)inImageData[io]) / 255.0f;
                }
        
            }
        }

        if (CoeffsCheckN (C)) {
            Linear4 ( // fallback
                inImageData, 
                width, height, channels, 
                iix, iiy, iChannel, 
                otImageData
            );  
        } else {
            CalcCoeffsQRN (C, Y, coeffs);
            ClampCoeffs (coeffs);
        
            const u64 scaledY = (iiy - 1) * width * channels;
            const u64 scaledX = (iix - 1) * channels;
            const u64 io = scaledY  + scaledX  + iChannel;
            
            const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel; // lu
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel; // ur
            const u64 ii2 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel; // dl
            const u64 ii3 = ((iiy - 0) * width * channels) + ((iix - 0) * channels) + iChannel; // rd

            auto&& i0 = inImageData[ii0];
            auto&& i1 = inImageData[ii1];
            auto&& i2 = inImageData[ii2];
            auto&& i3 = inImageData[ii3];

            r32 value = 
                coeffs[0] * i0 + 
                coeffs[1] * i1 + 
                coeffs[2] * i2 + 
                coeffs[3] * i3;

            ValueClamp (value, i0, i1, i2, i3);
            otImageData[io] = value; 
        }

    }


    void LinearFill (
        IT u8* CEF otImageData,
        IN u8* CEF inImageData,
        //
        IN u16 REF width,
        IN u16 REF height,
        IN u8  REF channels,
        //
        IN u8  REF cornerOffsetX,
        IN u8  REF cornerOffsetY,
        //
        IN u8  REF iChannel
    ) {

        // TOP
        for (u16 iiy = 1; iiy < cornerOffsetY; ++iiy) {
            for (u16 iix = 1; iix < width; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // BOT
        for (u16 iiy = height - cornerOffsetY; iiy < height; ++iiy) {
            for (u16 iix = 1; iix < width; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // LFT
        for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {
            for (u16 iix = 1; iix < cornerOffsetX; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // RTH
        for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {
            for (u16 iix = width - cornerOffsetX; iix < width; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        { // alone right-bottom pixel
            const u64 scaledY = ((height - 1) * width * channels);
            const u64 scaledX = ((width - 1) * channels);
            //const u64 offsetY = (width * 2) * channels;
            //const u64 offsetX = 1 * channels;

            const u64 io = scaledY + scaledX + iChannel;
            const u64 ii = ((height - 1) * width * channels) + ((width - 1) * channels) + iChannel;

            otImageData[io] = inImageData[ii];
        }

        // alone stripe-y
        for (u16 iiy = 1; iiy < height; ++iiy) {

            const u64 iix = width;
            const u64 scaledY = ((iiy - 1) * width * channels);
            const u64 scaledX = ((iix - 1) * channels);
            const u64 io = scaledY + scaledX + iChannel;

            const u64 ii0 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel;
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel;

            r32 i0 = inImageData[ii0];
            r32 i1 = inImageData[ii1];

            auto out = (i0 + i1) / 2.0f;
            out = fminf (fmaxf (out, 0.0f), 255.0f); // clamp to u8

            otImageData[io] = out;
        }

        // alone stripe-x
        for (u16 iix = 1; iix < width; ++iix) {

            const u64 iiy = height;
            const u64 scaledY = ((iiy - 1) * width * channels);
            const u64 scaledX = ((iix - 1) * channels);
            const u64 io = scaledY + scaledX + iChannel;

            const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel;
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel;

            r32 i0 = inImageData[ii0];
            r32 i1 = inImageData[ii1];

            auto out = (i0 + i1) / 2.0f;
            out = fminf (fmaxf (out, 0.0f), 255.0f); // clamp to u8

            otImageData[io] = out;
        }
    
    }

}


namespace DIAGONAL {

    //  ABOUT
    // It's the diagonal pixel so it is (A + B + C + D) / 4.
    //
    void Linear4 (
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        OT u8*        CEF out
    ) {

        const u64 scaledY = (iiy * (width * 2) * channels * 2);
        const u64 scaledX = (iix * channels * 2);
        const u64 offsetY = (width * 2) * channels;
        const u64 offsetX = 1 * channels;

        const u64 io = scaledY - offsetY + scaledX - offsetX + iChannel;

        const u64 ii0 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel;
        const u64 ii1 = ((iiy - 0) * width * channels) + ((iix - 0) * channels) + iChannel;
        const u64 ii2 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel;
        const u64 ii3 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel;

        r32 i0 = inImageData[ii0];
        r32 i1 = inImageData[ii1];
        r32 i2 = inImageData[ii2];
        r32 i3 = inImageData[ii3];

        auto value = (i0 + i1 + i2 + i3) / 4.0f;
        out[io] = fminf (fmaxf (value, 0.0f), 255.0f); // clamp to u8
    }

    // +0,+0; +1,+0; +2,+0; +3,+0;
    // +0,+1; +1,+1; +2,+1; +3,+1;
    // +0,+2; +1,+2; +2,+2; +3,+2;
    // +0,+3; +1,+3; +2,+3; +3,+3;
    //
    // vs
    //
    //
    //        +1,+0; +2,+0; 
    // +0,+1; +1,+1; +2,+1; +3,+1;
    // +0,+2; +1,+2; +2,+2; +3,+2;
    //        +1,+3; +2,+3; 

    void WindowSingle (
        IT u8*        CEF otImageData,
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u8         REF windowSizeX,
        IN u8         REF windowSizeY,
        IN u8         REF windowOffsetX,
        IN u8         REF windowOffsetY,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        IT vec16r32   REF Y,
        IT mat16x4r32 REF C, 
        IT mat4x1r32  REF coeffs
    ) {

        for (u8 wy = 0; wy < windowSizeY; ++wy) {     // window y iterator
            for (u8 wx = 0; wx < windowSizeX; ++wx) { // window x iterator

                u32 row = wy * windowSizeX + wx;

                {   // Y-vector Assignment.
                    const u64 y = iiy - windowOffsetY + wy;
                    const u64 x = iix - windowOffsetX + wx;

                    u64 io = (y * width * channels) + (x * channels) + iChannel;
                    Y[row] = ((r32)inImageData[io]) / 255.0f;
                }

                    
                {   // C-matrix Assignment.

                    { // lu pixel
                        const u64 y = iiy - windowOffsetY + wy - 1;
                        const u64 x = iix - windowOffsetX + wx - 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 0) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // ur pixel
                        const u64 y = iiy - windowOffsetY + wy - 1;
                        const u64 x = iix - windowOffsetX + wx + 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 1) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // dl pixel
                        const u64 y = iiy - windowOffsetY + wy + 1;
                        const u64 x = iix - windowOffsetX + wx - 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 2) = ((r32)inImageData[io]) / 255.0f;
                    }

                    { // rd pixel
                        const u64 y = iiy - windowOffsetY + wy + 1;
                        const u64 x = iix - windowOffsetX + wx + 1;

                        u64 io = (y * width * channels) + (x * channels) + iChannel;
                        C(row, 3) = ((r32)inImageData[io]) / 255.0f;
                    }

                }
            }
        }

        if (CoeffsCheck (C)) {
            Linear4 ( // fallback
                inImageData, 
                width, height, channels, 
                iix + 1, iiy + 1, iChannel, 
                otImageData
            );  
        } else {
            CalcCoeffsQR (C, Y, coeffs);
            ClampCoeffs (coeffs);
        
            const u64 scaledY = ((iiy) * (width * 2) * channels * 2);
            const u64 scaledX = ((iix) * channels * 2);
            const u64 offsetY = (width * 2) * channels;
            const u64 offsetX = 1 * channels;
            const u64 io = scaledY + offsetY + scaledX + offsetX + iChannel;  

            const u64 ii0 = ((iiy + 0) * width * channels) + ((iix + 0) * channels) + iChannel; // lu
            const u64 ii1 = ((iiy + 0) * width * channels) + ((iix + 1) * channels) + iChannel; // ur
            const u64 ii2 = ((iiy + 1) * width * channels) + ((iix + 0) * channels) + iChannel; // dl
            const u64 ii3 = ((iiy + 1) * width * channels) + ((iix + 1) * channels) + iChannel; // rd 

            auto&& i0 = inImageData[ii0];
            auto&& i1 = inImageData[ii1];
            auto&& i2 = inImageData[ii2];
            auto&& i3 = inImageData[ii3];

            auto value = 
                coeffs[0] * i0 + 
                coeffs[1] * i1 + 
                coeffs[2] * i2 + 
                coeffs[3] * i3;

            ValueClamp (value, i0, i1, i2, i3);
            otImageData[io] = value; 
        }

    }


    void LinearFill (
        IT u8* CEF otImageData,
        IN u8* CEF inImageData,
        //
        IN u16 REF width,
        IN u16 REF height,
        IN u8  REF channels,
        //
        IN u8  REF cornerOffsetX,
        IN u8  REF cornerOffsetY,
        //
        IN u8  REF iChannel
    ) {

        // TOP
        for (u16 iiy = 1; iiy < cornerOffsetY; ++iiy) {
            for (u16 iix = 1; iix < width; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // BOT
        for (u16 iiy = height - cornerOffsetY; iiy < height; ++iiy) {
            for (u16 iix = 1; iix < width; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // LFT
        for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {
            for (u16 iix = 1; iix < cornerOffsetX; ++iix) {
                Linear4 (
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        // RTH
        for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {
            for (u16 iix = width - cornerOffsetX; iix < width; ++iix) {
                Linear4 ( 
                    inImageData, 
                    width, height, channels, 
                    iix, iiy, iChannel, 
                    otImageData
                );
            }
        }

        { // alone right-bottom pixel
            const u64 scaledY = ((height) * (width * 2) * channels * 2);
            const u64 scaledX = ((width) * channels * 2);
            const u64 offsetY = (width * 2) * channels;
            const u64 offsetX = 1 * channels;

            const u64 io = scaledY - offsetY + scaledX - offsetX + iChannel;
            const u64 ii = ((height - 1) * width * channels) + ((width - 1) * channels) + iChannel;

            otImageData[io] = inImageData[ii];
        }

        // alone stripe-y
        for (u16 iiy = 1; iiy < height; ++iiy) {

            const u64 iix = width;
            const u64 scaledY = (iiy * (width * 2) * channels * 2);
            const u64 scaledX = (iix * channels * 2);
            const u64 offsetY = (width * 2) * channels;
            const u64 offsetX = 1 * channels;

            const u64 io = scaledY - offsetY + scaledX - offsetX + iChannel;

            const u64 ii0 = ((iiy - 0) * width * channels) + ((iix - 1) * channels) + iChannel;
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel;

            r32 i0 = inImageData[ii0];
            r32 i1 = inImageData[ii1];

            auto out = (i0 + i1) / 2.0f;
            out = fminf (fmaxf (out, 0.0f), 255.0f); // clamp to u8

            otImageData[io] = out;
        }

        // alone stripe-x
        for (u16 iix = 1; iix < width; ++iix) {

            const u64 iiy = height;
            const u64 scaledY = (iiy * (width * 2) * channels * 2);
            const u64 scaledX = (iix * channels * 2);
            const u64 offsetY = (width * 2) * channels;
            const u64 offsetX = 1 * channels;

            const u64 io = scaledY - offsetY + scaledX - offsetX + iChannel;

            const u64 ii0 = ((iiy - 1) * width * channels) + ((iix - 0) * channels) + iChannel;
            const u64 ii1 = ((iiy - 1) * width * channels) + ((iix - 1) * channels) + iChannel;

            r32 i0 = inImageData[ii0];
            r32 i1 = inImageData[ii1];

            auto out = (i0 + i1) / 2.0f;
            out = fminf (fmaxf (out, 0.0f), 255.0f); // clamp to u8

            otImageData[io] = out;
        }

    }

}


namespace AXIS {

    //  TODO
    // It's not diagonal pixel! So it's not (A + B + C + D) / 4.
    // it should be either A + B / 2 or A + C / 2.
    //
    void Linear4 (
        IN u8*        CEF otImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        IN u8         REF axisX,
        IN u8         REF axisY,
        //
        OT u8*        CEF out
    ) {

        const u64 scaledY = (iiy * (width * 2) * channels * 2);
        const u64 scaledX = (iix * channels * 2);
        const u64 offsetY = (width * 2) * channels;
        const u64 offsetX = 1 * channels;

        const u64 initY = scaledY - (axisY * offsetY) - offsetY;
        const u64 initX = scaledX - (axisX * offsetX) - offsetX;
        const u64 ii = initY + initX + iChannel;

        const u64 ii0 = initY - offsetY + initX + iChannel; // up
        const u64 ii1 = initY + initX + offsetX + iChannel; // right
        const u64 ii2 = initY + offsetY + initX + iChannel; // down
        const u64 ii3 = initY + initX - offsetX + iChannel; // left

        r32 i0 = otImageData[ii0];
        r32 i1 = otImageData[ii1];
        r32 i2 = otImageData[ii2];
        r32 i3 = otImageData[ii3];

        auto value = (i0 + i1 + i2 + i3) / 4.0f;
        out[ii] = fminf (fmaxf (value, 0.0f), 255.0f); // clamp to u8
        //out[ii] = 255.0f;
    }


    void WindowSingle (
        IT u8*        CEF otImageData,
        IN u8*        CEF inImageData,
        //
        IN u16        REF width,
        IN u16        REF height,
        IN u8         REF channels,
        //
        IN u8         REF windowSizeX,
        IN u8         REF windowSizeY,
        IN u8         REF windowOffsetY,
        IN u8         REF windowOffsetX,
        //
        IN u16        REF iix,
        IN u16        REF iiy,
        IN u8         REF iChannel,
        //
        IN u8         REF axisX,
        IN u8         REF axisY,
        //
        IT vec16r32   REF Y,
        IT mat16x4r32 REF C, 
        IT mat4x1r32  REF coeffs
    ) {
        for (u8 wY = 0; wY < windowSizeY; ++wY) {
            for (u8 wX = 0; wX < windowSizeX; ++wX) {
                SecondStepPoints (
                    otImageData, 
                    width, height, channels,
                    windowSizeX, windowSizeY, 
                    iix, iiy, iChannel,
                    axisX, axisY, 
                    wX, wY,
                    Y, C
                );
            }
        }

        //{ // DEBUG -> Print Eigen Matrix
        //    std::stringstream ss;
        //    ss << C;
        //    LOGINFO ("C:\n %s\n", ss.str().c_str());
        //
        //    ss.str (std::string());  // clear the buffer
        //    ss.clear ();             // reset flags (VERY IMPORTANT)
        //
        //    ss << Y;
        //    LOGINFO ("Y:\n %s\n", ss.str().c_str());
        //}

        if (CoeffsCheck (C)) {
            Linear4 ( // fallback
                otImageData, 
                width, height, channels, 
                iix + 1, iiy + 1, iChannel, 
                axisX, axisY, 
                otImageData
            );  
        } else {
            CalcCoeffsQR (C, Y, coeffs);
            ClampCoeffs (coeffs);
        
            const u64 scaledY = (iiy * (width * 2) * channels * 2);
            const u64 scaledX = (iix * channels * 2);
            const u64 offsetY = (width * 2) * channels;
            const u64 offsetX = 1 * channels;

            const u64 initY = scaledY + (axisY * offsetY) - offsetY;
            const u64 initX = scaledX + (axisX * offsetX) - offsetX;
            const u64 ii = initY + initX + iChannel;
        
            const u64 ii0 = initY - offsetY + initX + iChannel; // up
            const u64 ii1 = initY + initX + offsetX + iChannel; // right
            const u64 ii2 = initY + initX - offsetX + iChannel; // left
            const u64 ii3 = initY + offsetY + initX + iChannel; // down
        
            auto&& i0 = otImageData[ii0];
            auto&& i1 = otImageData[ii1];
            auto&& i2 = otImageData[ii2];
            auto&& i3 = otImageData[ii3];
        
            auto value = 
                coeffs[0] * i0 + 
                coeffs[1] * i1 + 
                coeffs[2] * i2 + 
                coeffs[3] * i3;
        
            ValueClamp (value, i0, i1, i2, i3);
            otImageData[ii] = value; 
        }
    }

}


void DiagonalOnlyNEDI_A (
    IT u8*                   CEF otImageData,
    IN u8*                   CEF inImageData,
    IN IMAGE::PACKED::Meta32 REF inImageMeta
) {
    REG u8 channels; REG u16 height; REG u16 width; 
    IMAGE::PACKED::Get (width, height, channels, inImageMeta);

    //  ABOUT
    // Window must fit in the image itself. so we don't read out of boundary. 
    //  Diagonal and axis pixels must fit in the source image also.
    // 
    const u8 windowSizeX    = 4;
    const u8 windowSizeY    = 4;
    const u8 windowHalfY    = windowSizeY / 2;
    const u8 windowHalfX    = windowSizeX / 2;
    const u8 cornerOffsetY  = windowHalfY + 1;
    const u8 cornerOffsetX  = windowHalfX + 1;

    //  ABOUT
    // Y-vector contains the values of the pixels contained on the window W.
    //
    //alignas (16 * 4) r32 bufferY[16]; vec16r32 Y (bufferY);
    alignas (16 * 4) r32 bufferY[16]; vec12r32 Y (bufferY);

    //  ABOUT
    // C-matrix contains in each row the values of the four diagonal neighbors of each pixel listed on Y-vector.
    //
    //alignas (16 * 4 * 4) r32 bufferC[16 * 4]; mat16x4r32 C (bufferC);
    alignas (16 * 4 * 4) r32 bufferC[16 * 4]; mat12x4r32 C (bufferC);

    //  ABOUT
    // coeffs-vector. Each coefficient answers: "How much should this neighbor contribute to the unknown diagonal pixel?"
    //
    alignas (4 * 4) r32 bufferCoeffs[4]; mat4x1r32 coeffs (bufferCoeffs);

    { // second-step -> diagonal

        mat anyC = C;
        vec anyY = Y;

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) {
            for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {    // image y iterator
                for (u16 iix = cornerOffsetX; iix < (width - cornerOffsetX); ++iix) { // image x iterator
                    DIAGONAL_ONLY::WindowSingleLUT (
                        otImageData, inImageData, 
                        width, height, channels, 
                        windowHalfX, windowHalfY,
                        iix, iiy, iChannel,
                        anyY, anyC, coeffs,
                        WINDOW_DIAGONAL_POSITIONS_CIRCULAR_4, 
                        WINDOW_DIAGONAL_POSITIONS_CIRCULAR_4_SIZE
                    );
                }
            }
            LOGINFO ("diagonal-channel-%d: finished\n", iChannel);
        }

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) { // Fill missing diagonal   
            DIAGONAL_ONLY::LinearFill (
                otImageData, inImageData,
                width, height, channels,
                cornerOffsetX, cornerOffsetY,
                iChannel
            );
            LOGINFO ("diagonal-fill-channel-%d: finished\n", iChannel);
        }
    }
}


void DiagonalOnlyNEDI_B (
    IT u8*                   CEF otImageData,
    IN u8*                   CEF inImageData,
    IN IMAGE::PACKED::Meta32 REF inImageMeta
) {
    REG u8 channels; REG u16 height; REG u16 width; 
    IMAGE::PACKED::Get (width, height, channels, inImageMeta);

    //  ABOUT
    // Window must fit in the image itself. so we don't read out of boundary. 
    //  Diagonal and axis pixels must fit in the source image also.
    // 
    const u8 windowSizeX    = 6;
    const u8 windowSizeY    = 6;
    const u8 windowHalfY    = windowSizeY / 2;
    const u8 windowHalfX    = windowSizeX / 2;
    const u8 cornerOffsetY  = windowHalfY + 1;
    const u8 cornerOffsetX  = windowHalfX + 1;

    //  ABOUT
    // Y-vector contains the values of the pixels contained on the window W.
    //
    //alignas (16 * 4) r32 bufferY[16]; vec16r32 Y (bufferY);
    alignas (32 * 4) r32 bufferY[24]; vec24r32 Y (bufferY);

    //  ABOUT
    // C-matrix contains in each row the values of the four diagonal neighbors of each pixel listed on Y-vector.
    //
    //alignas (16 * 4 * 4) r32 bufferC[16 * 4]; mat16x4r32 C (bufferC);
    alignas (32 * 4 * 4) r32 bufferC[24 * 4]; mat24x4r32 C (bufferC);

    //  ABOUT
    // coeffs-vector. Each coefficient answers: "How much should this neighbor contribute to the unknown diagonal pixel?"
    //
    alignas (4 * 4) r32 bufferCoeffs[4]; mat4x1r32 coeffs (bufferCoeffs);

    { // second-step -> diagonal

        mat anyC = C;
        vec anyY = Y;

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) {
            for (u16 iiy = cornerOffsetY; iiy < (height - cornerOffsetY); ++iiy) {    // image y iterator
                for (u16 iix = cornerOffsetX; iix < (width - cornerOffsetX); ++iix) { // image x iterator
                    DIAGONAL_ONLY::WindowSingleLUT (
                        otImageData, inImageData, 
                        width, height, channels, 
                        windowHalfX, windowHalfY,
                        iix, iiy, iChannel,
                        anyY, anyC, coeffs,
                        WINDOW_DIAGONAL_POSITIONS_CIRCULAR_6, 
                        WINDOW_DIAGONAL_POSITIONS_CIRCULAR_6_SIZE
                    );
                }
            }
            LOGINFO ("diagonal-channel-%d: finished\n", iChannel);
        }

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) { // Fill missing diagonal   
            DIAGONAL_ONLY::LinearFill (
                otImageData, inImageData,
                width, height, channels,
                cornerOffsetX, cornerOffsetY,
                iChannel
            );
            LOGINFO ("diagonal-fill-channel-%d: finished\n", iChannel);
        }
    }
}


void FullNEDI (
    IT u8*                   CEF otImageData,
    IN u8*                   CEF inImageData,
    IN IMAGE::PACKED::Meta32 REF inImageMeta
) {

    REG u8 channels; REG u16 height; REG u16 width; 
    IMAGE::PACKED::Get (width, height, channels, inImageMeta);

    //  ABOUT
    // Window must fit in the image itself. so we don't read out of boundary. 
    //  Diagonal and axis pixels must fit in the source image also.
    // 
    const u8 windowSizeX    = 4;
    const u8 windowSizeY    = 4;
    const u8 windowHalfY    = windowSizeY / 2;
    const u8 windowHalfX    = windowSizeX / 2;
    const u8 cornerOffsetY  = windowHalfY + 1;
    const u8 cornerOffsetX  = windowHalfX + 1;
    const u8 offsetLeftY    = cornerOffsetY - windowHalfY;
    const u8 offsetLeftX    = cornerOffsetX - windowHalfX;

    //  ABOUT
    // Y-vector contains the values of the pixels contained on the window W.
    //
    alignas (16 * 4) r32 bufferY[16]; vec16r32 Y (bufferY);

    //  ABOUT
    // C-matrix contains in each row the values of the four diagonal neighbors of each pixel listed on Y-vector.
    //
    alignas (16 * 4 * 4) r32 bufferC[16 * 4]; mat16x4r32 C (bufferC);
        
    //  ABOUT
    // coeffs-vector. Each coefficient answers: "How much should this neighbor contribute to the unknown diagonal pixel?"
    //
    alignas (4 * 4) r32 bufferCoeffs[4]; mat4x1r32 coeffs (bufferCoeffs);

    { // 1st-step DIAGONAL [1,1]

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) {
            for (u16 iiy = windowHalfY; iiy < (height - cornerOffsetY); ++iiy) {    // image y iterator
                for (u16 iix = windowHalfX; iix < (width - cornerOffsetX); ++iix) { // image x iterator
                    DIAGONAL::WindowSingle (
                        otImageData, inImageData, 
                        width, height, channels, 
                        windowSizeX, windowSizeY, offsetLeftX, offsetLeftY,
                        iix, iiy, iChannel,
                        Y, C, coeffs
                    );
                }
            }
            LOGINFO ("diagonal-channel-%d: finished\n", iChannel);
        }
        
        for (u8 iChannel = 0; iChannel < channels; ++iChannel) { // Fill missing diagonal   
            DIAGONAL::LinearFill (
                otImageData, inImageData,
                width, height, channels,
                cornerOffsetX, cornerOffsetY,
                iChannel
            );
            LOGINFO ("diagonal-fill-channel-%d: finished\n", iChannel);
        }
    }

    { // 2nd-step AXIS [1,0] & AXIS [0,1]

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) {
            for (u32 iiy = cornerOffsetY; iiy < (height - windowHalfY); ++iiy) {
                for (u32 iix = cornerOffsetX; iix < (width - windowHalfX); ++iix) {
                    AXIS::WindowSingle (
                        otImageData, inImageData, 
                        width, height, channels, 
                        windowSizeX, windowSizeY, offsetLeftX, offsetLeftY,
                        iix, iiy, iChannel,
                        0, 1,
                        Y, C, coeffs
                    );
                }
            }
            LOGINFO ("axis-1-channel-%d: finished\n", iChannel);
        }

        for (u8 iChannel = 0; iChannel < channels; ++iChannel) {
            for (u32 iiy = cornerOffsetY; iiy < (height - windowHalfY); ++iiy) {
                for (u32 iix = cornerOffsetX; iix < (width - windowHalfX); ++iix) {
                    AXIS::WindowSingle (
                        otImageData, inImageData, 
                        width, height, channels, 
                        windowSizeX, windowSizeY, offsetLeftX, offsetLeftY,
                        iix, iiy, iChannel,
                        1, 0,
                        Y, C, coeffs
                    );
                }
            }
            LOGINFO ("axis-2-channel-%d: finished\n", iChannel);
        }

        //  TODO
        // Axis fill
        //
        {}
    }
    
    

}


// wrong, adds offset to final nedi image.
//#define DOWNSAMPLE_TYPE "bicubic\\" 

// correct, but it has to be nn-left-top variant.
#define DOWNSAMPLE_TYPE "nearest\\" 


const c8 FILEPATH_DEFAULT [] = "res\\" DOWNSAMPLE_TYPE "0_320x320x3.png";
//const c8 FILEPATH_DEFAULT [] = "res\\" DOWNSAMPLE_TYPE "1_320x320x3.png";
//const c8 FILEPATH_DEFAULT [] = "res\\" DOWNSAMPLE_TYPE "2_320x320x3.png";
//const c8 FILEPATH_DEFAULT [] = "res\\" DOWNSAMPLE_TYPE "3_320x320x3.png";
//const c8 FILEPATH_DEFAULT [] = "res\\" DOWNSAMPLE_TYPE "4_320x320x3.png";


s32 main (s32 argumentsCount, c8* arguments[]) {

    BINIT ("Executing: NEDI\n");

    IMAGE::PACKED::Meta32 otImageMeta;
    u8* otImageData;
    IMAGE::PACKED::Meta32 inImageMeta;
    u8* inImageData;

    c8* inFilepath;
    u8 isFullNedi;

    // --- Args interpreter --- Issue -> Hardcoded
        switch (argumentsCount) { 

            case 3: {
                const u8 argIsFullNedi = 1;
                const u8 argOtFilepath = 2;

                LOGINFO ("[%d]-arg: %s\n", argIsFullNedi - 1, arguments[argIsFullNedi]);
                LOGINFO ("[%d]-arg: %s\n", argOtFilepath - 1, arguments[argOtFilepath]);

                isFullNedi = atoi(arguments[argIsFullNedi]);
                inFilepath = arguments[argOtFilepath];
            } break;

            default:
                LOGWARN ("Invalid number of arguments passed!\n");
            case 1: { // Apply DEFAULT values for arguments.
                inFilepath = (c8*)(void*)FILEPATH_DEFAULT;
                isFullNedi = false;
            };

        }
    // ---

    // Start runtime timer
        TIMESTAMP::Timestamp clock = TIMESTAMP::GetCurrent ();
    //


    { // Load input image metadata & data.
        IMAGE::PACKED::Load (inImageMeta, inImageData, inFilepath);
    }


    //DEBUG (DEBUG_FLAG_LOGGING) { // --- cuda-min-test -> required for debugger to return a success.
    //    const dim3 threads (16, 16);
    //    const dim3 blocks  (320 / 16, 320 / 16);
    //
    //    CudaDummy <<<blocks, threads>>> (
    //        320, 320
    //    );
    //} // ---


    if (isFullNedi) { // NEDI-ALL

        { // Create output image buffer.
            REG u8 channels; REG u16 height; REG u16 width; 
            IMAGE::PACKED::Get (width, height, channels, inImageMeta);

            LOGINFO ("image-in: w: %d, h: %d, c: %d\n", width, height, channels);

            height *= 2;
            width *= 2;

            u64 size = width * height * channels;

            IMAGE::PACKED::Set (otImageMeta, width, height, channels);
            ALLOCATE (u8, otImageData, size);
        }
    
        { // COPY (required)
            REG u8 channels; REG u16 height; REG u16 width; 
            IMAGE::PACKED::Get (width, height, channels, inImageMeta);
        
            for (u8 iC = 0; iC < channels; ++iC) {
                for (u16 iH = 0; iH < height; ++iH) {
                    for (u16 iW = 0; iW < width; ++iW) {
                        auto io = (iH * width * channels * 4) + (iW * channels * 2) + iC;
                        auto ii = (iH * width * channels) + (iW * channels) + iC;
                        otImageData[io] = inImageData[ii];
                    }
                }
            }
        }
    
        FullNEDI (otImageData, inImageData, inImageMeta);

        { // Create an output image in the filesystem.
            IMAGE::PNG::Save (otImageMeta, otImageData, IMAGE_FILEPATH_O1);
        }

    } else { // NEDI-DIAGONAL-ONLY

        // --- Create output image buffer.
            REG u8 channels; REG u16 height; REG u16 width; 
            IMAGE::PACKED::Get (width, height, channels, inImageMeta);
            
            LOGINFO ("image-in: w: %d, h: %d, c: %d\n", width, height, channels);
            
            u64 size = width * height * channels;
            
            IMAGE::PACKED::Set (otImageMeta, width, height, channels);
            ALLOCATE (u8, otImageData, size);
        // ---

        DiagonalOnlyNEDI_B (otImageData, inImageData, inImageMeta);

        { // Create an output image in the filesystem.
            IMAGE::PNG::Save (otImageMeta, otImageData, IMAGE_FILEPATH_O0);
        }
    }

    { // Free output image data.
        REG u8 channels; REG u16 height; REG u16 width; 
        IMAGE::PACKED::Get (width, height, channels, otImageMeta);
        REG u64 size = width * height * channels;
        FREE (size, otImageData);
    }

    IMAGE::PACKED::Free (inImageData);

    
    BSTOP ("Finalizing: NEDI\n");

    { // End runtime timer
        fprintf (stdout, "[" LOGGER_TIME_FORMAT "]", TIMESTAMP::GetElapsed (clock)); \
        fprintf (stdout, " NEDI-done\n");
    }

	return 0;
}
