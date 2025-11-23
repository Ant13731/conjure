//
//  TurboLUT.metal
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-22.
//

#include <metal_stdlib>
using namespace metal;

// Small struct that holds parameters we pass from Swift.
struct Params {
    float minDepth;    // minimum depth value to map (meters)
    float maxDepth;    // maximum depth value to map (meters)
};

// Kernel function — runs once per output pixel (gid = grid coordinate).
//kernel void depthToTurboKernel(
//    texture2d<float, access::sample> depthTex        [[texture(0)]], // input depth texture (r16float)
//    texture1d<float, access::sample> turboLUT       [[texture(1)]], // 2D LUT texture (rgba8Unorm)
//    texture2d<float, access::write> outTex          [[texture(2)]], // output RGBA8 texture (BGRA order handled in Swift or shader)
//    constant Params &params                           [[buffer(0)]],  // uniform params (min/max depth)
//    uint2 gid                                         [[thread_position_in_grid]] // pixel coords
//) {
//    // Query texture dimensions to bounds-check the thread coordinate
//    uint width = depthTex.get_width();
//    uint height = depthTex.get_height();
//    if (gid.x >= width || gid.y >= height) return;
//
//    // Sample depth at integer pixel coordinates. We use a pixel sampler (nearest).
////    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
////    float depth = depthTex.sample(s, uint2(gid.x, gid.y)).r;
//    float depth = depthTex.read(gid).r;
//
//    // If depth is not valid, write a placeholder (black/transparent)
//    if (!isfinite(depth) || depth <= 0.0) {
//        uint c = uint(255); // RGBA (0..255)
//        outTex.write(c, gid);
//        return;
//    }
//
//    // Normalize depth -> 0..1 using min/max from Swift
//    float norm = (depth - params.minDepth) / (params.maxDepth - params.minDepth);
//    norm = clamp(norm, 0.0, 1.0);
//    uint lutIndex = uint(norm*255.0);
//
//    // Sample the 1D Turbo LUT using normalized coordinate (0..1).
//    // We sample with normalized coords and linear filtering for smooth interpolation.
////    constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
////    float4 lutColorF = turboLUT.sample(lutSampler, norm); // float4 in 0..1
//    ushort4 lutColorF = turboLUT.read(lutIndex); // float4 in 0..1
//
//    // Convert sampled color to 0..255 and BGRA ordering (because CVPixelBuffer uses BGRA)
////    uint r = uint(lutColorF.r * 255.0 + 0.5);
////    uint g = uint(lutColorF.g * 255.0 + 0.5);
////    uint b = uint(lutColorF.b * 255.0 + 0.5);
////    uint a = 255;
//
//    // Pack into one 32-bit integer (BGRA order)
//    uint outColor = (lutColorF.b << 24) | (lutColorF.g << 16) | (lutColorF.r << 8) | lutColorF.a;
////    uint outColor = (b << 24) | (g << 16) | (r << 8) | a;
////    uint outColor = uint( (uint)(lutColorF.b * 255.0 + 0.5),
////                              (uint)(lutColorF.g * 255.0 + 0.5),
////                              (uint)(lutColorF.r * 255.0 + 0.5),
////                              (uint)255 );
//    outTex.write(outColor, gid);
//}
// GPU kernel: depth -> Turbo BGRA8
kernel void depthToTurboKernel(
    texture2d<float, access::read> depthTex [[ texture(0) ]],
    texture2d<float, access::sample> turboLUT [[ texture(1) ]],
    texture2d<float, access::write> outTex [[ texture(2) ]],
    constant Params &params [[ buffer(0) ]],
    uint2 gid [[ thread_position_in_grid ]]
//                               ,
//    sampler s [[ sampler(0) ]]
)
{
    if (gid.x >= depthTex.get_width() || gid.y >= depthTex.get_height()) return;

    // 1. Read depth (float)
    float depth = depthTex.read(gid).r;

    // 2. Normalize to [0,1]
    float t = clamp((depth - params.minDepth) / (params.maxDepth - params.minDepth), 0.0, 1.0);

    // 3. Map to LUT index (0..255)
    uint idx = uint(t * 255.0);

    // 4. Sample Turbo LUT (RGBA8Unorm) → float4 in [0,1]
    float4 color = turboLUT.read(uint2(idx, 0));

    // 5. Write output (GPU converts float4 → BGRA8 automatically)
    outTex.write(color, gid);
}
